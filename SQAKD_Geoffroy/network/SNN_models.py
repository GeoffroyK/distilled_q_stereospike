import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from spikingjelly.clock_driven import neuron, layer, surrogate

from .blocks import DownsamplingConv, SEWResBlock, NNConvUpsampling, MultiplyBy


####################
#   MOTHER CLASS   #
####################

class NeuromorphicNet(nn.Module):
    def __init__(self, surrogate_function=surrogate.ATan(), detach_reset=True, v_threshold=1.0, v_reset=0.0):
        super().__init__()
        self.surrogate_fct = surrogate_function
        self.detach_rst = detach_reset
        self.v_th = v_threshold
        self.v_rst = v_reset

        self.max_test_accuracy = float('inf')
        self.epoch = 0
        self.is_spiking = True

    def detach(self):
        for m in self.modules():
            if isinstance(m, neuron.BaseNode):
                m.v.detach_()
            elif isinstance(m, layer.Dropout):
                m.mask.detach_()

    def get_network_state(self):
        state = []
        for m in self.modules():
            if hasattr(m, 'reset'):
                state.append(m.v)
        return state

    def change_network_state(self, new_state):
        module_index = 0
        for m in self.modules():
            if hasattr(m, 'reset'):
                m.v = new_state[module_index]
                module_index += 1

    def set_init_depths_potentials(self, depth_prior):
        self.Ineurons.v = depth_prior

    def multiply_parameters(self, factor):
        """
        Function to absorb the factor of MultiplyBy modules into convolutional weights
        the factor used here should be equal to the inverse of the factor that was used in MultiplyBy during training

        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                print(m)
                m.weight = Parameter(m.weight * factor)
                if m.bias is not None:
                    m.bias = Parameter(m.bias * factor)

    def increment_epoch(self):
        self.epoch += 1

    def get_max_accuracy(self):
        return self.max_test_accuracy

    def update_max_accuracy(self, new_acc):
        self.max_test_accuracy = new_acc

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


####################
# 	BASELINES	   #
####################


class StereoSpike(NeuromorphicNet):
    """
    Third and last version of our model, capable of dense depth predictions from events.
    Now uses separable convolutions for even better performances and hardware-friendliness.

    Possibility to choose between monocular and binocular modalities.

    Features that make our model extremely hardware-friendly:
    - spiking (binary activations)
    - sparse
    - single timestep
    - separable convs
    - few parameters
    - no batchnorm
    - no biases in conv layers
    - nearest neighbor upsampling

    See this excellent article to know more about separable convolutions:
    https://www.paepper.com/blog/posts/depthwise-separable-convolutions-in-pytorch/
    """

    def __init__(self, input_chans=4, binocular=True, kernel_size=7, base_chans=32, use_plif=False, detach_reset=True, tau=10.,
                 v_threshold=1.0, v_reset=0.0, multiply_factor=1., surrogate_function=surrogate.Sigmoid(),
                 separable_convs=True, learnable_biases=False):
        super().__init__(detach_reset=detach_reset)

        self.alt_name = 'fromZero_feedforward_multiscale_tempo_Matt_sepConv_SpikeFlowNetLike'
        self.binocular = binocular

        C = [base_chans * (2 ** n) for n in range(5)]
        K = kernel_size
        P = (kernel_size - 1) // 2

        if use_plif:
            neuron_module = neuron.ParametricLIFNode
        else:
            neuron_module = neuron.LIFNode

        # bottom layer, preprocessing the input spike frame without downsampling
        self.bottom = nn.Sequential(
            # nn.Conv2d(in_channels=4, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.Conv2d(in_channels=input_chans, out_channels=input_chans, groups=input_chans, kernel_size=K, stride=1, padding=P, bias=False),
            nn.Conv2d(in_channels=input_chans, out_channels=C[0], kernel_size=1, stride=1, bias=False),
            MultiplyBy(multiply_factor),
            neuron_module(tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True, surrogate_function=surrogate_function),
        )

        # encoder layers (downsampling)
        self.conv1 = nn.Sequential(
            DownsamplingConv(in_channels=C[0], out_channels=C[1], kernel_size=K, bias=False, separable=separable_convs),
            MultiplyBy(multiply_factor),
            neuron_module(tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True, surrogate_function=surrogate_function),
        )
        self.conv2 = nn.Sequential(
            DownsamplingConv(in_channels=C[1], out_channels=C[2], kernel_size=K, bias=False, separable=separable_convs),
            MultiplyBy(multiply_factor),
            neuron_module(tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True, surrogate_function=surrogate_function),
        )
        self.conv3 = nn.Sequential(
            DownsamplingConv(in_channels=C[2], out_channels=C[3], kernel_size=K, bias=False, separable=separable_convs),
            MultiplyBy(multiply_factor),
            neuron_module(tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True, surrogate_function=surrogate_function),
        )
        self.conv4 = nn.Sequential(
            DownsamplingConv(in_channels=C[3], out_channels=C[4], kernel_size=K, bias=False, separable=separable_convs),
            MultiplyBy(multiply_factor),
            neuron_module(tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True, surrogate_function=surrogate_function),
        )

        # residual layers
        self.bottleneck = nn.Sequential(
            SEWResBlock(C[4], kernel_size=K, v_threshold=v_threshold, v_reset=v_reset, connect_function='ADD', multiply_factor=multiply_factor, use_plif=True, tau=tau, surrogate_function=surrogate_function, separable=separable_convs),
            SEWResBlock(C[4], kernel_size=K, v_threshold=v_threshold, v_reset=v_reset, connect_function='ADD', multiply_factor=multiply_factor, use_plif=True, tau=tau, surrogate_function=surrogate_function, separable=separable_convs),
        )

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            NNConvUpsampling(in_channels=C[4], out_channels=C[3], kernel_size=K, up_size=(33, 44), separable=separable_convs),
            MultiplyBy(multiply_factor),
            neuron_module(tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True, surrogate_function=surrogate_function),
        )
        self.deconv3 = nn.Sequential(
            NNConvUpsampling(in_channels=C[3], out_channels=C[2], kernel_size=K, up_size=(65, 87)),
            MultiplyBy(multiply_factor),
            neuron_module(tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True, surrogate_function=surrogate_function),
        )
        self.deconv2 = nn.Sequential(
            NNConvUpsampling(in_channels=C[2], out_channels=C[1], kernel_size=K, up_size=(130, 173), separable=separable_convs),
            MultiplyBy(multiply_factor),
            neuron_module(tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True, surrogate_function=surrogate_function),
        )
        self.deconv1 = nn.Sequential(
            NNConvUpsampling(in_channels=C[1], out_channels=C[0], kernel_size=K, up_size=(260, 346), separable=separable_convs),
            MultiplyBy(multiply_factor),
            neuron_module(tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True, surrogate_function=surrogate_function),
        )

        # these layers output depth maps at different scales, where depth is represented by the potential of IF neurons
        # that do not fire ("I-neurons"), i.e., with an infinite threshold.
        self.predict_depth4 = nn.Sequential(
            NNConvUpsampling(in_channels=C[3], out_channels=1, kernel_size=K, up_size=(260, 346), bias=True, separable=separable_convs),
            MultiplyBy(multiply_factor),
        )
        self.predict_depth3 = nn.Sequential(
            NNConvUpsampling(in_channels=C[2], out_channels=1, kernel_size=K, up_size=(260, 346), bias=True, separable=separable_convs),
            MultiplyBy(multiply_factor),
        )
        self.predict_depth2 = nn.Sequential(
            NNConvUpsampling(in_channels=C[1], out_channels=1, kernel_size=K, up_size=(260, 346), bias=True, separable=separable_convs),
            MultiplyBy(multiply_factor),
        )
        self.predict_depth1 = nn.Sequential(
            NNConvUpsampling(in_channels=C[0], out_channels=1, kernel_size=K, up_size=(260, 346), bias=True, separable=separable_convs),
            MultiplyBy(multiply_factor),
        )

        # learn the reset potentials of output I-neurons
        self.learnable_biases = learnable_biases
        self.biases = Parameter(torch.zeros((260, 1)))

        self.Ineurons = neuron.IFNode(v_threshold=float('inf'), v_reset=v_reset, surrogate_function=surrogate.ATan())

    @staticmethod
    def reformat_input_data(warmup_chunks_left, warmup_chunks_right, inference_chunks_left, inference_chunks_right):

        # get the dimensions of the data: (B, N, nfpdm, P, H, W)
        B, N_inference, nfpdm, P, H, W = inference_chunks_left.shape
        N_warmup = warmup_chunks_left.shape[1]

        """ uncomment this part depending on how you want to format the data
        
        # reshape the inputs (B, num_chunks, nfpdm, 2, 260, 346) --> (B, num_frames, 2, 260, 346)
        # where num_frames = num_chunks * nfpdm
        warmup_chunks_left = warmup_chunks_left.view(B, N_warmup * nfpdm, P, H, W)
        warmup_chunks_right = warmup_chunks_right.view(B, N_warmup * nfpdm, P, H, W)
        inference_chunks_left = inference_chunks_left.view(B, N_inference * nfpdm, P, H, W)
        inference_chunks_right = inference_chunks_right.view(B, N_inference * nfpdm, P, H, W)

        # concatenate train chunks channelwise: (B, num_frames, 2, 260, 346) --> (B, 1, num_frames*2, 260, 346)
        # (for "tempo" feedforward ANN models, comment for other models)
        warmup_chunks_left = warmup_chunks_left.view(B, 1, N_warmup * nfpdm * P, H, W)
        warmup_chunks_right = warmup_chunks_right.view(B, 1, N_warmup * nfpdm * P, H, W)
        inference_chunks_left = inference_chunks_left.view(B, 1, N_inference * nfpdm * P, H, W)
        inference_chunks_right = inference_chunks_right.view(B, 1, N_inference * nfpdm * P, H, W)
        """

        # sum consecutive chunks (B, N, nfpdm, 2, H, W) --> (B, nfpdm, 2, 260, 346)
        warmup_chunks_left = torch.sum(warmup_chunks_left, dim=1)
        warmup_chunks_right = torch.sum(warmup_chunks_right, dim=1)
        inference_chunks_left = torch.sum(inference_chunks_left, dim=1)
        inference_chunks_right = torch.sum(inference_chunks_right, dim=1)

        # concatenate train chunks channel-wise: (B, nfpdm, 2, 260, 346) --> (B, 1, num_frames*2, 260, 346)
        warmup_chunks_left = warmup_chunks_left.view(B, 1, 1 * nfpdm * 2, 260, 346)
        warmup_chunks_right = warmup_chunks_right.view(B, 1, 1 * nfpdm * 2, 260, 346)
        inference_chunks_left = inference_chunks_left.view(B, 1, 1 * nfpdm * 2, 260, 346)
        inference_chunks_right = inference_chunks_right.view(B, 1, 1 * nfpdm * 2, 260, 346)

        # concatenate L/R inputs channel-wise: 2 * (B, 1, num_frames*2, 260, 346) --> (B, 1, 2*num_frames*2, 260, 346)
        # (for binocular model)
        # TODO: if monocular or if binocular
        warmup_chunks = torch.cat((warmup_chunks_left, warmup_chunks_right), dim=2)
        inference_chunks = torch.cat((inference_chunks_left, inference_chunks_right), dim=2)

        return warmup_chunks, inference_chunks

    def forward(self, x):
        # x must be of shape [batch_size, num_frames_per_depth_map, 2 (polarities), W, H]

        if self.learnable_biases == True:
            self.set_output_potentials(x)

        frame = x[:, 0, :, :, :]

        # data is fed in through the bottom layer
        out_bottom = self.bottom(frame)

        # pass through encoder layers
        out_conv1 = self.conv1(out_bottom)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)

        # pass through residual blocks
        out_rconv = self.bottleneck(out_conv4)

        # gradually upsample while concatenating and passing through skip connections
        out_deconv4 = self.deconv4(out_rconv)
        out_add4 = out_deconv4 + out_conv3
        self.Ineurons(self.predict_depth4(out_add4))
        depth4 = self.Ineurons.v

        out_deconv3 = self.deconv3(out_add4)
        out_add3 = out_deconv3 + out_conv2
        self.Ineurons(self.predict_depth3(out_add3))
        depth3 = self.Ineurons.v

        out_deconv2 = self.deconv2(out_add3)
        out_add2 = out_deconv2 + out_conv1
        self.Ineurons(self.predict_depth2(out_add2))
        depth2 = self.Ineurons.v

        out_deconv1 = self.deconv1(out_add2)
        out_add1 = out_deconv1 + out_bottom
        self.Ineurons(self.predict_depth1(out_add1))
        depth1 = self.Ineurons.v

        # the membrane potentials of the output IF neuron carry the depth prediction
        return [depth1, depth2, depth3, depth4], [out_rconv, out_add4, out_add3, out_add2, out_add1]

    def calculate_firing_rates(self, x):

        # dictionary to store the firing rates for all layers
        firing_rates_dict = {
            'out_bottom': 0.,
            'out_conv1': 0.,
            'out_conv2': 0.,
            'out_conv3': 0.,
            'out_conv4': 0.,
            'out_rconv': 0.,
            'out_deconv4': 0.,
            'out_add4': 0.,
            'out_deconv3': 0.,
            'out_add3': 0.,
            'out_deconv2': 0.,
            'out_add2': 0.,
            'out_deconv1': 0.,
            'out_add1': 0.,
        }

        # x must be of shape [batch_size, num_frames_per_depth_map, 2 (polarities), W, H]
        frame = x[:, 0, :, :, :]

        # data is fed in through the bottom layer and passes through encoder layers
        out_bottom = self.bottom(frame);
        firing_rates_dict['out_bottom'] = out_bottom.count_nonzero() / out_bottom.numel()
        out_conv1 = self.conv1(out_bottom);
        firing_rates_dict['out_conv1'] = out_conv1.count_nonzero() / out_conv1.numel()
        out_conv2 = self.conv2(out_conv1);
        firing_rates_dict['out_conv2'] = out_conv2.count_nonzero() / out_conv2.numel()
        out_conv3 = self.conv3(out_conv2);
        firing_rates_dict['out_conv3'] = out_conv3.count_nonzero() / out_conv3.numel()
        out_conv4 = self.conv4(out_conv3);
        firing_rates_dict['out_conv4'] = out_conv4.count_nonzero() / out_conv4.numel()

        # pass through residual blocks
        out_rconv = self.bottleneck(out_conv4);
        firing_rates_dict['out_rconv'] = out_rconv.count_nonzero() / out_rconv.numel()

        # gradually upsample while concatenating and passing through skip connections
        out_deconv4 = self.deconv4(out_rconv);
        firing_rates_dict['out_deconv4'] = out_deconv4.count_nonzero() / out_deconv4.numel()
        out_add4 = out_deconv4 + out_conv3;
        firing_rates_dict['out_add4'] = out_add4.count_nonzero() / out_add4.numel()
        self.Ineurons(self.predict_depth4(out_add4))

        out_deconv3 = self.deconv3(out_add4);
        firing_rates_dict['out_deconv3'] = out_deconv3.count_nonzero() / out_deconv3.numel()
        out_add3 = out_deconv3 + out_conv2;
        firing_rates_dict['out_add3'] = out_add3.count_nonzero() / out_add3.numel()
        self.Ineurons(self.predict_depth3(out_add3))

        out_deconv2 = self.deconv2(out_add3);
        firing_rates_dict['out_deconv2'] = out_deconv2.count_nonzero() / out_deconv2.numel()
        out_add2 = out_deconv2 + out_conv1;
        firing_rates_dict['out_add2'] = out_add2.count_nonzero() / out_add2.numel()
        self.Ineurons(self.predict_depth2(out_add2))

        out_deconv1 = self.deconv1(out_add2);
        firing_rates_dict['out_deconv1'] = out_deconv1.count_nonzero() / out_deconv1.numel()
        out_add1 = out_deconv1 + out_bottom;
        firing_rates_dict['out_add1'] = out_add1.count_nonzero() / out_add1.numel()
        self.Ineurons(self.predict_depth1(out_add1))

        return firing_rates_dict
