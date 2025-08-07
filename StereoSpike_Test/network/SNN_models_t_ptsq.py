'''
2025-02-17
Andres Brito

Based on the original SNN_models_simpquant.py provided by the StereoSpike's author.
    - Test to find a way to modify quantized ifmaps to the corresponding 1's and 0's required for HW accelerator.
    - Create a new class with a test Quantizable Simplified StereoSpike Network --> based on the original Quantizable Simplified Network
'''

import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from spikingjelly.clock_driven import neuron, layer, surrogate

from .blocks import SEWResBlock, NNConvUpsampling, MultiplyBy, SeparableSEWResBlock, SeparableNNConvUpsampling
# from .blocks_old_quantize import InferenceOnlyHeaviside, QuantizedSeparableSEWResBlock
#from .blocks import forFLOPSSeparableSEWResBlock, forFLOPSSeparableNNConvUpsampling

from torch.quantization import QuantStub, DeQuantStub


class QUANTIZABLE_SIMPLIFIED_SeparableSEWResBlock(nn.Module):
    """
    Quantization-compatible version of the simplified resblock
    """

    def __init__(self, in_channels: int, connect_function='ADD', v_threshold=1., v_reset=0.,
                 surrogate_function=surrogate.Sigmoid(), use_plif=False, tau=2., multiply_factor=1., kernel_size=3):
        super(QUANTIZABLE_SIMPLIFIED_SeparableSEWResBlock, self).__init__()

        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.skip_conn = nn.quantized.FloatFunctional()

        self.conv1 = nn.Sequential(
            # nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=(3 - 1) // 2, bias=False),
            nn.Conv2d(in_channels, in_channels, groups=in_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False),
        )

        #self.sn1 = neuron.IFNode()

        self.conv2 = nn.Sequential(
            # nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=(3 - 1) // 2, bias=False),
            nn.Conv2d(in_channels, in_channels, groups=in_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False),
        )

        #self.sn2 = neuron.IFNode()

        self.connect_function = connect_function

    def forward(self, x):
        # expects an already-quantized input

        identity = x

        out = self.conv1(x)
        out = self.quant(torch.heaviside(self.dequant(out) - 1., torch.tensor(1., device=x.device)))
        out = self.conv2(out)
        out = self.quant(torch.heaviside(self.dequant(out) - 1, torch.tensor(1., device=x.device)))

        if self.connect_function == 'ADD':
            #out += identity
            res = self.skip_conn.add(out, identity)
        elif self.connect_function == 'MUL' or self.connect_function == 'AND':
            out *= identity
        elif self.connect_function == 'OR':
            out = surrogate.ATan(spiking=True)(out + identity)
        elif self.connect_function == 'NMUL':
            out = identity * (1. - out)
        else:
            raise NotImplementedError(self.connect_function)

        return out


class SIMPLIFIED_SeparableSEWResBlock(nn.Module):
    """
    Version of the SEW-Resblock using Separable Convolutions
    """

    def __init__(self, in_channels: int, connect_function='ADD', v_threshold=1., v_reset=0.,
                 surrogate_function=surrogate.Sigmoid(), use_plif=False, tau=2., multiply_factor=1., kernel_size=3):
        super(SIMPLIFIED_SeparableSEWResBlock, self).__init__()

        self.conv1 = nn.Sequential(
            # nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=(3 - 1) // 2, bias=False),
            nn.Conv2d(in_channels, in_channels, groups=in_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False),
        )

        #self.sn1 = neuron.IFNode()

        self.conv2 = nn.Sequential(
            # nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=(3 - 1) // 2, bias=False),
            nn.Conv2d(in_channels, in_channels, groups=in_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False),
        )

        #self.sn2 = neuron.IFNode()

        self.connect_function = connect_function

    def forward(self, x):

        identity = x

        # put conv1, sn1, conv2, sn2 in an nn.Seqential ?
        out = self.conv1(x)
        out = torch.heaviside(out - 1., torch.tensor(1., device=x.device))  # self.sn1(out)
        out = self.conv2(out)
        out = torch.heaviside(out - 1, torch.tensor(1., device=x.device))  # self.sn2(out)

        if self.connect_function == 'ADD':
            out += identity
        elif self.connect_function == 'MUL' or self.connect_function == 'AND':
            out *= identity
        elif self.connect_function == 'OR':
            out = surrogate.ATan(spiking=True)(out + identity)
        elif self.connect_function == 'NMUL':
            out = identity * (1. - out)
        else:
            raise NotImplementedError(self.connect_f)

        return out


def remove_module(sequential, index):
    modules = list(sequential.children())
    if 0 <= index < len(modules):
        del modules[index]
    return nn.Sequential(*modules)


class NeuromorphicNet(nn.Module):
    def __init__(self, surrogate_function=surrogate.ATan(), detach_reset=True, v_threshold=1.0, v_reset=0.0):
        super().__init__()
        self.surrogate_fct = surrogate_function
        self.detach_rst = detach_reset
        self.v_th = v_threshold
        self.v_rst = v_reset

        self.max_test_accuracy = float('inf')
        self.epoch = 0

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

    def set_output_potentials(self, new_pots):
        module_index = 0
        for m in self.modules():
            if isinstance(m, neuron.IFNode):
                m.v = new_pots[module_index]
                module_index += 1

    def increment_epoch(self):
        self.epoch += 1

    def get_max_accuracy(self):
        return self.max_test_accuracy

    def update_max_accuracy(self, new_acc):
        self.max_test_accuracy = new_acc

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class fromZero_feedforward_multiscale_tempo_Matt_sepConv_SpikeFlowNetLike(NeuromorphicNet):
    """
    Uses separable convolutions for a lighter model

    See this excellent article to know moreabout separable convolutions:
    https://www.paepper.com/blog/posts/depthwise-separable-convolutions-in-pytorch/
    """
    def __init__(self, input_chans=4, kernel_size=7, base_chans=32, use_plif=False, detach_reset=True, tau=10., v_threshold=1.0, v_reset=0.0, multiply_factor=1., surrogate_function=surrogate.Sigmoid(), learnable_biases=False):
        super().__init__(detach_reset=detach_reset)
        
        C = [base_chans * (2**n) for n in range(5)]
        K = kernel_size
        P = (kernel_size - 1) // 2
        self.multiply_factor = multiply_factor

        # bottom layer, preprocessing the input spike frame without downsampling
        self.bottom = nn.Sequential(
            # nn.Conv2d(in_channels=4, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.Conv2d(in_channels=input_chans, out_channels=input_chans, groups=input_chans, kernel_size=K, stride=1, padding=P, bias=False),
            nn.Conv2d(in_channels=input_chans, out_channels=C[0], kernel_size=1, stride=1, bias=False),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True, surrogate_function=surrogate_function) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # encoder layers (downsampling)
        self.conv1 = nn.Sequential(
            # nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.Conv2d(in_channels=C[0], out_channels=C[0], groups=C[0], kernel_size=K, stride=2, padding=P, bias=False),
            nn.Conv2d(in_channels=C[0], out_channels=C[1], kernel_size=1, stride=1, bias=False),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True, surrogate_function=surrogate_function) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv2 = nn.Sequential(
            # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.Conv2d(in_channels=C[1], out_channels=C[1], groups=C[1], kernel_size=K, stride=2, padding=P, bias=False),
            nn.Conv2d(in_channels=C[1], out_channels=C[2], kernel_size=1, stride=1, bias=False),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True, surrogate_function=surrogate_function) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv3 = nn.Sequential(
            # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2, bias=False),
            nn.Conv2d(in_channels=C[2], out_channels=C[2], groups=C[2], kernel_size=K, stride=2, padding=P, bias=False),
            nn.Conv2d(in_channels=C[2], out_channels=C[3], kernel_size=1, stride=1, bias=False),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True, surrogate_function=surrogate_function) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv4 = nn.Sequential(
            # nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2, bias=False),
            nn.Conv2d(in_channels=C[3], out_channels=C[3], groups=C[3], kernel_size=K, stride=2, padding=P, bias=False),
            nn.Conv2d(in_channels=C[3], out_channels=C[4], kernel_size=1, stride=1, bias=False),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True, surrogate_function=surrogate_function) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # residual layers
        self.bottleneck = nn.Sequential(
            SeparableSEWResBlock(C[4], kernel_size=K, v_threshold=v_threshold, v_reset=v_reset, connect_function='ADD', multiply_factor=multiply_factor, use_plif=True, tau=tau, surrogate_function=surrogate_function),
            SeparableSEWResBlock(C[4], kernel_size=K, v_threshold=v_threshold, v_reset=v_reset, connect_function='ADD', multiply_factor=multiply_factor, use_plif=True, tau=tau, surrogate_function=surrogate_function),
        )

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            SeparableNNConvUpsampling(in_channels=C[4], out_channels=C[3], kernel_size=K, up_size=(33, 44)),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True, surrogate_function=surrogate_function) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.deconv3 = nn.Sequential(
            SeparableNNConvUpsampling(in_channels=C[3], out_channels=C[2], kernel_size=K, up_size=(65, 87)),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True, surrogate_function=surrogate_function) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.deconv2 = nn.Sequential(
            SeparableNNConvUpsampling(in_channels=C[2], out_channels=C[1], kernel_size=K, up_size=(130, 173)),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True, surrogate_function=surrogate_function) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.deconv1 = nn.Sequential(
            SeparableNNConvUpsampling(in_channels=C[1], out_channels=C[0], kernel_size=K, up_size=(260, 346)),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True, surrogate_function=surrogate_function) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # these layers output depth maps at different scales, where depth is represented by the potential of IF neurons
        # that do not fire ("I-neurons"), i.e., with an infinite threshold.
        self.predict_depth4 = nn.Sequential(
            SeparableNNConvUpsampling(in_channels=C[3], out_channels=1, kernel_size=K, up_size=(260, 346), bias=True),
            MultiplyBy(multiply_factor),
        )
        self.predict_depth3 = nn.Sequential(
            SeparableNNConvUpsampling(in_channels=C[2], out_channels=1, kernel_size=K, up_size=(260, 346), bias=True),
            MultiplyBy(multiply_factor),
        )
        self.predict_depth2 = nn.Sequential(
            SeparableNNConvUpsampling(in_channels=C[1], out_channels=1, kernel_size=K, up_size=(260, 346), bias=True),
            MultiplyBy(multiply_factor),
        )
        self.predict_depth1 = nn.Sequential(
            SeparableNNConvUpsampling(in_channels=C[0], out_channels=1, kernel_size=K, up_size=(260, 346), bias=True),
            MultiplyBy(multiply_factor),
        )

        # learn the reset potentials of output I-neurons
        self.learnable_biases = learnable_biases
        self.biases = Parameter(torch.zeros((260, 1)))

        #self.Ineurons = neuron.IFNode(v_threshold=float('inf'), v_reset=v_reset, surrogate_function=surrogate.ATan())
        
    @staticmethod
    def reformat_input_data(warmup_chunks_left, warmup_chunks_right, inference_chunks_left, inference_chunks_right):
    
        # get the dimensions of the data: (B, N, nfpdm, P, H, W)
        B, N_inference, nfpdm, P, H, W = inference_chunks_left.shape
        N_warmup = warmup_chunks_left.shape[1]

        """
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
        warmup_chunks = torch.cat((warmup_chunks_left, warmup_chunks_right), dim=2)
        inference_chunks = torch.cat((inference_chunks_left, inference_chunks_right), dim=2)

        return warmup_chunks, inference_chunks

    def forward(self, x):
        # x must be of shape [batch_size, num_frames_per_depth_map, 2 (polarities), W, H]

        #self.Ineurons(torch.rand(1))
    
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
        depth4 = self.predict_depth4(out_add4) # * self.multiply_factor
        #self.Ineurons(self.predict_depth4(out_add4) * self.multiply_factor)
        #depth4 = self.Ineurons.v

        out_deconv3 = self.deconv3(out_add4)
        out_add3 = out_deconv3 + out_conv2
        depth3 = depth4 + self.predict_depth3(out_add3) # * self.multiply_factor
        #self.Ineurons(self.predict_depth3(out_add3) * self.multiply_factor)
        #depth3 = self.Ineurons.v

        out_deconv2 = self.deconv2(out_add3)
        out_add2 = out_deconv2 + out_conv1
        depth2  = depth3 + self.predict_depth2(out_add2) # * self.multiply_factor
        #self.Ineurons(self.predict_depth2(out_add2) * self.multiply_factor)
        #depth2 = self.Ineurons.v

        out_deconv1 = self.deconv1(out_add2)
        out_add1 = out_deconv1 + out_bottom
        depth1 = depth2 + self.predict_depth1(out_add1) # * self.multiply_factor
        #self.Ineurons(self.predict_depth1(out_add1) * self.multiply_factor)
        #depth1 = self.Ineurons.v

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
        out_bottom = self.bottom(frame); firing_rates_dict['out_bottom'] = out_bottom.count_nonzero()/out_bottom.numel()
        out_conv1 = self.conv1(out_bottom); firing_rates_dict['out_conv1'] = out_conv1.count_nonzero()/out_conv1.numel()
        out_conv2 = self.conv2(out_conv1); firing_rates_dict['out_conv2'] = out_conv2.count_nonzero()/out_conv2.numel()
        out_conv3 = self.conv3(out_conv2); firing_rates_dict['out_conv3'] = out_conv3.count_nonzero()/out_conv3.numel()
        out_conv4 = self.conv4(out_conv3); firing_rates_dict['out_conv4'] = out_conv4.count_nonzero()/out_conv4.numel()

        # pass through residual blocks
        out_rconv = self.bottleneck(out_conv4); firing_rates_dict['out_rconv'] = out_rconv.count_nonzero()/out_rconv.numel()

        # gradually upsample while concatenating and passing through skip connections
        out_deconv4 = self.deconv4(out_rconv); firing_rates_dict['out_deconv4'] = out_deconv4.count_nonzero()/out_deconv4.numel()
        out_add4 = out_deconv4 + out_conv3; firing_rates_dict['out_add4'] = out_add4.count_nonzero()/out_add4.numel()
        self.Ineurons(self.predict_depth4(out_add4))

        out_deconv3 = self.deconv3(out_add4); firing_rates_dict['out_deconv3'] = out_deconv3.count_nonzero()/out_deconv3.numel()
        out_add3 = out_deconv3 + out_conv2; firing_rates_dict['out_add3'] = out_add3.count_nonzero()/out_add3.numel()
        self.Ineurons(self.predict_depth3(out_add3))

        out_deconv2 = self.deconv2(out_add3); firing_rates_dict['out_deconv2'] = out_deconv2.count_nonzero()/out_deconv2.numel()
        out_add2 = out_deconv2 + out_conv1; firing_rates_dict['out_add2'] = out_add2.count_nonzero()/out_add2.numel()
        self.Ineurons(self.predict_depth2(out_add2))

        out_deconv1 = self.deconv1(out_add2); firing_rates_dict['out_deconv1'] = out_deconv1.count_nonzero()/out_deconv1.numel()
        out_add1 = out_deconv1 + out_bottom; firing_rates_dict['out_add1'] = out_add1.count_nonzero()/out_add1.numel()
        self.Ineurons(self.predict_depth1(out_add1))

        return firing_rates_dict

    def multiply_parameters(self, factor):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                print(m)
                m.weight = Parameter(m.weight * factor)
                if m.bias is not None:
                    m.bias = Parameter(m.bias * factor)

    def simplify_model(self):
        """ convert the model into an equivalent version in which multiplyBy and PLIF's taus are absorbed into the convs"""

        # encoder: remove MultiplyBy and PLIF
        self.bottom = remove_module(self.bottom, 2)
        self.bottom[0].weight = Parameter(self.bottom[0].weight * self.multiply_factor * torch.sigmoid(self.bottom[-1].w).item())
        self.bottom[-1] = neuron.IFNode()

        self.conv1 = remove_module(self.conv1, 2)
        self.conv1[0].weight = Parameter(self.conv1[0].weight * self.multiply_factor * torch.sigmoid(self.conv1[-1].w).item())
        self.conv1[-1] = neuron.IFNode()

        self.conv2 = remove_module(self.conv2, 2)
        self.conv2[0].weight = Parameter(self.conv2[0].weight * self.multiply_factor * torch.sigmoid(self.conv2[-1].w).item())
        self.conv2[-1] = neuron.IFNode()

        self.conv3 = remove_module(self.conv3, 2)
        self.conv3[0].weight = Parameter(self.conv3[0].weight * self.multiply_factor * torch.sigmoid(self.conv3[-1].w).item())
        self.conv3[-1] = neuron.IFNode()

        self.conv4 = remove_module(self.conv4, 2)
        self.conv4[0].weight = Parameter(self.conv4[0].weight * self.multiply_factor * torch.sigmoid(self.conv4[-1].w).item())
        self.conv4[-1] = neuron.IFNode()

        # bottleneck: remove MultiplyBy and PLIF        
        self.bottleneck[0].conv1 = remove_module(self.bottleneck[0].conv1, 2)
        self.bottleneck[0].conv1[0].weight = Parameter(self.bottleneck[0].conv1[0].weight * self.multiply_factor * torch.sigmoid(self.bottleneck[0].sn1.w).item())
        self.bottleneck[0].sn1 = neuron.IFNode()

        self.bottleneck[0].conv2 = remove_module(self.bottleneck[0].conv2, 2)
        self.bottleneck[0].conv2[0].weight = Parameter(self.bottleneck[0].conv2[0].weight * self.multiply_factor * torch.sigmoid(self.bottleneck[0].sn2.w).item())
        self.bottleneck[0].sn2 = neuron.IFNode()

        self.bottleneck[1].conv1 = remove_module(self.bottleneck[1].conv1, 2)
        self.bottleneck[1].conv1[0].weight = Parameter(self.bottleneck[1].conv1[0].weight * self.multiply_factor * torch.sigmoid(self.bottleneck[1].sn1.w).item())
        self.bottleneck[1].sn1 = neuron.IFNode()

        self.bottleneck[1].conv2 = remove_module(self.bottleneck[1].conv2, 2)
        self.bottleneck[1].conv2[0].weight = Parameter(self.bottleneck[1].conv2[0].weight * self.multiply_factor * torch.sigmoid(self.bottleneck[1].sn2.w).item())
        self.bottleneck[1].sn2 = neuron.IFNode()
                
        # decoder: remove MultiplyBy and PLIF 
        self.deconv4 = remove_module(self.deconv4, 1) 
        self.deconv4[0].up[1].weight = Parameter(self.deconv4[0].up[1].weight * self.multiply_factor * torch.sigmoid(self.deconv4[-1].w).item())
        self.deconv4[-1] = neuron.IFNode()

        self.deconv3 = remove_module(self.deconv3, 1)
        self.deconv3[0].up[1].weight = Parameter(self.deconv3[0].up[1].weight * self.multiply_factor * torch.sigmoid(self.deconv3[-1].w).item())
        self.deconv3[-1] = neuron.IFNode()

        self.deconv2 = remove_module(self.deconv2, 1)
        self.deconv2[0].up[1].weight = Parameter(self.deconv2[0].up[1].weight * self.multiply_factor * torch.sigmoid(self.deconv2[-1].w).item())
        self.deconv2[-1] = neuron.IFNode()

        self.deconv1 = remove_module(self.deconv1, 1)
        self.deconv1[0].up[1].weight = Parameter(self.deconv1[0].up[1].weight * self.multiply_factor * torch.sigmoid(self.deconv1[-1].w).item())
        self.deconv1[-1] = neuron.IFNode()
        
        # predictor: remove MultiplyBy (caution: output convs have biases !)
        self.predict_depth4 = self.predict_depth4[0]    # remove_module(self.predict_depth4, 1)
        self.predict_depth4.up[1].weight = Parameter(self.predict_depth4.up[1].weight * self.multiply_factor)
        self.predict_depth4.up[1].bias = Parameter(self.predict_depth4.up[1].bias * self.multiply_factor)
        self.predict_depth4.up[2].bias = Parameter(self.predict_depth4.up[2].bias * self.multiply_factor)

        self.predict_depth3 = self.predict_depth3[0]    # remove_module(self.predict_depth3, 1)
        self.predict_depth3.up[1].weight = Parameter(self.predict_depth3.up[1].weight * self.multiply_factor)
        self.predict_depth3.up[1].bias = Parameter(self.predict_depth3.up[1].bias * self.multiply_factor)
        self.predict_depth3.up[2].bias = Parameter(self.predict_depth3.up[2].bias * self.multiply_factor)

        self.predict_depth2 = self.predict_depth2[0]    # remove_module(self.predict_depth2, 1)
        self.predict_depth2.up[1].weight = Parameter(self.predict_depth2.up[1].weight * self.multiply_factor)
        self.predict_depth2.up[1].bias = Parameter(self.predict_depth2.up[1].bias * self.multiply_factor)
        self.predict_depth2.up[2].bias = Parameter(self.predict_depth2.up[2].bias * self.multiply_factor)

        self.predict_depth1 = self.predict_depth1[0]    # remove_module(self.predict_depth1, 1)
        self.predict_depth1.up[1].weight = Parameter(self.predict_depth1.up[1].weight * self.multiply_factor)
        self.predict_depth1.up[1].bias = Parameter(self.predict_depth1.up[1].bias * self.multiply_factor)
        self.predict_depth1.up[2].bias = Parameter(self.predict_depth1.up[2].bias * self.multiply_factor)


    def set_output_potentials(self, x):
        """
        Set the potentials of output neurons to their learned value (self.bias), so that the batchsize dimension matches
        the input tensor during inference
        """
        B = x.shape[0]
        pots = torch.zeros(B, 1, 260, 346).to(x.device)
        self.Ineurons.v = pots + self.biases


    def set_init_depths_potentials(self, depth_prior):
        self.Ineurons.v = depth_prior


class SIMPLIFIED_fromZero_feedforward_multiscale_tempo_Matt_sepConv_SpikeFlowNetLike(NeuromorphicNet):
    """
    Simplified but mathematically equivalent version of 'fromZero_feedforward_multiscale_tempo_Matt_sepConv_SpikeFlowNetLike'

    """
    def __init__(self, input_chans=4, kernel_size=7, base_chans=32, use_plif=False, detach_reset=True, tau=10., v_threshold=1.0, v_reset=0.0, multiply_factor=1., surrogate_function=surrogate.Sigmoid(), learnable_biases=False):
        super().__init__(detach_reset=detach_reset)

        C = [base_chans * (2**n) for n in range(5)]
        K = kernel_size
        P = (kernel_size - 1) // 2
        self.multiply_factor = multiply_factor

        # bottom layer, preprocessing the input spike frame without downsampling
        self.bottom = nn.Sequential(
            # nn.Conv2d(in_channels=4, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.Conv2d(in_channels=input_chans, out_channels=input_chans, groups=input_chans, kernel_size=K, stride=1, padding=P, bias=False),
            nn.Conv2d(in_channels=input_chans, out_channels=C[0], kernel_size=1, stride=1, bias=False),
            #neuron.IFNode()
        )

        # encoder layers (downsampling)
        self.conv1 = nn.Sequential(
            # nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.Conv2d(in_channels=C[0], out_channels=C[0], groups=C[0], kernel_size=K, stride=2, padding=P, bias=False),
            nn.Conv2d(in_channels=C[0], out_channels=C[1], kernel_size=1, stride=1, bias=False),
            #neuron.IFNode()
        )
        self.conv2 = nn.Sequential(
            # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.Conv2d(in_channels=C[1], out_channels=C[1], groups=C[1], kernel_size=K, stride=2, padding=P, bias=False),
            nn.Conv2d(in_channels=C[1], out_channels=C[2], kernel_size=1, stride=1, bias=False),
            #neuron.IFNode(),
        )
        self.conv3 = nn.Sequential(
            # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2, bias=False),
            nn.Conv2d(in_channels=C[2], out_channels=C[2], groups=C[2], kernel_size=K, stride=2, padding=P, bias=False),
            nn.Conv2d(in_channels=C[2], out_channels=C[3], kernel_size=1, stride=1, bias=False),
            #neuron.IFNode()
        )
        self.conv4 = nn.Sequential(
            # nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2, bias=False),
            nn.Conv2d(in_channels=C[3], out_channels=C[3], groups=C[3], kernel_size=K, stride=2, padding=P, bias=False),
            nn.Conv2d(in_channels=C[3], out_channels=C[4], kernel_size=1, stride=1, bias=False),
            #neuron.IFNode(),
        )

        # residual layers
        self.bottleneck = nn.Sequential(
            SIMPLIFIED_SeparableSEWResBlock(C[4], kernel_size=K, v_threshold=v_threshold, v_reset=v_reset, connect_function='ADD', multiply_factor=multiply_factor, use_plif=True, tau=tau, surrogate_function=surrogate_function),
            SIMPLIFIED_SeparableSEWResBlock(C[4], kernel_size=K, v_threshold=v_threshold, v_reset=v_reset, connect_function='ADD', multiply_factor=multiply_factor, use_plif=True, tau=tau, surrogate_function=surrogate_function),
        )

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            SeparableNNConvUpsampling(in_channels=C[4], out_channels=C[3], kernel_size=K, up_size=(33, 44)),
            #neuron.IFNode()
        )
        self.deconv3 = nn.Sequential(
            SeparableNNConvUpsampling(in_channels=C[3], out_channels=C[2], kernel_size=K, up_size=(65, 87)),
            #neuron.IFNode()
        )
        self.deconv2 = nn.Sequential(
            SeparableNNConvUpsampling(in_channels=C[2], out_channels=C[1], kernel_size=K, up_size=(130, 173)),
            #neuron.IFNode()
        )
        self.deconv1 = nn.Sequential(
            SeparableNNConvUpsampling(in_channels=C[1], out_channels=C[0], kernel_size=K, up_size=(260, 346)),
            #neuron.IFNode()
        )

        # these layers output depth maps at different scales, where depth is represented by the potential of IF neurons
        # that do not fire ("I-neurons"), i.e., with an infinite threshold.
        self.predict_depth4 = SeparableNNConvUpsampling(in_channels=C[3], out_channels=1, kernel_size=K, up_size=(260, 346), bias=True)
        self.predict_depth3 = SeparableNNConvUpsampling(in_channels=C[2], out_channels=1, kernel_size=K, up_size=(260, 346), bias=True)
        self.predict_depth2 = SeparableNNConvUpsampling(in_channels=C[1], out_channels=1, kernel_size=K, up_size=(260, 346), bias=True)
        self.predict_depth1 = SeparableNNConvUpsampling(in_channels=C[0], out_channels=1, kernel_size=K, up_size=(260, 346), bias=True)
        
        # learn the reset potentials of output I-neurons
        self.learnable_biases = learnable_biases
        self.biases = Parameter(torch.zeros((260, 1)))

    @staticmethod
    def reformat_input_data(warmup_chunks_left, warmup_chunks_right, inference_chunks_left, inference_chunks_right):

        # get the dimensions of the data: (B, N, nfpdm, P, H, W)
        B, N_inference, nfpdm, P, H, W = inference_chunks_left.shape
        N_warmup = warmup_chunks_left.shape[1]

        """
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
        warmup_chunks = torch.cat((warmup_chunks_left, warmup_chunks_right), dim=2)
        inference_chunks = torch.cat((inference_chunks_left, inference_chunks_right), dim=2)

        return warmup_chunks, inference_chunks

    def forward(self, x):
        # x must be of shape [batch_size, num_frames_per_depth_map, 2 (polarities), W, H]

        if self.learnable_biases == True:
            self.set_output_potentials(x)

        frame = x[:, 0, :, :, :]

        # data is fed in through the bottom layer
        out_bottom = torch.heaviside(self.bottom(frame) - 1., torch.tensor(0.0, device=x.device))

        # pass through encoder layers
        out_conv1 = torch.heaviside(self.conv1(out_bottom) - 1., torch.tensor(0.0, device=x.device))
        out_conv2 = torch.heaviside(self.conv2(out_conv1) - 1., torch.tensor(0.0, device=x.device))
        out_conv3 = torch.heaviside(self.conv3(out_conv2) - 1., torch.tensor(0.0, device=x.device))
        out_conv4 = torch.heaviside(self.conv4(out_conv3) - 1., torch.tensor(0.0, device=x.device))

        # pass through residual blocks
        out_rconv = self.bottleneck(out_conv4)

        # gradually upsample while concatenating and passing through skip connections
        out_deconv4 = torch.heaviside(self.deconv4(out_rconv) - 1., torch.tensor(0.0, device=x.device))
        out_add4 = out_deconv4 + out_conv3
        depth4 = self.predict_depth4(out_add4) # * self.multiply_factor
        #self.Ineurons(self.predict_depth4(out_add4) * self.multiply_factor)
        #depth4 = self.Ineurons.v

        out_deconv3 = torch.heaviside(self.deconv3(out_add4) - 1., torch.tensor(0.0, device=x.device))
        out_add3 = out_deconv3 + out_conv2
        depth3 = depth4 + self.predict_depth3(out_add3) # * self.multiply_factor
        #self.Ineurons(self.predict_depth3(out_add3) * self.multiply_factor)
        #depth3 = self.Ineurons.v

        out_deconv2 = torch.heaviside(self.deconv2(out_add3) - 1., torch.tensor(0.0, device=x.device))
        out_add2 = out_deconv2 + out_conv1
        depth2  = depth3 + self.predict_depth2(out_add2) # * self.multiply_factor
        #self.Ineurons(self.predict_depth2(out_add2) * self.multiply_factor)
        #depth2 = self.Ineurons.v

        out_deconv1 = torch.heaviside(self.deconv1(out_add2) - 1., torch.tensor(0.0, device=x.device))
        out_add1 = out_deconv1 + out_bottom
        depth1 = depth2 + self.predict_depth1(out_add1) # * self.multiply_factor
        #self.Ineurons(self.predict_depth1(out_add1) * self.multiply_factor)
        #depth1 = self.Ineurons.v

        # the membrane potentials of the output IF neuron carry the depth prediction
        return [depth1, depth2, depth3, depth4], [out_rconv, out_add4, out_add3, out_add2, out_add1]


class QUANTIZABLE_SIMPLIFIED_fromZero_feedforward_multiscale_tempo_Matt_sepConv_SpikeFlowNetLike(NeuromorphicNet):
    """
    Quantization-compatible version of the simplified model

    """
    def __init__(self, input_chans=4, kernel_size=7, base_chans=32, use_plif=False, detach_reset=True, tau=10., v_threshold=1.0, v_reset=0.0, multiply_factor=1., surrogate_function=surrogate.Sigmoid(), learnable_biases=False):
        super().__init__(detach_reset=detach_reset)

        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.skip_add = nn.quantized.FloatFunctional()
        self.pred_add = nn.quantized.FloatFunctional()

        C = [base_chans * (2**n) for n in range(5)]
        K = kernel_size
        P = (kernel_size - 1) // 2
        self.multiply_factor = multiply_factor

        # bottom layer, preprocessing the input spike frame without downsampling
        self.bottom = nn.Sequential(
            # nn.Conv2d(in_channels=4, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.Conv2d(in_channels=input_chans, out_channels=input_chans, groups=input_chans, kernel_size=K, stride=1, padding=P, bias=False),
            nn.Conv2d(in_channels=input_chans, out_channels=C[0], kernel_size=1, stride=1, bias=False),
            #neuron.IFNode()
        )

        # encoder layers (downsampling)
        self.conv1 = nn.Sequential(
            # nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.Conv2d(in_channels=C[0], out_channels=C[0], groups=C[0], kernel_size=K, stride=2, padding=P, bias=False),
            nn.Conv2d(in_channels=C[0], out_channels=C[1], kernel_size=1, stride=1, bias=False),
            #neuron.IFNode()
        )
        self.conv2 = nn.Sequential(
            # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.Conv2d(in_channels=C[1], out_channels=C[1], groups=C[1], kernel_size=K, stride=2, padding=P, bias=False),
            nn.Conv2d(in_channels=C[1], out_channels=C[2], kernel_size=1, stride=1, bias=False),
            #neuron.IFNode(),
        )
        self.conv3 = nn.Sequential(
            # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2, bias=False),
            nn.Conv2d(in_channels=C[2], out_channels=C[2], groups=C[2], kernel_size=K, stride=2, padding=P, bias=False),
            nn.Conv2d(in_channels=C[2], out_channels=C[3], kernel_size=1, stride=1, bias=False),
            #neuron.IFNode()
        )
        self.conv4 = nn.Sequential(
            # nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2, bias=False),
            nn.Conv2d(in_channels=C[3], out_channels=C[3], groups=C[3], kernel_size=K, stride=2, padding=P, bias=False),
            nn.Conv2d(in_channels=C[3], out_channels=C[4], kernel_size=1, stride=1, bias=False),
            #neuron.IFNode(),
        )

        # residual layers
        self.bottleneck = nn.Sequential(
            QUANTIZABLE_SIMPLIFIED_SeparableSEWResBlock(C[4], kernel_size=K, v_threshold=v_threshold, v_reset=v_reset, connect_function='ADD', multiply_factor=multiply_factor, use_plif=True, tau=tau, surrogate_function=surrogate_function),
            QUANTIZABLE_SIMPLIFIED_SeparableSEWResBlock(C[4], kernel_size=K, v_threshold=v_threshold, v_reset=v_reset, connect_function='ADD', multiply_factor=multiply_factor, use_plif=True, tau=tau, surrogate_function=surrogate_function),
        )

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            SeparableNNConvUpsampling(in_channels=C[4], out_channels=C[3], kernel_size=K, up_size=(33, 44)),
            #neuron.IFNode()
        )
        self.deconv3 = nn.Sequential(
            SeparableNNConvUpsampling(in_channels=C[3], out_channels=C[2], kernel_size=K, up_size=(65, 87)),
            #neuron.IFNode()
        )
        self.deconv2 = nn.Sequential(
            SeparableNNConvUpsampling(in_channels=C[2], out_channels=C[1], kernel_size=K, up_size=(130, 173)),
            #neuron.IFNode()
        )
        self.deconv1 = nn.Sequential(
            SeparableNNConvUpsampling(in_channels=C[1], out_channels=C[0], kernel_size=K, up_size=(260, 346)),
            #neuron.IFNode()
        )

        # these layers output depth maps at different scales, where depth is represented by the potential of IF neurons
        # that do not fire ("I-neurons"), i.e., with an infinite threshold.
        self.predict_depth4 = SeparableNNConvUpsampling(in_channels=C[3], out_channels=1, kernel_size=K, up_size=(260, 346), bias=True)
        self.predict_depth3 = SeparableNNConvUpsampling(in_channels=C[2], out_channels=1, kernel_size=K, up_size=(260, 346), bias=True)
        self.predict_depth2 = SeparableNNConvUpsampling(in_channels=C[1], out_channels=1, kernel_size=K, up_size=(260, 346), bias=True)
        self.predict_depth1 = SeparableNNConvUpsampling(in_channels=C[0], out_channels=1, kernel_size=K, up_size=(260, 346), bias=True)

        # learn the reset potentials of output I-neurons
        self.learnable_biases = learnable_biases
        self.biases = Parameter(torch.zeros((260, 1)))


    @staticmethod
    def reformat_input_data(warmup_chunks_left, warmup_chunks_right, inference_chunks_left, inference_chunks_right):

        # get the dimensions of the data: (B, N, nfpdm, P, H, W)
        B, N_inference, nfpdm, P, H, W = inference_chunks_left.shape
        N_warmup = warmup_chunks_left.shape[1]

        """
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
        warmup_chunks = torch.cat((warmup_chunks_left, warmup_chunks_right), dim=2)
        inference_chunks = torch.cat((inference_chunks_left, inference_chunks_right), dim=2)

        return warmup_chunks, inference_chunks

    def forward(self, x):
        # x must be of shape [batch_size, num_frames_per_depth_map, 2 (polarities), W, H]

        if self.learnable_biases == True:
            self.set_output_potentials(x)

        # quantize input tensor
        x = self.quant(x)

        frame = x[:, 0, :, :, :]

        # data is fed in through the bottom layer
        out_bottom = self.quant(torch.heaviside(self.dequant(self.bottom(frame)) - 1., torch.tensor(0.0, device=x.device)))

        # pass through encoder layers
        #out_conv1 = torch.heaviside(self.conv1(out_bottom) - 1., torch.tensor(0.0, device=x.device))
        out_conv1 = self.quant(torch.heaviside(self.dequant(self.conv1(out_bottom)) - 1., torch.tensor(0.0, device=x.device)))
        out_conv2 = self.quant(torch.heaviside(self.dequant(self.conv2(out_conv1)) - 1., torch.tensor(0.0, device=x.device)))
        out_conv3 = self.quant(torch.heaviside(self.dequant(self.conv3(out_conv2)) - 1., torch.tensor(0.0, device=x.device)))
        out_conv4 = self.quant(torch.heaviside(self.dequant(self.conv4(out_conv3)) - 1., torch.tensor(0.0, device=x.device)))

        # pass through residual blocks
        out_rconv = self.bottleneck(out_conv4)

        # gradually upsample while concatenating and passing through skip connections
        out_deconv4 = self.quant(torch.heaviside(self.dequant(self.deconv4(out_rconv)) - 1., torch.tensor(0.0, device=x.device)))
        out_add4 = self.skip_add.add(out_deconv4, out_conv3)
        depth4 = self.predict_depth4(out_add4) 

        out_deconv3 = self.quant(torch.heaviside(self.dequant(self.deconv3(out_add4)) - 1., torch.tensor(0.0, device=x.device)))
        out_add3 = self.skip_add.add(out_deconv3, out_conv2)
        depth3 = self.pred_add.add(depth4, self.predict_depth3(out_add3)) 

        out_deconv2 = self.quant(torch.heaviside(self.dequant(self.deconv2(out_add3)) - 1., torch.tensor(0.0, device=x.device)))
        out_add2 = self.skip_add.add(out_deconv2, out_conv1)
        depth2  = self.pred_add.add(depth3, self.predict_depth2(out_add2))

        out_deconv1 = self.quant(torch.heaviside(self.dequant(self.deconv1(out_add2)) - 1., torch.tensor(0.0, device=x.device)))
        out_add1 = self.skip_add.add(out_deconv1, out_bottom)
        depth1 = self.pred_add.add(depth2, self.predict_depth1(out_add1))

        # de-quantize output tensors
        depth4 = self.dequant(depth4)
        depth3 = self.dequant(depth3)
        depth2 = self.dequant(depth2)
        depth1 = self.dequant(depth1)
        out_rconv = self.dequant(out_rconv)
        out_add4 = self.dequant(out_add4)
        out_add3 = self.dequant(out_add3)
        out_add2 = self.dequant(out_add2)
        out_add1 = self.dequant(out_add1)

        # the membrane potentials of the output IF neuron carry the depth prediction
        return [depth1, depth2, depth3, depth4], [out_rconv, out_add4, out_add3, out_add2, out_add1]

class QUANTIZABLE_SIMPLIFIED_SteroSpike(NeuromorphicNet):
    """
    Based on the original Quantization-compatible version of the simplified model (QUANTIZABLE_SIMPLIFIED_fromZero_feedforward_multiscale_tempo_Matt_sepConv_SpikeFlowNetLike)
        - Modified to allow the ifmaps to retain the original integer representation.
        - Extract intermediate results before and after activation function (stem layer)

    """
    def __init__(self, input_chans=4, kernel_size=7, base_chans=32, use_plif=False, detach_reset=True, tau=10., v_threshold=1.0, v_reset=0.0, multiply_factor=1., surrogate_function=surrogate.Sigmoid(), learnable_biases=False):
        super().__init__(detach_reset=detach_reset)

        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.skip_add = nn.quantized.FloatFunctional()
        self.pred_add = nn.quantized.FloatFunctional()

        C = [base_chans * (2**n) for n in range(5)]
        K = kernel_size
        P = (kernel_size - 1) // 2
        self.multiply_factor = multiply_factor

        # bottom layer, preprocessing the input spike frame without downsampling
        self.bottom = nn.Sequential(
            # nn.Conv2d(in_channels=4, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.Conv2d(in_channels=input_chans, out_channels=input_chans, groups=input_chans, kernel_size=K, stride=1, padding=P, bias=False),
            nn.Conv2d(in_channels=input_chans, out_channels=C[0], kernel_size=1, stride=1, bias=False),
            #neuron.IFNode()
        )

        # encoder layers (downsampling)
        self.conv1 = nn.Sequential(
            # nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.Conv2d(in_channels=C[0], out_channels=C[0], groups=C[0], kernel_size=K, stride=2, padding=P, bias=False),
            nn.Conv2d(in_channels=C[0], out_channels=C[1], kernel_size=1, stride=1, bias=False),
            #neuron.IFNode()
        )
        self.conv2 = nn.Sequential(
            # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.Conv2d(in_channels=C[1], out_channels=C[1], groups=C[1], kernel_size=K, stride=2, padding=P, bias=False),
            nn.Conv2d(in_channels=C[1], out_channels=C[2], kernel_size=1, stride=1, bias=False),
            #neuron.IFNode(),
        )
        self.conv3 = nn.Sequential(
            # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2, bias=False),
            nn.Conv2d(in_channels=C[2], out_channels=C[2], groups=C[2], kernel_size=K, stride=2, padding=P, bias=False),
            nn.Conv2d(in_channels=C[2], out_channels=C[3], kernel_size=1, stride=1, bias=False),
            #neuron.IFNode()
        )
        self.conv4 = nn.Sequential(
            # nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2, bias=False),
            nn.Conv2d(in_channels=C[3], out_channels=C[3], groups=C[3], kernel_size=K, stride=2, padding=P, bias=False),
            nn.Conv2d(in_channels=C[3], out_channels=C[4], kernel_size=1, stride=1, bias=False),
            #neuron.IFNode(),
        )

        # residual layers
        self.bottleneck = nn.Sequential(
            QUANTIZABLE_SIMPLIFIED_SeparableSEWResBlock(C[4], kernel_size=K, v_threshold=v_threshold, v_reset=v_reset, connect_function='ADD', multiply_factor=multiply_factor, use_plif=True, tau=tau, surrogate_function=surrogate_function),
            QUANTIZABLE_SIMPLIFIED_SeparableSEWResBlock(C[4], kernel_size=K, v_threshold=v_threshold, v_reset=v_reset, connect_function='ADD', multiply_factor=multiply_factor, use_plif=True, tau=tau, surrogate_function=surrogate_function),
        )

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            SeparableNNConvUpsampling(in_channels=C[4], out_channels=C[3], kernel_size=K, up_size=(33, 44)),
            #neuron.IFNode()
        )
        self.deconv3 = nn.Sequential(
            SeparableNNConvUpsampling(in_channels=C[3], out_channels=C[2], kernel_size=K, up_size=(65, 87)),
            #neuron.IFNode()
        )
        self.deconv2 = nn.Sequential(
            SeparableNNConvUpsampling(in_channels=C[2], out_channels=C[1], kernel_size=K, up_size=(130, 173)),
            #neuron.IFNode()
        )
        self.deconv1 = nn.Sequential(
            SeparableNNConvUpsampling(in_channels=C[1], out_channels=C[0], kernel_size=K, up_size=(260, 346)),
            #neuron.IFNode()
        )

        # these layers output depth maps at different scales, where depth is represented by the potential of IF neurons
        # that do not fire ("I-neurons"), i.e., with an infinite threshold.
        self.predict_depth4 = SeparableNNConvUpsampling(in_channels=C[3], out_channels=1, kernel_size=K, up_size=(260, 346), bias=True)
        self.predict_depth3 = SeparableNNConvUpsampling(in_channels=C[2], out_channels=1, kernel_size=K, up_size=(260, 346), bias=True)
        self.predict_depth2 = SeparableNNConvUpsampling(in_channels=C[1], out_channels=1, kernel_size=K, up_size=(260, 346), bias=True)
        self.predict_depth1 = SeparableNNConvUpsampling(in_channels=C[0], out_channels=1, kernel_size=K, up_size=(260, 346), bias=True)

        # learn the reset potentials of output I-neurons
        self.learnable_biases = learnable_biases
        self.biases = Parameter(torch.zeros((260, 1)))


    @staticmethod
    def reformat_input_data(warmup_chunks_left, warmup_chunks_right, inference_chunks_left, inference_chunks_right):

        # get the dimensions of the data: (B, N, nfpdm, P, H, W)
        B, N_inference, nfpdm, P, H, W = inference_chunks_left.shape
        N_warmup = warmup_chunks_left.shape[1]

        """
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
        warmup_chunks = torch.cat((warmup_chunks_left, warmup_chunks_right), dim=2)
        inference_chunks = torch.cat((inference_chunks_left, inference_chunks_right), dim=2)

        return warmup_chunks, inference_chunks

    def forward(self, x):
        # x must be of shape [batch_size, num_frames_per_depth_map, 2 (polarities), W, H]

        if self.learnable_biases == True:
            self.set_output_potentials(x)

        # quantize input tensor
        # x = self.quant(x)
        # quantize input tensor, but preserve the original integer value that it has
        # Quantization formula: q = round(x/scale) + zero_point
        scale_int = 1
        zero_point_int = 0
        # Convert the tensor to a quantized tensor using per-tensor quantization
        x = torch.quantize_per_tensor(x, scale_int, zero_point_int, torch.quint8)

        # Eliminate the batch dimension
        frame = x[:, 0, :, :, :]

        # # data is fed in through the bottom layer
        # out_bottom = self.quant(torch.heaviside(self.dequant(self.bottom(frame)) - 1., torch.tensor(0.0, device=x.device)))

        """
        Modified stem layer
        """
        out_bottom_int = self.bottom(frame)
        out_bottom_fp = self.dequant(out_bottom_int)
        out_bottom_act = torch.heaviside(out_bottom_fp - 1., torch.tensor(0.0, device=x.device))
        out_bottom = self.quant(out_bottom_act)

        # pass through encoder layers
        #out_conv1 = torch.heaviside(self.conv1(out_bottom) - 1., torch.tensor(0.0, device=x.device))
        out_conv1 = self.quant(torch.heaviside(self.dequant(self.conv1(out_bottom)) - 1., torch.tensor(0.0, device=x.device)))
        out_conv2 = self.quant(torch.heaviside(self.dequant(self.conv2(out_conv1)) - 1., torch.tensor(0.0, device=x.device)))
        out_conv3 = self.quant(torch.heaviside(self.dequant(self.conv3(out_conv2)) - 1., torch.tensor(0.0, device=x.device)))
        out_conv4 = self.quant(torch.heaviside(self.dequant(self.conv4(out_conv3)) - 1., torch.tensor(0.0, device=x.device)))

        # pass through residual blocks
        out_rconv = self.bottleneck(out_conv4)

        # gradually upsample while concatenating and passing through skip connections
        out_deconv4 = self.quant(torch.heaviside(self.dequant(self.deconv4(out_rconv)) - 1., torch.tensor(0.0, device=x.device)))
        out_add4 = self.skip_add.add(out_deconv4, out_conv3)
        depth4 = self.predict_depth4(out_add4) 

        out_deconv3 = self.quant(torch.heaviside(self.dequant(self.deconv3(out_add4)) - 1., torch.tensor(0.0, device=x.device)))
        out_add3 = self.skip_add.add(out_deconv3, out_conv2)
        depth3 = self.pred_add.add(depth4, self.predict_depth3(out_add3)) 

        out_deconv2 = self.quant(torch.heaviside(self.dequant(self.deconv2(out_add3)) - 1., torch.tensor(0.0, device=x.device)))
        out_add2 = self.skip_add.add(out_deconv2, out_conv1)
        depth2  = self.pred_add.add(depth3, self.predict_depth2(out_add2))

        out_deconv1 = self.quant(torch.heaviside(self.dequant(self.deconv1(out_add2)) - 1., torch.tensor(0.0, device=x.device)))
        out_add1 = self.skip_add.add(out_deconv1, out_bottom)
        depth1 = self.pred_add.add(depth2, self.predict_depth1(out_add1))

        # de-quantize output tensors
        depth4 = self.dequant(depth4)
        depth3 = self.dequant(depth3)
        depth2 = self.dequant(depth2)
        depth1 = self.dequant(depth1)
        out_rconv = self.dequant(out_rconv)
        out_add4 = self.dequant(out_add4)
        out_add3 = self.dequant(out_add3)
        out_add2 = self.dequant(out_add2)
        out_add1 = self.dequant(out_add1)

        # the membrane potentials of the output IF neuron carry the depth prediction
        return (
                [depth1, depth2, depth3, depth4], 
                [out_rconv, out_add4, out_add3, out_add2, out_add1],
                [frame, out_bottom],
                [out_bottom_int],
                [out_bottom_fp],
                [out_bottom_act]
               )