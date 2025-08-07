'''
2025-03-10
Andres Brito
    - Simplified Full-Precision Model without Skip Connections from Tomo's implementation.
'''

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from spikingjelly.clock_driven import neuron, layer, surrogate

from .blocks import SeparableNNConvUpsampling

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


class Simplified_StereoSpike_NoSkip(NeuromorphicNet):
    """
    Simplified but mathematically equivalent version of 'fromZero_feedforward_multiscale_tempo_Matt_sepConv_SpikeFlowNetLike' with no skip connections.
        - A dictionay is included to store internal variables for statistical analysis.
    """
    def __init__(self, input_chans=4, kernel_size=7, base_chans=32, use_plif=False, detach_reset=True, tau=10., v_threshold=1.0, v_reset=0.0, multiply_factor=1., surrogate_function=surrogate.Sigmoid(), learnable_biases=False):
        super().__init__(detach_reset=detach_reset)

        # Dictionary to store ofmaps after activation function (binary ofmaps)
        self.activations = {}

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
            SIMPLIFIED_SeparableSEWResBlock_noskip(C[4], kernel_size=K, v_threshold=v_threshold, v_reset=v_reset, multiply_factor=multiply_factor, use_plif=True, tau=tau, surrogate_function=surrogate_function),
            SIMPLIFIED_SeparableSEWResBlock_noskip(C[4], kernel_size=K, v_threshold=v_threshold, v_reset=v_reset, multiply_factor=multiply_factor, use_plif=True, tau=tau, surrogate_function=surrogate_function),
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
        # Track ofmap from Stem Layer
        self.activations["out_bottom"] = out_bottom.detach().cpu()

        # pass through encoder layers
        out_conv1 = torch.heaviside(self.conv1(out_bottom) - 1., torch.tensor(0.0, device=x.device))
        # Track ofmap from conv1 Layer
        self.activations["out_conv1"] = out_conv1.detach().cpu()
        out_conv2 = torch.heaviside(self.conv2(out_conv1) - 1., torch.tensor(0.0, device=x.device))
        # Track ofmap from conv2 Layer
        self.activations["out_conv2"] = out_conv2.detach().cpu()
        out_conv3 = torch.heaviside(self.conv3(out_conv2) - 1., torch.tensor(0.0, device=x.device))
        # Track ofmap from conv3 Layer
        self.activations["out_conv3"] = out_conv3.detach().cpu()
        out_conv4 = torch.heaviside(self.conv4(out_conv3) - 1., torch.tensor(0.0, device=x.device))
        # Track ofmap from conv4 Layer
        self.activations["out_conv4"] = out_conv4.detach().cpu()

        # pass through residual blocks
        out_rconv = self.bottleneck(out_conv4)
        # Inside the bottleneck there are 2 layers, we need to retrieve the ofmaps that were already stored
        for i in range(2):
            for name, activation in self.bottleneck[i].activations.items():
                self.activations['b' + str(i+1) + '_' + name] = activation 


        # gradually upsample while concatenating and passing through skip connections
        out_deconv4 = torch.heaviside(self.deconv4(out_rconv) - 1., torch.tensor(0.0, device=x.device))
        # Track ofmap from deconv4 Layer
        self.activations["out_deconv4"] = out_deconv4.detach().cpu()
        
        out_add4 = out_deconv4 #+ out_conv3
        depth4 = self.predict_depth4(out_add4) # * self.multiply_factor
        # Track ofmap from predict4 Layer (no accumulation)
        self.activations["out_depth4"] = depth4.detach().cpu()
        #self.Ineurons(self.predict_depth4(out_add4) * self.multiply_factor)
        #depth4 = self.Ineurons.v

        out_deconv3 = torch.heaviside(self.deconv3(out_add4) - 1., torch.tensor(0.0, device=x.device))
        # Track ofmap from deconv3 Layer
        self.activations["out_deconv3"] = out_deconv3.detach().cpu()

        out_add3 = out_deconv3 #+ out_conv2
        out_depth3 = self.predict_depth3(out_add3) # * self.multiply_factor
        depth3 = depth4 + out_depth3
        # Track ofmap from predict3 Layer (no accumulation)
        self.activations["out_depth3"] = out_depth3.detach().cpu()
        #self.Ineurons(self.predict_depth3(out_add3) * self.multiply_factor)
        #depth3 = self.Ineurons.v

        out_deconv2 = torch.heaviside(self.deconv2(out_add3) - 1., torch.tensor(0.0, device=x.device))
        # Track ofmap from deconv2 Layer
        self.activations["out_deconv2"] = out_deconv2.detach().cpu()

        out_add2 = out_deconv2 #+ out_conv1
        out_depth2 = self.predict_depth2(out_add2) # * self.multiply_factor
        depth2  = depth3 + out_depth2
        # Track ofmap from predict2 Layer (no accumulation)
        self.activations["out_depth2"] = out_depth2.detach().cpu()
        #self.Ineurons(self.predict_depth2(out_add2) * self.multiply_factor)
        #depth2 = self.Ineurons.v

        out_deconv1 = torch.heaviside(self.deconv1(out_add2) - 1., torch.tensor(0.0, device=x.device))
        # Track ofmap from deconv1 Layer
        self.activations["out_deconv1"] = out_deconv1.detach().cpu()

        out_add1 = out_deconv1 #+ out_bottom
        out_depth1 = self.predict_depth1(out_add1) # * self.multiply_factor
        depth1 = depth2 + out_depth1
        # Track ofmap from predict1 Layer (no accumulation)
        self.activations["out_depth1"] = out_depth1.detach().cpu()
        #self.Ineurons(self.predict_depth1(out_add1) * self.multiply_factor)
        #depth1 = self.Ineurons.v

        # the membrane potentials of the output IF neuron carry the depth prediction
        return [depth1, depth2, depth3, depth4], [out_rconv, out_add4, out_add3, out_add2, out_add1]
    
class SIMPLIFIED_SeparableSEWResBlock_noskip(nn.Module):
    """
    Version of the SEW-Resblock using Separable Convolutions (No Skip Connections)
        - Include a dictionary to store internal variables for statistical analysis.
    """

    def __init__(self, in_channels: int, connect_function='ADD', v_threshold=1., v_reset=0.,
                 surrogate_function=surrogate.Sigmoid(), use_plif=False, tau=2., multiply_factor=1., kernel_size=3):
        super(SIMPLIFIED_SeparableSEWResBlock_noskip, self).__init__()

        # Dictionary to store ofmaps after activation function (binary ofmaps)
        self.activations = {}
        
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
        # Track ofmap from Bottleneck conv1 Layer
        self.activations["conv1"] = out.detach().cpu()
        out = self.conv2(out)
        out = torch.heaviside(out - 1, torch.tensor(1., device=x.device))  # self.sn2(out)
        # Track ofmap from Bottleneck conv2 Layer
        self.activations["conv2"] = out.detach().cpu()

        return out