import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from spikingjelly.clock_driven import neuron, layer, surrogate

from .blocks import SEWResBlock, NNConvUpsampling, MultiplyBy, SeparableSEWResBlock, SeparableNNConvUpsampling
#from .blocks import InferenceOnlyHeaviside, QuantizedSeparableSEWResBlock
#from .blocks import forFLOPSSeparableSEWResBlock, forFLOPSSeparableNNConvUpsampling


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


class StereoSpike(NeuromorphicNet):
    """
    Baseline model, with which we report state-of-the-art performances in the second version of our paper.

    - all neuron potentials must be reset at each timestep
    - predict_depth layers do have biases, but it is equivalent to remove them and reset output I-neurons to the sum
           of all 4 biases, instead of 0.
    """
    def __init__(self, surrogate_function=surrogate.ATan(), detach_reset=True, v_threshold=1.0, v_reset=0.0, multiply_factor=1.):
        super().__init__(surrogate_function=surrogate_function, detach_reset=detach_reset)

        # bottom layer, preprocessing the input spike frame without downsampling
        self.bottom = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False),
            MultiplyBy(multiply_factor),
            neuron.IFNode(v_threshold=self.v_th, v_reset=self.v_rst, surrogate_function=self.surrogate_fct, detach_reset=True),
        )

        # encoder layers (downsampling)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2, bias=False),
            MultiplyBy(multiply_factor),
            neuron.IFNode(v_threshold=self.v_th, v_reset=self.v_rst, surrogate_function=self.surrogate_fct, detach_reset=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2, bias=False),
            MultiplyBy(multiply_factor),
            neuron.IFNode(v_threshold=self.v_th, v_reset=self.v_rst, surrogate_function=self.surrogate_fct, detach_reset=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2, bias=False),
            MultiplyBy(multiply_factor),
            neuron.IFNode(v_threshold=self.v_th, v_reset=self.v_rst, surrogate_function=self.surrogate_fct, detach_reset=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2, bias=False),
            MultiplyBy(multiply_factor),
            neuron.IFNode(v_threshold=self.v_th, v_reset=self.v_rst, surrogate_function=self.surrogate_fct, detach_reset=True),
        )

        # residual layers
        self.bottleneck = nn.Sequential(
            SEWResBlock(512, v_threshold=self.v_th, v_reset=self.v_rst, connect_function='ADD', multiply_factor=multiply_factor),
            SEWResBlock(512, v_threshold=self.v_th, v_reset=self.v_rst, connect_function='ADD', multiply_factor=multiply_factor),
        )

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            NNConvUpsampling(in_channels=512, out_channels=256, kernel_size=5, up_size=(33, 44)),
            MultiplyBy(multiply_factor),
            neuron.IFNode(v_threshold=self.v_th, v_reset=self.v_rst, surrogate_function=self.surrogate_fct, detach_reset=True),
        )
        self.deconv3 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=128, kernel_size=5, up_size=(65, 87)),
            MultiplyBy(multiply_factor),
            neuron.IFNode(v_threshold=self.v_th, v_reset=self.v_rst, surrogate_function=self.surrogate_fct, detach_reset=True),
        )
        self.deconv2 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=64, kernel_size=5, up_size=(130, 173)),
            MultiplyBy(multiply_factor),
            neuron.IFNode(v_threshold=self.v_th, v_reset=self.v_rst, surrogate_function=self.surrogate_fct, detach_reset=True),
        )
        self.deconv1 = nn.Sequential(
            NNConvUpsampling(in_channels=64, out_channels=32, kernel_size=5, up_size=(260, 346)),
            MultiplyBy(multiply_factor),
            neuron.IFNode(v_threshold=self.v_th, v_reset=self.v_rst, surrogate_function=self.surrogate_fct, detach_reset=True),
        )

        # these layers output depth maps at different scales, where depth is represented by the potential of IF neurons
        # that do not fire ("I-neurons"), i.e., with an infinite threshold.
        self.predict_depth4 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=1, kernel_size=3, up_size=(260, 346), bias=True),
            MultiplyBy(multiply_factor),
        )
        self.predict_depth3 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=1, kernel_size=3, up_size=(260, 346), bias=True),
            MultiplyBy(multiply_factor),
        )
        self.predict_depth2 = nn.Sequential(
            NNConvUpsampling(in_channels=64, out_channels=1, kernel_size=3, up_size=(260, 346), bias=True),
            MultiplyBy(multiply_factor),
        )
        self.predict_depth1 = nn.Sequential(
            NNConvUpsampling(in_channels=32, out_channels=1, kernel_size=3, up_size=(260, 346), bias=True),
            MultiplyBy(multiply_factor),
        )

        self.Ineurons = neuron.IFNode(v_threshold=float('inf'), v_reset=0.0, surrogate_function=self.surrogate_fct)

    def forward(self, x, save_spike_tensors=False):

        # x must be of shape [batch_size, num_frames_per_depth_map, 4 (2 cameras - 2 polarities), W, H]
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
        # also output intermediate spike tensors
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
            'out_combined': 0.,
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

    def set_init_depths_potentials(self, depth_prior):
        self.Ineurons.v = depth_prior


####################
# ABLATION STUDIES #
####################

class fromZero_feedforward_multiscale_tempo_Matt_noskip_SpikeFlowNetLike(NeuromorphicNet):
    """
    model that makes intermediate predictions at different times thanks to a pool of neuron that is common.

    IT IS FULLY SPIKING AND HAS THE BEST MDE WE'VE SEEN SO FAR !!!

    """
    def __init__(self, use_plif=False, detach_reset=True, tau=10., v_threshold=1.0, v_reset=0.0, final_activation=nn.Identity, multiply_factor=1.):
        super().__init__(use_plif=use_plif, detach_reset=detach_reset)

        self.is_cext_model = False

        # bottom layer, preprocessing the input spike frame without downsampling
        self.bottom = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # encoder layers (downsampling)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2, bias=False),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2, bias=False),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2, bias=False),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2, bias=False),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # residual layers
        self.bottleneck = nn.Sequential(
            SEWResBlock(512, tau=tau, v_threshold=v_threshold, v_reset=v_reset, connect_function='ADD', use_plif=use_plif, multiply_factor=multiply_factor),
            SEWResBlock(512, tau=tau, v_threshold=v_threshold, v_reset=v_reset, connect_function='ADD', use_plif=use_plif, multiply_factor=multiply_factor),
        )

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            NNConvUpsampling(in_channels=512, out_channels=256, kernel_size=5, up_size=(33, 44)),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.deconv3 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=128, kernel_size=5, up_size=(65, 87)),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.deconv2 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=64, kernel_size=5, up_size=(130, 173)),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.deconv1 = nn.Sequential(
            NNConvUpsampling(in_channels=64, out_channels=32, kernel_size=5, up_size=(260, 346)),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # these layers output depth maps at different scales, where depth is represented by the potential of IF neurons
        # that do not fire ("I-neurons"), i.e., with an infinite threshold.
        self.predict_depth4 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=1, kernel_size=3, up_size=(260, 346), bias=True),
            MultiplyBy(multiply_factor),
        )
        self.predict_depth3 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=1, kernel_size=3, up_size=(260, 346), bias=True),
            MultiplyBy(multiply_factor),
        )
        self.predict_depth2 = nn.Sequential(
            NNConvUpsampling(in_channels=64, out_channels=1, kernel_size=3, up_size=(260, 346), bias=True),
            MultiplyBy(multiply_factor),
        )
        self.predict_depth1 = nn.Sequential(
            NNConvUpsampling(in_channels=32, out_channels=1, kernel_size=3, up_size=(260, 346), bias=True),
            MultiplyBy(multiply_factor),
        )

        self.Ineurons = neuron.IFNode(v_threshold=float('inf'), v_reset=v_reset, surrogate_function=surrogate.ATan())

        self.final_activation = final_activation

    def forward(self, x):
        # x must be of shape [batch_size, num_frames_per_depth_map, 2 (polarities), W, H]

        frame = x[:, 0, :, :, :]
        print(frame.shape)

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
        # NO SKIP CONNECTION
        out_deconv4 = self.deconv4(out_rconv)
        out_add4 = out_deconv4 #+ out_conv3
        self.Ineurons(self.predict_depth4(out_add4))
        depth4 = self.Ineurons.v

        out_deconv3 = self.deconv3(out_add4)
        out_add3 = out_deconv3 #+ out_conv2
        self.Ineurons(self.predict_depth3(out_add3))
        depth3 = self.Ineurons.v

        out_deconv2 = self.deconv2(out_add3)
        out_add2 = out_deconv2 #+ out_conv1
        self.Ineurons(self.predict_depth2(out_add2))
        depth2 = self.Ineurons.v

        out_deconv1 = self.deconv1(out_add2)
        out_add1 = out_deconv1 #+ out_bottom
        self.Ineurons(self.predict_depth1(out_add1))
        depth1 = self.Ineurons.v

        # the membrane potentials of the output IF neuron carry the depth prediction
        return [depth1, depth2, depth3, depth4]

    def set_init_depths_potentials(self, depth_prior):
        self.Ineurons.v = depth_prior


class fromZero_feedforward_multiscale_tempo_Matt_cutpredict_SpikeFlowNetLike(NeuromorphicNet):
    """
    Removed deepest prediction layer
    """
    def __init__(self, use_plif=False, detach_reset=True, tau=10., v_threshold=1.0, v_reset=0.0, final_activation=nn.Identity, multiply_factor=1.):
        super().__init__(use_plif=use_plif, detach_reset=detach_reset)

        self.is_cext_model = False

        # bottom layer, preprocessing the input spike frame without downsampling
        self.bottom = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # encoder layers (downsampling)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2, bias=False),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2, bias=False),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2, bias=False),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2, bias=False),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # residual layers
        self.bottleneck = nn.Sequential(
            SEWResBlock(512, tau=tau, v_threshold=v_threshold, v_reset=v_reset, connect_function='ADD', use_plif=use_plif, multiply_factor=multiply_factor),
            SEWResBlock(512, tau=tau, v_threshold=v_threshold, v_reset=v_reset, connect_function='ADD', use_plif=use_plif, multiply_factor=multiply_factor),
        )

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            NNConvUpsampling(in_channels=512, out_channels=256, kernel_size=5, up_size=(33, 44)),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.deconv3 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=128, kernel_size=5, up_size=(65, 87)),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.deconv2 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=64, kernel_size=5, up_size=(130, 173)),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.deconv1 = nn.Sequential(
            NNConvUpsampling(in_channels=64, out_channels=32, kernel_size=5, up_size=(260, 346)),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # these layers output depth maps at different scales, where depth is represented by the potential of IF neurons
        # that do not fire ("I-neurons"), i.e., with an infinite threshold.
        self.predict_depth4 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=1, kernel_size=3, up_size=(260, 346), bias=True),
            MultiplyBy(multiply_factor),
        )
        self.predict_depth3 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=1, kernel_size=3, up_size=(260, 346), bias=True),
            MultiplyBy(multiply_factor),
        )
        self.predict_depth2 = nn.Sequential(
            NNConvUpsampling(in_channels=64, out_channels=1, kernel_size=3, up_size=(260, 346), bias=True),
            MultiplyBy(multiply_factor),
        )
        self.predict_depth1 = nn.Sequential(
            NNConvUpsampling(in_channels=32, out_channels=1, kernel_size=3, up_size=(260, 346), bias=True),
            MultiplyBy(multiply_factor),
        )

        self.Ineurons = neuron.IFNode(v_threshold=float('inf'), v_reset=v_reset, surrogate_function=surrogate.ATan())

        self.final_activation = final_activation

    def forward(self, x):
        # x must be of shape [batch_size, num_frames_per_depth_map, 2 (polarities), W, H]

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
        #self.Ineurons(self.predict_depth4(out_add4))
        #depth4 = self.Ineurons.v

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
        return [depth1, depth2, depth3]#, depth4]

    def set_init_depths_potentials(self, depth_prior):
        self.Ineurons.v = depth_prior


class ADDOUTPUT_fromZero_feedforward_multiscale_tempo_Matt_sepConv_SpikeFlowNetLike(NeuromorphicNet):
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

        depth4, depth3, depth2, depth1 = torch.zeros(x.shape[0], 1, 260, 346)
        
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
        depth4 = self.predict_depth4(out_add4) 

        #self.Ineurons(self.predict_depth4(out_add4))
        #depth4 = self.Ineurons.v

        out_deconv3 = self.deconv3(out_add4)
        out_add3 = out_deconv3 + out_conv2
        depth3 = depth4 + self.predict_depth3(out_add3)
        
        #self.Ineurons(self.predict_depth3(out_add3))
        #depth3 = self.Ineurons.v

        out_deconv2 = self.deconv2(out_add3)
        out_add2 = out_deconv2 + out_conv1
        depth2 = depth3 + self.predict_depth2(out_add2)
        
        #self.Ineurons(self.predict_depth2(out_add2))
        #depth2 = self.Ineurons.v

        out_deconv1 = self.deconv1(out_add2)
        out_add1 = out_deconv1 + out_bottom
        depth1 = depth2 + self.predict_depth1(out_add1)

        #self.Ineurons(self.predict_depth1(out_add1))
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
            'out_combined': 0.,
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

      
class PixelShuffle_fromZero_feedforward_multiscale_tempo_Matt_sepConv_SpikeFlowNetLike(NeuromorphicNet):
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
            #SeparableNNConvUpsampling(in_channels=C[3], out_channels=1, kernel_size=K, up_size=(260, 346), bias=True),
            #MultiplyBy(multiply_factor),
            nn.Conv2d(in_channels=C[3], out_channels=C[3], groups=C[3], kernel_size=K, stride=1, padding=P, bias=True),
            nn.Conv2d(in_channels=C[3], out_channels=64, kernel_size=1, stride=1, bias=True),
            MultiplyBy(multiply_factor),
            nn.PixelShuffle(8)
        )
        self.predict_depth3 = nn.Sequential(
            #SeparableNNConvUpsampling(in_channels=C[2], out_channels=1, kernel_size=K, up_size=(260, 346), bias=True),
            #MultiplyBy(multiply_factor),
            nn.Conv2d(in_channels=C[2], out_channels=C[2], groups=C[2], kernel_size=K, stride=1, padding=P, bias=True),
            nn.Conv2d(in_channels=C[2], out_channels=16, kernel_size=1, stride=1, bias=True),
            MultiplyBy(multiply_factor),
            nn.PixelShuffle(4)
        )
        self.predict_depth2 = nn.Sequential(
            #SeparableNNConvUpsampling(in_channels=C[1], out_channels=1, kernel_size=K, up_size=(260, 346), bias=True),
            #MultiplyBy(multiply_factor),
            nn.Conv2d(in_channels=C[1], out_channels=C[1], groups=C[1], kernel_size=K, stride=1, padding=P, bias=True),
            nn.Conv2d(in_channels=C[1], out_channels=4, kernel_size=1, stride=1, bias=True),
            MultiplyBy(multiply_factor),
            nn.PixelShuffle(2)
        )
        self.predict_depth1 = nn.Sequential(
            SeparableNNConvUpsampling(in_channels=C[0], out_channels=1, kernel_size=K, up_size=(260, 346), bias=True),
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
        self.Ineurons(self.predict_depth4(out_add4)[:,:,0:260, 0:346])
        depth4 = self.Ineurons.v

        out_deconv3 = self.deconv3(out_add4)
        out_add3 = out_deconv3 + out_conv2
        pred3 = self.predict_depth3(out_add3)
        self.Ineurons(self.predict_depth3(out_add3)[:,:,0:260, 0:346])
        depth3 = self.Ineurons.v

        out_deconv2 = self.deconv2(out_add3)
        out_add2 = out_deconv2 + out_conv1
        self.Ineurons(self.predict_depth2(out_add2)[:,:,0:260, 0:346])
        depth2 = self.Ineurons.v

        out_deconv1 = self.deconv1(out_add2)
        out_add1 = out_deconv1 + out_bottom
        self.Ineurons(self.predict_depth1(out_add1)[:,:,0:260, 0:346])
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

        self.Ineurons = neuron.IFNode(v_threshold=float('inf'), v_reset=v_reset, surrogate_function=surrogate.ATan())
        
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


class fromZero_feedforward_multiscale_tempo_Matt_sepConv_SpikeFlowNetLike_withoutSkipConnection(NeuromorphicNet):
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

        self.Ineurons = neuron.IFNode(v_threshold=float('inf'), v_reset=v_reset, surrogate_function=surrogate.ATan())
        
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
        out_add4 = out_deconv4 # + out_conv3
        self.Ineurons(self.predict_depth4(out_add4))
        depth4 = self.Ineurons.v

        out_deconv3 = self.deconv3(out_add4)
        out_add3 = out_deconv3 # + out_conv2
        self.Ineurons(self.predict_depth3(out_add3))
        depth3 = self.Ineurons.v

        out_deconv2 = self.deconv2(out_add3)
        out_add2 = out_deconv2 # + out_conv1
        self.Ineurons(self.predict_depth2(out_add2))
        depth2 = self.Ineurons.v

        out_deconv1 = self.deconv1(out_add2)
        out_add1 = out_deconv1 # + out_bottom
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


class fromZero_feedforward_monocular_multiscale_tempo_Matt_sepConv_SpikeFlowNetLike(NeuromorphicNet):
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
        # that do not fire ("I-neurons"), i.e., with an infinite threshold
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

        self.Ineurons = neuron.IFNode(v_threshold=float('inf'), v_reset=v_reset, surrogate_function=surrogate.ATan())

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

        return warmup_chunks_left, inference_chunks_left

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
            'out_combined': 0.,
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
                m.weight = Parameter(m.weight * factor)
                if m.bias is not None:
                    m.bias = Parameter(m.bias * 10)

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




####################
# ABLATION STUDIES #
####################

from torch.quantization import QuantStub, DeQuantStub


class InferenceOnlyHeaviside(nn.Module):

    def __init__(self, v_threshold=1., tau=3.):
        super(InferenceOnlyHeaviside, self).__init__()
        self.v_threshold = v_threshold
        self.tau = tau

    def forward(self, x):
        if x.dtype == torch.quint8:
            value = torch.quantize_per_tensor(torch.Tensor([0.]), 1., 0, torch.quint8)
            result = torch.zeros_like(x)
            #result = torch.quantize_per_tensor(torch.zeros_like(x), 1., 0, torch.quint8)
        else:
            value = torch.Tensor([0.])
            result = torch.zeros_like(x)
        torch.gt(x, value, out=result)
        return result


class quantized_OPT_fromZero_feedforward_multiscale_tempo_Matt_sepConv_SpikeFlowNetLike(NeuromorphicNet):
    """
    Uses separable convolutions for a lighter model

    See this excellent article to know moreabout separable convolutions:
    https://www.paepper.com/blog/posts/depthwise-separable-convolutions-in-pytorch/
    """

    def __init__(self, input_chans=4, kernel_size=7, base_chans=32, use_plif=False, detach_reset=True, tau=10.,
                 v_threshold=1.0, v_reset=0.0, multiply_factor=1., surrogate_function=surrogate.Sigmoid(),
                 learnable_biases=False):
        super().__init__(detach_reset=detach_reset)

        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.skip_add = nn.quantized.FloatFunctional()
        self.pred_add = nn.quantized.FloatFunctional()

        C = [base_chans * (2 ** n) for n in range(5)]
        K = kernel_size
        P = (kernel_size - 1) // 2

        # bottom layer, preprocessing the input spike frame without downsampling
        self.bottom = nn.Sequential(
            nn.Conv2d(in_channels=input_chans, out_channels=input_chans, groups=input_chans, kernel_size=K, stride=1, padding=P, bias=False),
            nn.Conv2d(in_channels=input_chans, out_channels=C[0], kernel_size=1, stride=1, bias=False),
        )

        # encoder layers (downsampling)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=C[0], out_channels=C[0], groups=C[0], kernel_size=K, stride=2, padding=P, bias=False),
            nn.Conv2d(in_channels=C[0], out_channels=C[1], kernel_size=1, stride=1, bias=False),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=C[1], out_channels=C[1], groups=C[1], kernel_size=K, stride=2, padding=P, bias=False),
            nn.Conv2d(in_channels=C[1], out_channels=C[2], kernel_size=1, stride=1, bias=False),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=C[2], out_channels=C[2], groups=C[2], kernel_size=K, stride=2, padding=P, bias=False),
            nn.Conv2d(in_channels=C[2], out_channels=C[3], kernel_size=1, stride=1, bias=False),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=C[3], out_channels=C[3], groups=C[3], kernel_size=K, stride=2, padding=P, bias=False),
            nn.Conv2d(in_channels=C[3], out_channels=C[4], kernel_size=1, stride=1, bias=False),
        )

        # residual layers
        """
        self.bottleneck = nn.Sequential(
            QuantizedSeparableSEWResBlock(C[4], kernel_size=K, v_threshold=v_threshold, v_reset=v_reset, connect_function='ADD',
                                 multiply_factor=multiply_factor, use_plif=True, tau=tau,
                                 surrogate_function=surrogate_function),
            QuantizedSeparableSEWResBlock(C[4], kernel_size=K, v_threshold=v_threshold, v_reset=v_reset, connect_function='ADD',
                                 multiply_factor=multiply_factor, use_plif=True, tau=tau,
                                 surrogate_function=surrogate_function),
        )
        """

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            SeparableNNConvUpsampling(in_channels=C[4], out_channels=C[3], kernel_size=K, up_size=(33, 44)),
        )
        self.deconv3 = nn.Sequential(
            SeparableNNConvUpsampling(in_channels=C[3], out_channels=C[2], kernel_size=K, up_size=(65, 87)),
        )
        self.deconv2 = nn.Sequential(
            SeparableNNConvUpsampling(in_channels=C[2], out_channels=C[1], kernel_size=K, up_size=(130, 173)),
        )
        self.deconv1 = nn.Sequential(
            SeparableNNConvUpsampling(in_channels=C[1], out_channels=C[0], kernel_size=K, up_size=(260, 346)),
        )

        # these layers output depth maps at different scales, where depth is represented by the potential of IF neurons
        # that do not fire ("I-neurons"), i.e., with an infinite threshold.
        self.predict_depth4 = nn.Sequential(
            SeparableNNConvUpsampling(in_channels=C[3], out_channels=1, kernel_size=K, up_size=(260, 346), bias=True),
        )
        self.predict_depth3 = nn.Sequential(
            SeparableNNConvUpsampling(in_channels=C[2], out_channels=1, kernel_size=K, up_size=(260, 346), bias=True),
        )
        self.predict_depth2 = nn.Sequential(
            SeparableNNConvUpsampling(in_channels=C[1], out_channels=1, kernel_size=K, up_size=(260, 346), bias=True),
        )
        self.predict_depth1 = nn.Sequential(
            SeparableNNConvUpsampling(in_channels=C[0], out_channels=1, kernel_size=K, up_size=(260, 346), bias=True),
        )

        # learn the reset potentials of output I-neurons
        self.learnable_biases = learnable_biases
        self.biases = Parameter(torch.zeros((260, 1)))

    @staticmethod
    def reformat_input_data(warmup_chunks_left, warmup_chunks_right, inference_chunks_left, inference_chunks_right):
        # get the dimensions of the data: (B, N, nfpdm, P, H, W)
        B, N_inference, nfpdm, P, H, W = inference_chunks_left.shape

        # sum consecutive chunks (B, N, nfpdm, 2, H, W) --> (B, nfpdm, 2, 260, 346)
        warmup_chunks_left = torch.sum(warmup_chunks_left, dim=1)
        warmup_chunks_right = torch.sum(warmup_chunks_right, dim=1)
        inference_chunks_left = torch.sum(inference_chunks_left, dim=1)
        inference_chunks_right = torch.sum(inference_chunks_right, dim=1)

        # concatenate frames channel-wise: (B, nfpdm, 2, 260, 346) --> (B, 1, num_frames*2, 260, 346)
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

        depth1 = torch.zeros(x.shape[0], 1, 260, 346)
        depth2 = torch.zeros(x.shape[0], 1, 260, 346)
        depth3 = torch.zeros(x.shape[0], 1, 260, 346) 
        depth4 = torch.zeros(x.shape[0], 1, 260, 346)

        # quantize input tensor
        x = self.quant(x)

        frame = x[:, 0, :, :, :]

        # data is fed in through the bottom layer
        out_bottom = self.bottom(frame)
        out_bottom_S = torch.zeros(out_bottom.shape)
        torch.gt(out_bottom, torch.zeros(out_bottom.shape), out_bottom_S)

        # pass through encoder layers
        out_conv1 = self.conv1(out_bottom)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)

        # pass through residual blocks
        out_rconv = out_conv4  # self.bottleneck(out_conv4)

        # gradually upsample while concatenating and passing through skip connections
        out_deconv4 = self.deconv4(out_rconv)
        out_add4 = self.skip_add.add(out_deconv4, out_conv3)
        depth4 = self.predict_depth4(out_add4) 

        out_deconv3 = self.deconv3(out_add4)
        out_add3 = self.skip_add.add(out_deconv3, out_conv2)
        depth3 = self.pred_add.add(depth4, self.predict_depth3(out_add3)) # depth4 + self.predict_depth3(out_add3)

        out_deconv2 = self.deconv2(out_add3)
        out_add2 = self.skip_add.add(out_deconv2, out_conv1)
        depth2 = self.pred_add.add(depth3, self.predict_depth2(out_add2)) # depth3 + self.predict_depth2(out_add2)

        out_deconv1 = self.deconv1(out_add2)
        out_add1 = self.skip_add.add(out_deconv1, out_bottom)
        depth1 = self.pred_add.add(depth2, self.predict_depth2(out_add2)) # depth2 + self.predict_depth1(out_add1)

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

    def multiply_parameters(self, factor):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.groups == 1:
                    m.weight = Parameter(m.weight * factor)
                    if m.bias is not None:
                        m.bias = Parameter(m.bias * factor)

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

