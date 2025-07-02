import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor
from torch.nn.parameter import Parameter

from spikingjelly.clock_driven import functional, surrogate, neuron, layer


##############
#   COMMON   #
##############

class DownsamplingConv(nn.Module):
    """
    Downsampling (stride=2) convolution block.

    Regular or separable convolutions.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, bias: bool = False, separable=True):
        super(DownsamplingConv, self).__init__()

        if separable:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, groups=in_channels, kernel_size=kernel_size, stride=2, padding=(kernel_size - 1) // 2, bias=bias),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=bias),
            )
        else:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=(kernel_size - 1) // 2, bias=bias),
            )

    def forward(self, x):
        return self.down(x)


##############
# SNN BLOCKS #
##############

class SEWResBlock(nn.Module):
    """
    Spike-Element-Wise (SEW) residual block as it is described in the paper "Spike-based residual blocks".
    See https://arxiv.org/abs/2102.04159

    Vanilla or Separable convolutions

    """

    def __init__(self, in_channels: int, kernel_size=7, connect_function='ADD', v_threshold=1., v_reset=0.,
                 surrogate_function=surrogate.Sigmoid(), use_plif=False, tau=2.,  multiply_factor=1., separable=True):
        super(SEWResBlock, self).__init__()

        if separable:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, groups=in_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False),
                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False),
                MultiplyBy(multiply_factor),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, groups=in_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False),
                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False),
                MultiplyBy(multiply_factor),
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=(3 - 1) // 2, bias=False),
                MultiplyBy(multiply_factor),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=(3 - 1) // 2, bias=False),
                MultiplyBy(multiply_factor),
            )

        self.sn1 = neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=True) if use_plif else neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=True)
        self.sn2 = neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=True) if use_plif else neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=True)

        self.connect_function = connect_function

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.sn1(out)
        out = self.conv2(out)
        out = self.sn2(out)

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


class NNConvUpsampling(nn.Module):
    """
    Upsampling block made to address the production of checkerboard artifacts by transposed convolutions.
    Nearest neighbour (NN) upsampling combined to regular convolutions efficiently gets rid of these patterns.
    Linear interpolation (among others) can produce non-integer values (e.g. 0.5 between 0 and 1), which go against the
    philosophy of spiking neural networks, which receive integer amounts of input spikes. Having noticed that, NN
    upsampling is compatible with SNN's philosophy and could be implemented on dedicated hardware.
    See https://distill.pub/2016/deconv-checkerboard/ for more insights on this phenomenon.

    Vanilla or Separable convolutions
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, up_size: tuple, bias: bool = False,
                 separable=True):
        super(NNConvUpsampling, self).__init__()

        if separable:
            self.up = nn.Sequential(
                nn.UpsamplingNearest2d(size=(up_size[0] + (kernel_size - 1), up_size[1] + (kernel_size - 1))),
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, groups=in_channels, kernel_size=kernel_size, stride=1, padding=0, bias=bias),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=bias),
            )
        else:
            self.up = nn.Sequential(
                nn.UpsamplingNearest2d(size=(up_size[0] + (kernel_size - 1), up_size[1] + (kernel_size - 1))),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=0, bias=bias),
            )

    def forward(self, x):
        out = self.up(x)
        return out


##############
# ANN BLOCKS #
##############

class ResBlock(nn.Module):
    """
    Standard residual block for ANNs, with vanilla or separable convolutions

    """
    def __init__(self, in_channels: int, kernel_size: int = 3, bias: bool = False,
                 activation_function: nn.Module = nn.Tanh(), separable=True):
        super(ResBlock, self).__init__()

        if separable:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, groups=in_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False),
                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False),
                activation_function,
                nn.BatchNorm2d(in_channels),
            )

            self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, groups=in_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False),
                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False),
                activation_function,
                nn.BatchNorm2d(in_channels),
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=bias),
                activation_function,
                nn.BatchNorm2d(in_channels),
            )

            self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=bias),
                activation_function,
                nn.BatchNorm2d(in_channels),
            )

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        out += identity

        return out


class BilinConvUpsampling(nn.Module):
    """
    Same as NNConvUpsampling, but using Bilinear interpolation instead of nearest neighbors.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, up_size: tuple, bias: bool = False,
                 separable=True):
        super(BilinConvUpsampling, self).__init__()

        if separable:
            self.up = nn.Sequential(
                nn.UpsamplingBilinear2d(size=(up_size[0] + (kernel_size - 1), up_size[1] + (kernel_size - 1))),
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, groups=in_channels, kernel_size=kernel_size, stride=1, padding=0, bias=bias),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=bias),
            )
        else:
            self.up = nn.Sequential(
                nn.UpsamplingBilinear2d(size=(up_size[0] + (kernel_size - 1), up_size[1] + (kernel_size - 1))),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=0, bias=bias),
            )

    def forward(self, x):
        out = self.up(x)
        return out


##############
#   OTHERS   #
##############
class MultiplyBy(nn.Module):
    """
    Allows to multiply inputs by a certain factor; this module helped us overcome some vanishing gradient problems
    when inputs were too far away from spiking neuron's threshold (i.e. either too weak or too strong).

    """

    def __init__(self, scale_value: float = 5., learnable: bool = False) -> None:
        super(MultiplyBy, self).__init__()

        if learnable:
            self.scale_value = Parameter(Tensor([scale_value]))
        else:
            self.scale_value = scale_value

    def forward(self, input: Tensor) -> Tensor:
        return torch.mul(input, self.scale_value)
