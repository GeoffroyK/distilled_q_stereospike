import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import spikingjelly
from spikingjelly.clock_driven import functional, surrogate, neuron, layer, rnn

from .blocks import SEWResBlock, NNConvUpsampling, MultiplyBy, SeparableSEWResBlock, SeparableNNConvUpsampling, SeparableNNConvUpsampling_intermediatespike
from .SNN_models_simpquant import NeuromorphicNet


class sQuantizeSymmetric(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, s, rounding_mode="floor", b=8):
        """
        Applies Symmetric Quantization to tensor x using scale s.
            - Signed symmetric quantization with positive and negative values.

        Args:
            x (torch.Tensor): The floating-point input tensor.
            s (torch.Tensor): The scale factor.
            rounding_mode (str): "floor" for rounding down, "ceil" for rounding up.
            b (int): Number of bits for representation (8 bits by default)

        Returns:
            torch.Tensor: Quantized tensor.
        """
        # Normalize by scale
        x_scaled = x / s

        # Apply rounding (differentiable version)
        if rounding_mode == "floor":
            x_rounded = torch.floor(x_scaled)
        elif rounding_mode == "ceil":
            x_rounded = torch.ceil(x_scaled)
        else:
            raise ValueError("rounding_mode must be 'floor' or 'ceil'")

        # Lower limit
        x_min = -2**(b-1)
        # Upper limit
        x_max = 2**(b-1) - 1 

        # Apply clamping
        x_clamped = torch.clamp(x_rounded, x_min, x_max)

        # Store for backward pass
        ctx.save_for_backward(x, s)
        ctx.rounding_mode = rounding_mode
        ctx.x_min = x_min
        ctx.x_max = x_max

        return x_clamped

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backpropagation for differentiable symmetric quantization.

        Args:
            grad_output (torch.Tensor): Gradient of loss with respect to output --> this allows the chain rule to be applied

        Returns:
            Tuple of gradients for (x, s).
        """
        x, s = ctx.saved_tensors
        rounding_mode = ctx.rounding_mode

        # Gradient w.r.t. scale s
        grad_s = (-x / (s**2)) * grad_output
        grad_s = grad_s.sum().view_as(s)  # Aggregate over batch

        # return a gradient for all inputs to the function (expected 3)
        return None, grad_s, None
        
        
class Qs_Conv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        """
        A Conv2d layer that applies a learnable scaling factor s_q to its fixed weights.
            - Signed Symmetric Quantization --> differentiable function
        All other arguments are passed to the standard Qs_Conv2d.
        """
        super(Qs_Conv2d, self).__init__(*args, **kwargs)
        # Freeze weight and bias so they are not updated during optimization.
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        
        # Define a trainable scaling parameter (s_e)
        # This variable can take any real number (+/-) and will be used as an exponent
        # to ensure that the real scaling parameter (s_q) is always positive
        self.s_e = nn.Parameter(torch.ones(1))  
        # TODO? change shape for finer granularity, i.e., one learnable scale per channel and/or position
        #self.s_e = nn.Parameter(torch.ones(self.weight.shape))  # one scale per weight
        #self.s_e = nn.Parameter(torch.ones(self.weight.shape[0]))  # one scale per channel  (Cout, K, K, Cin)
        
        self.s_q = None
        # This attribute will store the effective weight for debugging.
        self.effective_weight = None
    
    def forward(self, input):
        # Exponential transformation to scaling factor to ensure s_q > 0 (constraint)
        self.s_q = torch.exp(self.s_e)
        # Apply symmetric quantization (int8)
        # Save it as an attribute for debugging purposes.
        self.effective_weight = sQuantizeSymmetric.apply(self.weight, self.s_q, "floor")
        # Use the effective weight in the convolution.
        return F.conv2d(input, self.effective_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
                        
                        
class SIMPLIFIED_SeparableSEWResBlock_noskip_scaled(nn.Module):
    """
    Version of the SEW-Resblock using Andres' conv2d layers with sclearnable quantization scaling factors
    """

    def __init__(self, in_channels: int, connect_function='ADD', v_threshold=1., v_reset=0.,
                 surrogate_function=surrogate.Sigmoid(), use_plif=False, tau=2., multiply_factor=1., kernel_size=3):
        super(SIMPLIFIED_SeparableSEWResBlock_noskip_scaled, self).__init__()

        self.conv1 = nn.Sequential(
            Qs_Conv2d(in_channels, in_channels, groups=in_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False),
            Qs_Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False),
        )

        self.conv2 = nn.Sequential(
            Qs_Conv2d(in_channels, in_channels, groups=in_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False),
            Qs_Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False),
        )

        self.connect_function = connect_function

    def forward(self, x):

        identity = x

        # put conv1, sn1, conv2, sn2 in an nn.Seqential ?
        out = self.conv1(x)
        out = surrogate.ATan(spiking=True)(out - 1.)  # self.sn1(out)
        out = self.conv2(out)
        out = surrogate.ATan(spiking=True)(out - 1)  # self.sn2(out)

        return out
                        
                        
class SeparableNNConvUpsampling_scaled(nn.Module):
    """
    Version of NNConvUpsampling using Andres' conv2d layers with learnable quantization scaling factors
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, up_size: tuple, bias: bool = False):
        super(SeparableNNConvUpsampling_scaled, self).__init__()

        self.up = nn.Sequential(
            nn.UpsamplingNearest2d(size=(up_size[0] + (kernel_size - 1), up_size[1] + (kernel_size - 1))),
            Qs_Conv2d(in_channels=in_channels, out_channels=in_channels, groups=in_channels, kernel_size=kernel_size, stride=1, padding=0, bias=bias),
            Qs_Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=bias),
        )

    def forward(self, x):
        out = self.up(x)
        return out

                        
class SIMPLIFIED_fromZero_feedforward_multiscale_tempo_Matt_NoskipAll_sepConv_SpikeFlowNetLike_v2_scaled(NeuromorphicNet):
    """
    Uses Andres' conv2d layers with sclearnable quantization scaling factors

    """
    def __init__(self, input_chans=4, kernel_size=7, base_chans=32, use_plif=False, detach_reset=True, tau=10., v_threshold=1.0, v_reset=0.0, multiply_factor=1., surrogate_function=surrogate.Sigmoid(), learnable_biases=False):
        super().__init__(detach_reset=detach_reset)

        C = [base_chans * (2**n) for n in range(5)]
        K = kernel_size
        P = (kernel_size - 1) // 2
        self.multiply_factor = multiply_factor

        # bottom layer, preprocessing the input spike frame without downsampling
        self.bottom = nn.Sequential(
            Qs_Conv2d(in_channels=input_chans, out_channels=input_chans, groups=input_chans, kernel_size=K, stride=1, padding=P, bias=False),
            Qs_Conv2d(in_channels=input_chans, out_channels=C[0], kernel_size=1, stride=1, bias=False),
        )

        # encoder layers (downsampling)
        self.conv1 = nn.Sequential(
            Qs_Conv2d(in_channels=C[0], out_channels=C[0], groups=C[0], kernel_size=K, stride=2, padding=P, bias=False),
            Qs_Conv2d(in_channels=C[0], out_channels=C[1], kernel_size=1, stride=1, bias=False),
        )
        self.conv2 = nn.Sequential(
            Qs_Conv2d(in_channels=C[1], out_channels=C[1], groups=C[1], kernel_size=K, stride=2, padding=P, bias=False),
            Qs_Conv2d(in_channels=C[1], out_channels=C[2], kernel_size=1, stride=1, bias=False),
        )
        self.conv3 = nn.Sequential(
            Qs_Conv2d(in_channels=C[2], out_channels=C[2], groups=C[2], kernel_size=K, stride=2, padding=P, bias=False),
            Qs_Conv2d(in_channels=C[2], out_channels=C[3], kernel_size=1, stride=1, bias=False),
        )
        self.conv4 = nn.Sequential(
            Qs_Conv2d(in_channels=C[3], out_channels=C[3], groups=C[3], kernel_size=K, stride=2, padding=P, bias=False),
            Qs_Conv2d(in_channels=C[3], out_channels=C[4], kernel_size=1, stride=1, bias=False),
        )

        # residual layers
        self.bottleneck = nn.Sequential(
            SIMPLIFIED_SeparableSEWResBlock_noskip_scaled(C[4], kernel_size=K, v_threshold=v_threshold, v_reset=v_reset, multiply_factor=multiply_factor, use_plif=True, tau=tau, surrogate_function=surrogate_function),
            SIMPLIFIED_SeparableSEWResBlock_noskip_scaled(C[4], kernel_size=K, v_threshold=v_threshold, v_reset=v_reset, multiply_factor=multiply_factor, use_plif=True, tau=tau, surrogate_function=surrogate_function),
        )

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            SeparableNNConvUpsampling_scaled(in_channels=C[4], out_channels=C[3], kernel_size=K, up_size=(33, 44)),
        )
        self.deconv3 = nn.Sequential(
            SeparableNNConvUpsampling_scaled(in_channels=C[3], out_channels=C[2], kernel_size=K, up_size=(65, 87)),
        )
        self.deconv2 = nn.Sequential(
            SeparableNNConvUpsampling_scaled(in_channels=C[2], out_channels=C[1], kernel_size=K, up_size=(130, 173)),
        )
        self.deconv1 = nn.Sequential(
            SeparableNNConvUpsampling_scaled(in_channels=C[1], out_channels=C[0], kernel_size=K, up_size=(260, 346)),
        )

        # these layers output depth maps at different scales, where depth is represented by the potential of IF neurons
        # that do not fire ("I-neurons"), i.e., with an infinite threshold.
        self.predict_depth4 = SeparableNNConvUpsampling_scaled(in_channels=C[3], out_channels=1, kernel_size=K, up_size=(260, 346), bias=True)
        self.predict_depth3 = SeparableNNConvUpsampling_scaled(in_channels=C[2], out_channels=1, kernel_size=K, up_size=(260, 346), bias=True)
        self.predict_depth2 = SeparableNNConvUpsampling_scaled(in_channels=C[1], out_channels=1, kernel_size=K, up_size=(260, 346), bias=True)
        self.predict_depth1 = SeparableNNConvUpsampling_scaled(in_channels=C[0], out_channels=1, kernel_size=K, up_size=(260, 346), bias=True)
        
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
        out_bottom = surrogate.ATan(spiking=True)(self.bottom(frame) - 1.)

        # pass through encoder layers
        out_conv1 = surrogate.ATan(spiking=True)(self.conv1(out_bottom) - 1.)
        out_conv2 = surrogate.ATan(spiking=True)(self.conv2(out_conv1) - 1.)
        out_conv3 = surrogate.ATan(spiking=True)(self.conv3(out_conv2) - 1.)
        out_conv4 = surrogate.ATan(spiking=True)(self.conv4(out_conv3) - 1.)

        # pass through residual blocks
        out_rconv = self.bottleneck(out_conv4)

        # gradually upsample while concatenating and passing through skip connections
        out_deconv4 = surrogate.ATan(spiking=True)(self.deconv4(out_rconv) - 1.)
        out_add4 = out_deconv4 #+ out_conv3
        depth4 = self.predict_depth4(out_add4) # * self.multiply_factor
        #self.Ineurons(self.predict_depth4(out_add4) * self.multiply_factor)
        #depth4 = self.Ineurons.v

        out_deconv3 = surrogate.ATan(spiking=True)(self.deconv3(out_add4) - 1.)
        out_add3 = out_deconv3 #+ out_conv2
        depth3 = depth4 + self.predict_depth3(out_add3) # * self.multiply_factor
        #self.Ineurons(self.predict_depth3(out_add3) * self.multiply_factor)
        #depth3 = self.Ineurons.v

        out_deconv2 = surrogate.ATan(spiking=True)(self.deconv2(out_add3) - 1.)
        out_add2 = out_deconv2 #+ out_conv1
        depth2  = depth3 + self.predict_depth2(out_add2) # * self.multiply_factor
        #self.Ineurons(self.predict_depth2(out_add2) * self.multiply_factor)
        #depth2 = self.Ineurons.v

        out_deconv1 = surrogate.ATan(spiking=True)(self.deconv1(out_add2) - 1.)
        out_add1 = out_deconv1 #+ out_bottom
        depth1 = depth2 + self.predict_depth1(out_add1) # * self.multiply_factor
        #self.Ineurons(self.predict_depth1(out_add1) * self.multiply_factor)
        #depth1 = self.Ineurons.v

        # the membrane potentials of the output IF neuron carry the depth prediction
        return [depth1, depth2, depth3, depth4], [out_rconv, out_add4, out_add3, out_add2, out_add1]

