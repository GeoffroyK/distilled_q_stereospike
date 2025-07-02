import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from spikingjelly.clock_driven import functional, neuron, layer, surrogate

from .blocks import DownsamplingConv, ResBlock, BilinConvUpsampling


class AnalogNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_test_accuracy = float('inf')
        self.epoch = 0
        self.is_spiking = False

    def detach(self):
        pass

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


class StereoSpike_equivalentANN(AnalogNet):
    """
    Analog equivalent of StereoSpike. Main differences are:
      - activation function (ReLU, LeakyReLU, Sigmoid...)
      - BatchNorm
      - learnable biases in convolutional layers
      - bilinear upsampling

    Such equivalent ANNs have been shown in our paper to have worse accuracy than StereoSpike spiking baseline.
    
    """
    def __init__(self, input_chans=4, kernel_size=7, base_chans=32, activation_function=nn.LeakyReLU(0.1), separable_convs=True):
        super().__init__()

        C = [base_chans * (2**n) for n in range(5)]
        K = kernel_size
        P = (kernel_size - 1) // 2

        # bottom layer, preprocessing the input spike frame without downsampling
        self.bottom = nn.Sequential(
            # nn.Conv2d(in_channels=4, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.Conv2d(in_channels=input_chans, out_channels=input_chans, groups=input_chans, kernel_size=K, stride=1, padding=P, bias=False),
            nn.Conv2d(in_channels=input_chans, out_channels=C[0], kernel_size=1, stride=1, bias=False),
            activation_function,
            nn.BatchNorm2d(C[0]),
        )

        # encoder layers (downsampling)
        self.conv1 = nn.Sequential(
            DownsamplingConv(in_channels=C[0], out_channels=C[1], kernel_size=K, bias=False, separable=separable_convs),
            activation_function,
            nn.BatchNorm2d(C[1]),
        )
        self.conv2 = nn.Sequential(
            DownsamplingConv(in_channels=C[1], out_channels=C[2], kernel_size=K, bias=False, separable=separable_convs),
            activation_function,
            nn.BatchNorm2d(C[2]),
        )
        self.conv3 = nn.Sequential(
            DownsamplingConv(in_channels=C[2], out_channels=C[3], kernel_size=K, bias=False, separable=separable_convs),
            activation_function,
            nn.BatchNorm2d(C[3]),
        )
        self.conv4 = nn.Sequential(
            DownsamplingConv(in_channels=C[3], out_channels=C[4], kernel_size=K, bias=False, separable=separable_convs),
            activation_function,
            nn.BatchNorm2d(C[4]),
        )

        # residual layers
        self.bottleneck = nn.Sequential(
            ResBlock(C[4], kernel_size=K, activation_function=activation_function, separable=separable_convs),
            ResBlock(C[4], kernel_size=K, activation_function=activation_function, separable=separable_convs),
        )

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            BilinConvUpsampling(in_channels=C[4], out_channels=C[3], kernel_size=K, up_size=(33, 44), separable=separable_convs),
            activation_function,
            nn.BatchNorm2d(C[3]),
        )
        self.deconv3 = nn.Sequential(
            BilinConvUpsampling(in_channels=C[3], out_channels=C[2], kernel_size=K, up_size=(65, 87), separable=separable_convs),
            activation_function,
            nn.BatchNorm2d(C[2]),
        )
        self.deconv2 = nn.Sequential(
            BilinConvUpsampling(in_channels=C[2], out_channels=C[1], kernel_size=K, up_size=(130, 173), separable=separable_convs),
            activation_function,
            nn.BatchNorm2d(C[1]),
        )
        self.deconv1 = nn.Sequential(
            BilinConvUpsampling(in_channels=C[1], out_channels=C[0], kernel_size=K, up_size=(260, 346), separable=separable_convs),
            activation_function,
            nn.BatchNorm2d(C[0]),
        )

        # these layers output depth maps at different scales, where depth is represented by the potential of IF neurons
        # that do not fire ("I-neurons"), i.e., with an infinite threshold.
        self.predict_depth4 = nn.Sequential(
            BilinConvUpsampling(in_channels=C[3], out_channels=1, kernel_size=K, up_size=(260, 346), bias=True, separable=separable_convs),
        )
        self.predict_depth3 = nn.Sequential(
            BilinConvUpsampling(in_channels=C[2], out_channels=1, kernel_size=K, up_size=(260, 346), bias=True, separable=separable_convs),
        )
        self.predict_depth2 = nn.Sequential(
            BilinConvUpsampling(in_channels=C[1], out_channels=1, kernel_size=K, up_size=(260, 346), bias=True, separable=separable_convs),
        )
        self.predict_depth1 = nn.Sequential(
            BilinConvUpsampling(in_channels=C[0], out_channels=1, kernel_size=K, up_size=(260, 346), bias=True, separable=separable_convs),
        )

        self.Ineurons = neuron.IFNode(v_threshold=float('inf'), v_reset=0., surrogate_function=surrogate.ATan())

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
