import torch
from utils.common.fourier_transfroms import fft2c_new, ifft2c_new


class AIRSLayer(torch.nn.Module):
    """
    AIRSLayer
    https://slideslive.com/38942422/featurelevel-multidomain-learning-with-a-standardization-for-multichannel-mri-data?ref=speaker-58967
    """

    def __init__(self, in_chans: int = 2, out_chans: int = 2, num_pools: int = 4, chans: int = 32):
        """

        :param in_chans: Number of channels in the input.
        :param out_chans:  Number of channels in the output.
        :param num_pools: Number of layers of up and down sampling layers.
        :param chans: Number of channels of the output of the first convolutional layer.
        """

        super().__init__()
        self.down_sample_layers = torch.nn.ModuleList()
        ch = chans
        for i in range(num_pools):
            if i == 0:
                self.down_sample_layers.append(FLMDM(in_chans, ch))
                self.down_sample_layers.append(FLMDM(ch, ch))
            else:
                self.down_sample_layers.append(FLMDM(ch, ch * 2))
                ch *= 2
                self.down_sample_layers.append(FLMDM(ch, ch))
            self.down_sample_layers.append(torch.nn.MaxPool2d(2))

        self.same_sample_layers = torch.nn.Sequential(
            FLMDM(ch, ch),
            FLMDM(ch, ch)
        )

        self.up_sample_layers = torch.nn.ModuleList()

        ch *= 2
        for i in range(num_pools):
            self.up_sample_layers.append(torch.nn.UpsamplingBilinear2d(scale_factor=2))
            self.up_sample_layers.append(FLMDM(ch, ch // 4, sampling_method='up'))
            ch = ch // 4
            self.up_sample_layers.append(FLMDM(ch, ch, sampling_method='up'))
            ch *= 2

        self.end_layer = FLMDM(ch // 2, out_chans, kernel_size=1)

    def forward(self, x):
        stack = []
        # input = torch.view_as_complex(input)
        # input = self.complex_to_chan_dim(input)
        # output = input

        for i, layer in enumerate(self.down_sample_layers):
            # maxpooling layer
            if i % 3 == 2:
                x = self.complex_to_chan_dim(x)
                x = layer(x)
                x = self.chan_complex_to_last_dim(x)
            else:
                x = layer(x)
                if i % 3 == 1:
                    stack.append(x)

        x = self.same_sample_layers(x)

        for i, layer in enumerate(self.up_sample_layers):
            if i % 3 == 0:
                x = self.complex_to_chan_dim(x)
                x = layer(x)
                x = self.chan_complex_to_last_dim(x)
                x = torch.cat([x, stack.pop()], dim=1)
            else:
                x = layer(x)

        x = self.end_layer(x)
        return x

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, two * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()  # contiguous 메모리 배치를 바꾸어줌.


class FLMDM(torch.nn.Module):
    """
    feature-level multi-domain module of AIRS model
    """

    def __init__(self, in_chans: int = 1, out_chans: int = 1, sampling_method: str = 'same', kernel_size: int = 3):
        """

        :param in_chans: Number of channels in the input.
        :param out_chans: Number of channels in the output.
        :param sampling_method: 'down', 'up', 'same'
        """
        super().__init__()
        assert sampling_method == 'down' or sampling_method == 'up' or sampling_method == 'same'

        self.sampling_method = sampling_method

        # padding for "same" mode
        padding = (kernel_size - 1) // 2


        # 여기 기존은 transpose임!!!
        if sampling_method == 'up':
            self.conv_image = torch.nn.ConvTranspose2d(in_chans, out_chans, kernel_size, padding=padding, bias=False)
            self.conv_freq = torch.nn.ConvTranspose2d(in_chans, out_chans, kernel_size, padding=padding, bias=False)
        else:
            self.conv_image = torch.nn.Conv2d(in_chans, out_chans, kernel_size, padding=padding, bias=False)
            self.conv_freq = torch.nn.Conv2d(in_chans, out_chans, kernel_size, padding=padding, bias=False)

        self.leakyRelu = torch.nn.LeakyReLU(0.1, True)
        self.regularization_param = torch.nn.Parameter(torch.ones(1))
        # self.instanceNorm2d1 = torch.nn.InstanceNorm2d(2)
        # self.instanceNorm2d2 = torch.nn.InstanceNorm2d(2)

    def forward(self, input):
        # image domain
        input1 = self.complex_to_chan_dim(input)
        input1 = self.conv_image(input1)
        input1 = self.chan_complex_to_last_dim(input1)

        # frequency domain
        input = fft2c_new(input)
        input = self.complex_to_chan_dim(input)
        input = self.conv_freq(input)
        input = self.chan_complex_to_last_dim(input)
        input = ifft2c_new(input)

        # input = (input1 + self.regularization_param * input) / (1 + self.regularization_param)
        input = input + input1

        return self.leakyRelu(input)


    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()  # contiguous 메모리 배치를 바꾸어줌.

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, two * c, h, w)
