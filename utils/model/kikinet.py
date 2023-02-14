import torch
from torch import nn


class Conv2dRelu(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, x):
        return self.layers(x)


class Inet(nn.Module):

    def __init__(self, in_chans, out_chans, num_layers):
        super().__init__()
        self.I_Fr = nn.Conv2d(in_chans, 1, kernel_size=3, padding=1)
        self.I_Fi = nn.Conv2d(in_chans, 1, kernel_size=3, padding=1)
        chan_num = 2
        self.conv_layers = nn.Sequential()
        for i in range(num_layers):
            # self.conv_layers.append(Conv2dRelu(chan_num, 64))
            self.conv_layers.add_module(f'conv_{i+1}', Conv2dRelu(chan_num, 64))
            chan_num = 64
        self.I_Rr = nn.Conv2d(chan_num, 1, kernel_size=3, padding=1)
        self.I_Ri = nn.Conv2d(chan_num, out_chans, kernel_size=3, padding=1)

    def forward(self, input):
        input, mean, std = self.norm(input)
        input = input.unsqueeze(1)
        fr = self.I_Fr(torch.real(input))
        fi = self.I_Fi(torch.imag(input))
        f = torch.cat((fr, fi), 1)
        for layer in self.conv_layers:
            f = layer(f)
        rr = self.I_Rr(f)
        ri = self.I_Ri(f)
        output = torch.complex(rr, ri)
        output = torch.add(input, output)
        output = output.squeeze(1)
        output = self.unnorm(output, mean, std)
        output = torch.abs(output)
        return output

    def norm(self, x):
        b, h, w = x.shape  # batch, height, width
        x = x.view(b, h * w)  # resize to (b, h*w)
        mean = x.mean(dim=1).view(b, 1, 1)
        std = x.std(dim=1).view(b, 1, 1)
        x = x.view(b, h, w)
        return (x - mean) / std, mean, std

    def unnorm(self, x, mean, std):
        return x * std + mean


class Knet(nn.Module):

    def __init__(self, in_chans, out_chans, num_layers):
        super().__init__()
        self.K_Fr = nn.Conv2d(in_chans, 1, kernel_size=3, padding=1)
        self.K_Fi = nn.Conv2d(in_chans, 1, kernel_size=3, padding=1)
        chan_num = 2
        self.conv_layers = nn.Sequential()
        for i in range(num_layers):
            # self.conv_layers.append(Conv2dRelu(chan_num, 64))
            self.conv_layers.add_module(f'conv_{i+1}', Conv2dRelu(chan_num, 64))
            chan_num = 64
        self.K_Kr = nn.Conv2d(chan_num, 1, kernel_size=3, padding=1)
        self.K_Ki = nn.Conv2d(chan_num, out_chans, kernel_size=3, padding=1)

    def forward(self, input):
        input, mean, std = self.norm(input)
        input = input.unsqueeze(1)
        real = torch.real(input)
        imag = torch.imag(input)
        fr = self.K_Fr(real)
        fi = self.K_Fi(imag)
        f = torch.cat((fr, fi), 1)
        for layer in self.conv_layers:
            f = layer(f)
        rr = self.K_Kr(f)
        ri = self.K_Ki(f)
        output = torch.complex(rr, ri)
        output = output.squeeze(1)
        return self.unnorm(output, mean, std)

    def norm(self, x):
        b, h, w = x.shape  # batch, height, width
        x = x.view(b, h * w)  # resize to (b, h*w)
        mean = x.mean(dim=1).view(b, 1, 1)
        std = x.std(dim=1).view(b, 1, 1)
        x = x.view(b, h, w)
        return (x - mean) / std, mean, std

    def unnorm(self, x, mean, std):
        return x * std + mean