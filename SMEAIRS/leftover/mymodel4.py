from collections import OrderedDict
import torch
from airs_module import AIRSLayer
from utils.common.utils import fftc_torch, ifftc_torch, MedianPool2d
from utils.common.fourier_transfroms import ifft2c_new, fft2c_new
import numpy as np
from plot_result import plot_image


# using 2 channel input : raw input and grappa plus add some noise to it
class MyModel_V8(torch.nn.Module):
    def __init__(self, airs_inchans=2,
                 airs_outchans: int = 2,
                 airs_pools: int = 4,
                 airs_chans: int = 128,
                 num_airs_layers: int = 1,
                 airs_layers_stateDict: OrderedDict = None,
                 disable_train_index: int = -1,
                 adding_noise:bool = False,
                 retrain: bool = False,
                 train: bool = True
                 ):
        super().__init__()
        self.air_inchans = airs_inchans
        self.air_outchans = airs_outchans
        self.airs_pools = airs_pools
        self.airs_chans = airs_chans

        self.num_airs_layers = num_airs_layers
        self.disable_train_index = disable_train_index
        self.retrain = retrain
        self.airs_layers = torch.nn.ModuleList()
        self.adding_noise = adding_noise
        self.medianPooling = MedianPool2d(kernel_size=11, same=True)

        if train:
            n = disable_train_index + 2 if self.retrain else disable_train_index + 1

            if airs_layers_stateDict is not None:
                for i in range(n):
                    if i == 0:
                        self.airs_layers.append(AIRSLayer(airs_inchans, airs_outchans, airs_pools, airs_chans))
                    else:
                        self.airs_layers.append(
                            AIRSLayer(airs_outchans, airs_outchans, airs_pools, airs_chans))
                self.airs_layers.load_state_dict(airs_layers_stateDict)

            if airs_layers_stateDict is not None:
                if self.retrain:
                    n = 0
                else:
                    n = self.num_airs_layers - len(self.airs_layers)
            else:
                n = self.num_airs_layers

            for i in range(n):
                self.airs_layers.append(AIRSLayer(airs_outchans, airs_outchans, airs_pools, airs_chans))
        else:
            for i in range(num_airs_layers):
                if i == 0:
                    self.airs_layers.append(AIRSLayer(airs_inchans, airs_outchans, airs_pools, airs_chans))
                else:
                    self.airs_layers.append(AIRSLayer(airs_outchans, airs_outchans, airs_pools, airs_chans))
            self.airs_layers.load_state_dict(airs_layers_stateDict)

    def forward(self, output: torch.Tensor) -> torch.Tensor:
        # input (b, h, w) grappa image
        # input = input.unsqueeze(1)
        output = torch.complex(output, torch.zeros_like(output, dtype=torch.float32))
        output = torch.view_as_real(output)
        # output = input
        for i, layer in enumerate(self.airs_layers):
            if i <= self.disable_train_index:
                with torch.no_grad():
                    output = self.forward_cascade(layer, output, i)
            else:
                output = self.forward_cascade(layer, output, i)

        # output = self.unnorm(output, mean, std)
        output = torch.abs(torch.view_as_complex(output))
        if self.adding_noise:
            output = self.add_noise(output)

        output = output.squeeze(1)
        output = torch.clamp(output.squeeze(1), 0, 0.0014163791202008724)
        return output

    def forward_cascade(self, layer, output, i):
        if i > 0:
            add_later = output[:, :1]
        else:
            add_later = output[:, :1]
        output = layer(output)
        output = add_later + output
        return output

    def add_noise(self, output, sigma=0.02):
        h, w = output.shape[-2], output.shape[-1]
        max_value = output.max()
        output = output / max_value
        median = sigma * torch.sqrt(self.medianPooling(output))
        output += median * torch.randn(h, w, device=0)
        return output * max_value
