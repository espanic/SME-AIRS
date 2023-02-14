from collections import OrderedDict
import torch
from SMEAIRS.airs_module import AIRSLayer
from utils.common.utils import fftc_torch, ifftc_torch
from utils.common.fourier_transfroms import ifft2c_new, fft2c_new
import numpy as np


# using 2 channel input : grappa and high_freq_filtered grappa image
class MyModel_V6(torch.nn.Module):
    def __init__(self, airs_inchans=2,
                 airs_outchans: int = 2,
                 air_inchans_with_filtered: int = 4,
                 airs_pools: int = 4,
                 airs_chans: int = 128,
                 num_airs_layers: int = 1,
                 airs_layers_stateDict: OrderedDict = None,
                 disable_train_index: int = -1,
                 filtering: bool = True,
                 r: float = 0.0001,
                 p: float = 1.3,
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
        self.high_freq_filter = self.get_filter(r, p) if filtering else None

        if train:
            n = disable_train_index + 2 if self.retrain else disable_train_index + 1

            if airs_layers_stateDict is not None:
                for i in range(n):
                    if i == 0:
                        self.airs_layers.append(AIRSLayer(airs_inchans, airs_outchans, airs_pools, airs_chans))
                    else:
                        self.airs_layers.append(
                            AIRSLayer(air_inchans_with_filtered, airs_outchans, airs_pools, airs_chans))
                self.airs_layers.load_state_dict(airs_layers_stateDict)

            if airs_layers_stateDict is not None:
                if self.retrain:
                    n = 0
                else:
                    n = self.num_airs_layers - len(self.airs_layers)
            else:
                n = self.num_airs_layers

            for i in range(n):
                self.airs_layers.append(AIRSLayer(air_inchans_with_filtered, airs_outchans, airs_pools, airs_chans))
        else:
            for i in range(num_airs_layers):
                if i == 0:
                    self.airs_layers.append(AIRSLayer(airs_inchans, airs_outchans, airs_pools, airs_chans))
                else:
                    self.airs_layers.append(AIRSLayer(air_inchans_with_filtered, airs_outchans, airs_pools, airs_chans))
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
                    if i == 0 or self.high_freq_filter is None:
                        output = self.forward_cascade(layer, output, i)
                    else:
                        output = torch.view_as_complex(output)
                        output_filtered = fftc_torch(output) * self.high_freq_filter
                        output_filtered = ifftc_torch(output_filtered)
                        output = torch.cat([output, output_filtered], dim=1)
                        output = torch.view_as_real(output)
                        output = self.forward_cascade(layer, output, i)
            elif i == 0 or self.high_freq_filter is None:
                output = self.forward_cascade(layer, output, i)
            else:
                output = torch.view_as_complex(output)
                output_filtered = fftc_torch(output) * self.high_freq_filter
                output_filtered = ifftc_torch(output_filtered)
                output = torch.cat([output, output_filtered], dim=1)
                output = torch.view_as_real(output)
                output = self.forward_cascade(layer, output, i)

        # output = self.unnorm(output, mean, std)
        output = torch.abs(torch.view_as_complex(output))
        output = torch.clamp(output.squeeze(1), 0, 0.0014163791202008724)
        return output

    def forward_cascade(self, layer, output, i):

        # 여기 0으로 하면 input을  1로 하면 grappa를 함.
        add_later = output[:, 0:1]
        output = layer(output)
        output = add_later + output
        return output

    def get_filter(self, r, p):
        h = torch.arange(0, 384)
        w = torch.arange(0, 384)
        h, w = torch.meshgrid(h, w)

        h = h - 191.5
        w = w - 191.5

        filter = torch.square(h) + torch.square(w)
        filter = torch.pow(filter * r, p)
        filter = filter.cuda(non_blocking=True)
        return filter
