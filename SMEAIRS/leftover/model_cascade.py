from collections import OrderedDict
import torch
from SMEAIRS.airs_module_v2 import AIRSLayer
from utils.common.utils import fftc_torch, ifftc_torch
from utils.common.fourier_transfroms import ifft2c_new, fft2c_new
from SMEAIRS.plot_result import plot_image
import numpy as np


# using 2 channel input : grappa and high_freq_filtered grappa image
class MyModel_Cascade(torch.nn.Module):
    def __init__(self, airs_inchans=4,
                 airs_outchans: int = 2,
                 airs_pools: int = 4,
                 airs_chans: int = 64,
                 num_airs_layers: int = 1,
                 ):
        super().__init__()
        self.air_inchans = airs_inchans
        self.air_outchans = airs_outchans
        self.airs_pools = airs_pools
        self.airs_chans = airs_chans

        self.num_airs_layers = num_airs_layers
        self.airs_layers = torch.nn.ModuleList()

        for i in range(num_airs_layers):
            if i == 0:
                self.airs_layers.append(AIRSLayer(airs_inchans, airs_outchans, airs_pools, airs_chans, first_layer_optimize=True))
            else:
                self.airs_layers.append(AIRSLayer(airs_outchans, airs_outchans, airs_pools, airs_chans))

    def forward(self, x: torch.Tensor, mask = None) -> torch.Tensor:
        # input (b, h, w) grappa image
        # input = input.unsqueeze(1)

        max_val = self.get_max_val(x)
        x = x / max_val

        x = torch.complex(x, torch.zeros_like(x, dtype=torch.float32))
        x = torch.view_as_real(x)
        undersampled_input = x[:, 0:1]

        for i, layer in enumerate(self.airs_layers):
            x = self.forward_cascade(layer, x, undersampled_input)

        # output = self.unnorm(output, mean, std)
        x = torch.abs(torch.view_as_complex(x))
        x = x * max_val[:, :1, ...]
        # x = torch.clamp(x.squeeze(1), 0, 0.0014163791202008724)
        if mask is None:
            return x.squeeze(1)
        return x.squeeze(1) * mask

    def forward_cascade(self, layer, output, undersampled_input):
        # 여기 0으로 하면 input을  1로 하면 grappa를 함.
        output = undersampled_input + layer(output)
        return output

    def get_max_val(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, h * w)
        max_val = torch.max(x, dim=2).values.view(b, c, 1, 1)
        return max_val
