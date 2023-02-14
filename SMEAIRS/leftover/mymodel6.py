from collections import OrderedDict
from torch import Tensor
from typing import Tuple, List
import math
import torch
from SMEAIRS.airs_module import AIRSLayer
from torch.nn import functional as F
from utils.common.utils import fftc_torch, ifftc_torch
from utils.common.fourier_transfroms import fft2c_new, ifft2c_new
from SMEAIRS.plot_result import plot_image


# weak layers for each of the coil image

class MyModel_V10(torch.nn.Module):
    def __init__(self,
                 airs_inchans=2,
                 airs_outchans: int = 2,
                 airs_pools: int = 4,
                 airs_chans: int = 64,
                 num_airs_layers: int = 1,
                 disable_train_index: int = -1,
                 retrain: bool = False
                 ):
        super().__init__()
        self.airs_inchans = airs_inchans
        self.airs_outchans = airs_outchans
        self.airs_pools = airs_pools
        self.airs_chans = airs_chans
        self.num_airs_layers = num_airs_layers
        self.disable_train_index = disable_train_index
        self.retrain = retrain
        self.params = torch.nn.ParameterList()
        self.airs_layers = torch.nn.ModuleList()
        for i in range(num_airs_layers):
            if i == 0:
                self.airs_layers.append(AIRSLayer(airs_inchans, airs_outchans, airs_pools, airs_chans))
            else:
                self.airs_layers.append(AIRSLayer(airs_outchans, airs_outchans, airs_pools, airs_chans))
            self.params.append(torch.nn.Parameter(torch.tensor(0.01, dtype=torch.float32)))

    def forward(self, masked_kspaces, mask):
        # x (b,c, h, w) undersampled coil image
        x = ifftc_torch(masked_kspaces)
        x = self.rss_combine_torch(x)
        max_val = x.max()
        x = x / max_val
        # for data consistency
        masked_kspaces = fftc_torch(x) * mask

        x, pad_sizes = self.pad(x)

        # grappa = grappa.unsqueeze(1)
        # x = torch.cat([undersampled_img, grappa, x], dim=1)
        x = torch.complex(x, torch.zeros_like(x, dtype=torch.float32))
        x = torch.view_as_real(x)

        for i, layer in enumerate(self.airs_layers):
            if i <= self.disable_train_index:
                with torch.no_grad():
                    x = layer(x)
                    x = self.unpad(x, pad_sizes)
                    x = fft2c_new(x)
                    x = x * (1 - mask) + (x * mask + self.params[i] * masked_kspaces) / (1 + self.params[i])
                    x = ifft2c_new(x)
            else:
                x = layer(x)
                x = torch.view_as_complex(x)
                x = self.unpad(x, pad_sizes[0], pad_sizes[1], pad_sizes[2], pad_sizes[3])
                x = fftc_torch(x)
                x = x * (1 - mask) + (x * mask + self.params[i] * masked_kspaces) / (1 + self.params[i])
                x = ifftc_torch(x)
                x = torch.view_as_real(x)

        x = x * max_val
        x = torch.abs(torch.view_as_complex(x))
        x = self.crop(x)
        x = torch.clamp(x.squeeze(1), 0, 0.0014163791202008724)
        return x

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True, prior_trained_cascade_level: int = -1):

        assert prior_trained_cascade_level >= 0
        self.airs_layers = torch.nn.ModuleList()
        for i in range(prior_trained_cascade_level + 1):
            if i == 0:
                self.airs_layers.append(
                    AIRSLayer(self.airs_inchans, self.airs_outchans, self.airs_pools, self.airs_chans))
            else:
                self.airs_layers.append(
                    AIRSLayer(self.airs_outchans, self.airs_outchans, self.airs_pools, self.airs_chans))
        self.airs_layers.load_state_dict(state_dict['airs_layers'])
        for i in range(len(self.airs_layers), self.num_airs_layers):
            self.airs_layers.append(AIRSLayer(self.airs_outchans, self.airs_outchans, self.airs_pools, self.airs_chans))

    def rss_combine_torch(self, data, axis=1):
        return torch.sqrt(torch.square(torch.abs(data)).sum(axis, keepdim=True))

    def crop(self, input, hw = (384, 384)):
        h, w = hw
        center = input.shape[-2] // 2, input.shape[-1] // 2
        return input[..., center[0] - h // 2: center[0] + h // 2, center[1] - w // 2: center[1] + w // 2]


    def pad(
            self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
            self,
            x: torch.Tensor,
            h_pad: List[int],
            w_pad: List[int],
            h_mult: int,
            w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0]: h_mult - h_pad[1], w_pad[0]: w_mult - w_pad[1]]
