from collections import OrderedDict
import torch
from airs_module import AIRSLayer
from SME import AdaptiveSensitivityModel
from utils.common.utils import fftc_torch, ifftc_torch, complex_conj, complex_mul
from utils.common.fourier_transfroms import ifft2c_new, fft2c_new
import numpy as np
from plot_result import plot_image


class MyModel_V7(torch.nn.Module):
    def __init__(self, airs_inchans= 4,
                 airs_outchans: int = 2,
                 air_inchans_next: int = 4,
                 airs_pools: int = 4,
                 airs_chans: int = 128,
                 sme_chans: int = 8,
                 sme_pools: int = 4,
                 num_airs_layers: int = 1,
                 airs_layers_stateDict: OrderedDict = None,
                 sme_stateDict: OrderedDict = None,
                 disable_train_index: int = -1,
                 train: bool = True,
                 retrain: bool = False
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

        self.sme = AdaptiveSensitivityModel(chans=sme_chans, num_pools=sme_pools)

        if train:
            n = disable_train_index + 2 if self.retrain else disable_train_index + 1

            if airs_layers_stateDict is not None:
                for i in range(n):
                    if i == 0:
                        self.airs_layers.append(AIRSLayer(airs_inchans, airs_outchans, airs_pools, airs_chans))
                    else:
                        self.airs_layers.append(
                            AIRSLayer(air_inchans_next, airs_outchans, airs_pools, airs_chans))
                self.airs_layers.load_state_dict(airs_layers_stateDict)

            if airs_layers_stateDict is not None:
                if self.retrain:
                    n = 0
                else:
                    n = self.num_airs_layers - len(self.airs_layers)
            else:
                n = self.num_airs_layers

            for i in range(n):
                self.airs_layers.append(AIRSLayer(air_inchans_next, airs_outchans, airs_pools, airs_chans))
        else:
            for i in range(num_airs_layers):
                if i == 0:
                    self.airs_layers.append(AIRSLayer(airs_inchans, airs_outchans, airs_pools, airs_chans))
                else:
                    self.airs_layers.append(AIRSLayer(air_inchans_next, airs_outchans, airs_pools, airs_chans))
            self.airs_layers.load_state_dict(airs_layers_stateDict)
            self.sme.load_state_dict(sme_stateDict)

    def forward(self, input: torch.Tensor, grappa: torch.Tensor, acs: torch.Tensor) -> torch.Tensor:
        # input (b,c, h, w) kspace
        sens_map = self.sme(input, acs)
        input = self.sens_reduce(input, sens_map)
        grappa = grappa.unsqueeze(1)
        grappa = torch.stack([grappa, torch.zeros_like(grappa)], dim=-1)
        input = torch.cat([input, grappa], dim=1)
        for i, layer in enumerate(self.airs_layers):
            if i <= self.disable_train_index:
                with torch.no_grad():
                    input = self.forward_cascade(layer, input, i)
            else:
                input = self.forward_cascade(layer, input, i)

        input = torch.abs(torch.view_as_complex(input))
        # input = self.sens_expand(input, sens_map)
        # input = self.rss_combine_torch(input)
        input = torch.clamp(input.squeeze(1), 0, 0.0014163791202008724)
        return input

    def forward_cascade(self, layer, output, i):
        if i > 0:
            add_later = output[:, :1]
        else:
            add_later = output[:, :1]
        output = layer(output)
        output = add_later + output
        return output

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor):
        if x.shape[-1] != 2:
            x = ifftc_torch(x)
            x = torch.view_as_real(x)
        else:
            x = ifft2c_new(x)
        return complex_mul(x, complex_conj(sens_maps)).sum(dim=1, keepdim=True)

    def sens_expand(self, x: torch.Tensor, sens_map):
        return complex_mul(x, complex_mul(x, sens_map))

    def rss_combine_torch(self, data, axis = 1):
        if data.shape[-1] == 2:
            data = torch.view_as_complex(data)
        return torch.sqrt(torch.square(torch.abs(data)).sum(axis))
