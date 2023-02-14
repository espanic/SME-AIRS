from SME import AdaptiveSensitivityModel
from airs_module import AIRSLayer
from utils.common.utils import ifftc_torch, fftc_torch, complex_conj, complex_mul
from utils.common.fourier_transfroms import ifft2c_new, fft2c_new
from collections import OrderedDict
import torch
from plot_result import plot_image


class FinalModel(torch.nn.Module):
    def __init__(self, airs_inchans: int = 4,
                 airs_outchans: int = 2,
                 airs_pools: int = 4,
                 airs_chans: int = 64,
                 sme_chans: int = 8,
                 sme_pools: int = 4,
                 num_airs_layers: int = 1,
                 target_shape=(384, 384),
                 do_dc: bool = True,
                 crop_and_put: bool = True):
        super().__init__()
        self.air_inchans = airs_inchans
        self.air_outchans = airs_outchans
        self.airs_pools = airs_pools
        self.airs_chans = airs_chans
        self.sme_chans = sme_chans
        self.sme_pools = sme_pools
        self.num_airs_layers = num_airs_layers
        self.target_shape = target_shape
        self.do_dc = do_dc
        self.crop_and_put = crop_and_put

        self.airs_layers = torch.nn.ModuleList()
        self.regularization_params = torch.nn.ParameterList()
        self.sme = AdaptiveSensitivityModel(self.sme_chans, self.sme_pools)

        for _ in range(num_airs_layers):
            self.airs_layers.append(AIRSLayer(airs_inchans, airs_outchans, airs_pools, airs_chans))
            self.regularization_params.append(torch.nn.Parameter(torch.tensor(0.01)))

    def forward(self, kspace, grappa, mask) -> torch.Tensor:
        # input (1, c, h, w) shape kspace tensor
        grappa = grappa.unsqueeze(1)
        grappa = torch.complex(grappa, torch.zeros_like(grappa))
        grappa = torch.view_as_real(grappa)

        sens_maps = self.sme(kspace, mask)
        sens_maps = torch.view_as_real(sens_maps)

        kspace = torch.view_as_real(kspace)
        ref_kspace = kspace.clone()
        for i, layer in enumerate(self.airs_layers):
            kspace = self.foward_cascade(layer, kspace, ref_kspace, grappa, mask, sens_maps, i)

        x = ifft2c_new(kspace)
        x = torch.view_as_complex(x)
        x = self.rss_combine_torch(x, 1)

        # combine the result

        x, _ = self.crop(x)
        return x

    def init_sens_map(self, state_dict):
        self.sme.load_state_dict(state_dict)

    def set_sme_requires_grad(self, requires_grad):
        self.sme.requires_grad_(requires_grad)

    def foward_cascade(self, layer, masked_kspace, ref_kspace, grappa, mask, sens_maps, i):
        reduced = self.sens_reduce(masked_kspace, sens_maps)
        if self.crop_and_put:
            x, cropped_loc = self.crop(reduced)
            max_val = self.get_max_val(x)
            x = torch.cat([x, grappa], dim=1)
            x = x / max_val
            x = layer(x)
            x = x * max_val
        result = torch.zeros_like(reduced)
        result[..., cropped_loc[0]: cropped_loc[1], cropped_loc[2]: cropped_loc[3], :] = x
        result = self.sens_expand(result, sens_maps)
        if self.do_dc:
            result = fft2c_new(result)
            return self.data_consistency(result, ref_kspace, mask, i)

    def data_consistency(self, kspace, ref_kspace, mask, i):
        kspace = torch.view_as_complex(kspace)
        ref_kspace = torch.view_as_complex(ref_kspace)
        kspace = kspace * (1. - mask) + (ref_kspace + self.regularization_params[i] *
                                         kspace * mask) / (1 + self.regularization_params[i])
        return torch.view_as_real(kspace)

    def get_max_val(self, x):
        x = torch.view_as_complex(x)
        return torch.abs(x).max()

    def sens_reduce(self, kspace: torch.Tensor, sens_maps: torch.Tensor):
        kspace = ifft2c_new(kspace)
        return complex_mul(kspace, complex_conj(sens_maps)).sum(dim=1, keepdim=True)

    def sens_expand(self, x: torch.Tensor, sens_map):
        return complex_mul(x, complex_mul(x, sens_map))

    def rss_combine_torch(self, data, axis):
        return torch.sqrt(torch.square(torch.abs(data)).sum(axis))

    def pad_black_region(self, x, target_shape):
        h, w = x.shape[..., -2, -1]

    def crop(self, input, hw=None):
        if hw is None:
            h, w = self.target_shape
        else:
            h, w = hw
        if input.shape[-1] == 2:
            center = input.shape[-3] // 2, input.shape[-2] // 2
            return input[..., center[0] - h // 2: center[0] + h // 2, center[1] - w // 2: center[1] + w // 2, :], \
                   (center[0] - h // 2, center[0] + h // 2, center[1] - w // 2, center[1] + w // 2)

        else:
            center = input.shape[-2] // 2, input.shape[-1] // 2
            return input[..., center[0] - h // 2: center[0] + h // 2, center[1] - w // 2: center[1] + w // 2], \
                   (center[0] - h // 2, center[0] + h // 2, center[1] - w // 2, center[1] + w // 2)


class FinalModel_v2(torch.nn.Module):
    def __init__(self, airs_inchans: int = 4,
                 airs_outchans: int = 2,
                 airs_pools: int = 4,
                 airs_chans: int = 128,
                 sme_chans: int = 8,
                 sme_pools: int = 4,
                 num_airs_layers: int = 1,
                 target_shape=(384, 384),
                 do_strict_dc: bool = False,
                 do_crop:bool = True,
                 ):
        super().__init__()
        self.air_inchans = airs_inchans
        self.air_outchans = airs_outchans
        self.airs_pools = airs_pools
        self.airs_chans = airs_chans
        self.sme_chans = sme_chans
        self.sme_pools = sme_pools
        self.num_airs_layers = num_airs_layers
        self.target_shape = target_shape
        self.do_strict_dc = do_strict_dc
        self.do_crop = do_crop
        self.airs_layers = torch.nn.ModuleList()
        self.regularization_params = torch.nn.ParameterList() if self.do_strict_dc else []
        self.sme = AdaptiveSensitivityModel(self.sme_chans, self.sme_pools)

        for i in range(num_airs_layers):
            if i == 0:
                self.airs_layers.append(AIRSLayer(airs_inchans, airs_outchans, airs_pools, airs_chans))
            else:
                self.airs_layers.append(AIRSLayer(airs_outchans, airs_outchans, airs_pools, airs_chans))
            if self.do_strict_dc:
                self.regularization_params.append(torch.nn.Parameter(torch.tensor(1.)))
            else:
                self.regularization_params.append(1)

    def forward(self, kspace, grappa, mask) -> torch.Tensor:
        # input (1, c, h, w) shape kspace tensor
        grappa = grappa.unsqueeze(1)
        grappa = torch.complex(grappa, torch.zeros_like(grappa))
        grappa = torch.view_as_real(grappa)
        sens_maps = self.sme(kspace, mask)
        sens_maps = torch.view_as_real(sens_maps)
        kspace = torch.view_as_real(kspace)
        ref_kspace = kspace.clone()
        for i, layer in enumerate(self.airs_layers):
            kspace = self.foward_cascade(layer, kspace, ref_kspace, grappa, mask, sens_maps, i)
        x = ifft2c_new(kspace)
        x = torch.view_as_complex(x)
        x = self.rss_combine_torch(x, 1)
        # combine the result
        x, _ = self.crop(x)
        return x

    def init_sens_map(self, state_dict):
        self.sme.load_state_dict(state_dict)

    def init_layer(self, state_dict):
        self.airs_layers.load_state_dict(state_dict)

    def set_sme_requires_grad(self, requires_grad):
        self.sme.requires_grad_(requires_grad)

    def foward_cascade(self, layer, masked_kspace, ref_kspace, grappa, mask, sens_maps, i):
        reduced = self.sens_reduce(masked_kspace, sens_maps)

        x, cropped_loc = self.crop(reduced)
        if i == 0:
            x = torch.cat([x, grappa], dim=1)
        max_val = self.get_max_val(x)
        x = x / max_val
        undersampled_img = x[:, :1]
        x = layer(x)
        x = x + undersampled_img
        x = x * max_val[:, :1, ...]
        result = torch.zeros_like(reduced)
        # result = reduced.clone()
        result[..., cropped_loc[0]: cropped_loc[1], cropped_loc[2]: cropped_loc[3], :] = x
        result = self.sens_expand(result, sens_maps)
        result = fft2c_new(result)
        return self.data_consistency(result, ref_kspace, mask, i)

    def data_consistency(self, kspace, ref_kspace, mask, i):
        kspace = torch.view_as_complex(kspace)
        ref_kspace = torch.view_as_complex(ref_kspace)
        sub = kspace * mask - ref_kspace
        kspace = kspace - sub * self.regularization_params[i]
        return torch.view_as_real(kspace)

    def get_max_val(self, x):
        x = torch.view_as_complex(x)
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w)
        max_val = torch.max(x.abs(), dim=2).values.view(b, c, 1, 1, 1)
        return max_val

    def sens_reduce(self, kspace: torch.Tensor, sens_maps: torch.Tensor):
        kspace = ifft2c_new(kspace)
        return complex_mul(kspace, complex_conj(sens_maps)).sum(dim=1, keepdim=True)

    def sens_expand(self, x: torch.Tensor, sens_map):
        return complex_mul(x, complex_mul(x, sens_map))

    def rss_combine_torch(self, data, axis):
        return torch.sqrt(torch.square(torch.abs(data)).sum(axis))

    def crop(self, input, hw=None):
        if hw is None:
            h, w = self.target_shape
        else:
            h, w = hw
        if input.shape[-1] == 2:
            center = input.shape[-3] // 2, input.shape[-2] // 2
            return input[..., center[0] - h // 2: center[0] + h // 2, center[1] - w // 2: center[1] + w // 2, :], \
                   (center[0] - h // 2, center[0] + h // 2, center[1] - w // 2, center[1] + w // 2)

        else:
            center = input.shape[-2] // 2, input.shape[-1] // 2
            return input[..., center[0] - h // 2: center[0] + h // 2, center[1] - w // 2: center[1] + w // 2], \
                   (center[0] - h // 2, center[0] + h // 2, center[1] - w // 2, center[1] + w // 2)
