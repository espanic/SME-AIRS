from SME import AdaptiveSensitivityModel
from airs_module import AIRSLayer
from utils.common.utils import ifftc_torch, fftc_torch, complex_conj, complex_mul
from utils.common.fourier_transfroms import ifft2c_new, fft2c_new
from collections import OrderedDict
import torch
from plot_result import plot_image


# epoch 0 0.9698415670230116

class Coil_Model(torch.nn.Module):
    def __init__(self, airs_inchans: int = 44,
                 airs_outchans: int = 2,
                 airs_pools: int = 4,
                 airs_chans: int = 128,
                 sme_chans: int = 8,
                 sme_pools: int = 4,
                 target_shape=(384, 384),
                 max_coil=20
                 ):
        super().__init__()
        self.airs_inchans = airs_inchans
        self.airs_outchans = airs_outchans
        self.airs_pools = airs_pools
        self.airs_chans = airs_chans
        self.sme_chans = sme_chans
        self.sme_pools = sme_pools

        self.target_shape = target_shape
        self.enable_sme_train = False
        self.max_coil = max_coil

        self.airs_module = AIRSLayer(airs_inchans, airs_outchans, airs_pools, airs_chans)
        self.sme = AdaptiveSensitivityModel(self.sme_chans, self.sme_pools)


    def forward(self, kspace,grappa, mask) -> torch.Tensor:
        coil_pad = self.max_coil - kspace.shape[1]
        assert coil_pad >= 0
        coil_imgs = ifftc_torch(kspace)
        coil_imgs, _ = self.crop(coil_imgs)
        under_sampled_img = self.rss_combine_torch(coil_imgs)
        under_sampled_img = torch.complex(under_sampled_img, torch.zeros_like(under_sampled_img))
        under_sampled_img = torch.view_as_real(under_sampled_img)

        grappa = grappa.unsqueeze(1)
        grappa = torch.complex(grappa, torch.zeros_like(grappa))
        grappa = torch.view_as_real(grappa)

        sens_maps = self.sme(kspace, mask)
        sens_maps, _ = self.crop(sens_maps)
        sens_maps = torch.view_as_real(sens_maps)

        residual = self.sens_expand(under_sampled_img, sens_maps)
        residual = torch.view_as_real(coil_imgs) - residual

        x = torch.cat([under_sampled_img,  under_sampled_img - grappa, residual], dim=1)
        if coil_pad > 0:
            b, _, h, w, two = x.shape
            zero_pad = torch.zeros((b, coil_pad, h, w, two), dtype=torch.float32, device=0)
            x = torch.cat([x, zero_pad], dim=1)

        x = self.airs_module(x)
        x = x + under_sampled_img
        x = torch.view_as_complex(x)
        return torch.abs(x.squeeze(1))

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

    def rss_combine_torch(self, data, axis=1):
        if data.shape[-1] == 2:
            data = torch.view_as_complex(data)
        return torch.sqrt(torch.square(torch.abs(data)).sum(axis, keepdim=True))

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
