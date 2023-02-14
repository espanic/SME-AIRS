from collections import defaultdict
from typing import Optional, Tuple

import torch
import torch.nn as nn
from unet import NormUnet
from utils.common.utils import ifftc_torch
from plot_result import plot_image


class AdaptiveSensitivityModel(nn.Module):
    """
    Model for learning sensitivity map estimation from k-space data.
    This model applies an IFFT to multichannel k-space data and then a U-Net
    to the coil images to estimate coil sensitivities. It can be used with the
    end-to-end variational network.
    """

    def __init__(
            self,
            chans: int,
            num_pools: int,
            in_chans: int = 2,
            out_chans: int = 2,
            drop_prob: float = 0.0,
            low_freq_lines: int = None,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.

        """
        super().__init__()
        self.chans = chans
        self.num_pools = num_pools
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob
        self.low_freq_lines = low_freq_lines
        self.norm_unet = NormUnet(
            chans,
            num_pools,
            in_chans=in_chans,
            out_chans=out_chans,
            drop_prob=drop_prob,
        )

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w, comp = x.shape

        return x.view(b * c, 1, h, w, comp), b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, _, h, w, comp = x.shape
        c = bc // batch_size

        return x.view(batch_size, c, h, w, comp)

    # def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
    #     return x / fastmri.rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

    def rss_combine_torch(self, data, axis):
        return torch.sqrt(torch.square(torch.abs(data)).sum(axis))

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # (1, c, h, w, 2)

        # applying acs mask to undersampled kspace

        if self.low_freq_lines is not None:
            mask = self.enforce_low_freq_lines(mask)
        x = torch.mul(masked_kspace, mask)

        # ifft to image domain
        x = ifftc_torch(x)
        x = torch.view_as_real(x)
        x, b = self.chans_to_batch_dim(x)
        x = self.norm_unet(x)
        x = self.batch_chans_to_chan_dim(x, b)
        x = torch.view_as_complex(x)
        return x / self.rss_combine_torch(x, 1).unsqueeze(1)

    def enforce_low_freq_lines(self, mask):
        center = mask.shape[-1] // 2
        left = torch.argmin(mask[:center].flip(), dim=0)
        right = torch.argmin(mask[center:], dim=0)
        if torch.min(left, right) * 2 < self.low_freq_lines:
            raise RuntimeError(
                    "acs region too small"
                )
        acs = torch.zeros_like(mask, dtype=mask.dtype)
        acs[center - self.low_freq_lines // 2 + 1:center + self.low_freq_lines + 1] = 1.
        return acs



