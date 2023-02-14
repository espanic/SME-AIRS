from collections import OrderedDict
from torch import Tensor
import torch
from SMEAIRS.airs_module import AIRSLayer
from SMEAIRS.plot_result import plot_image


# weak layers for each of the coil image

class MyModel_V9(torch.nn.Module):
    def __init__(self,
                 airs_inchans=4,
                 airs_outchans: int = 2,
                 airs_pools: int = 4,
                 airs_chans: int = 128,
                 weak_inchans=2,
                 weak_outchans=2,
                 weak_pools=2,
                 weak_chans=16,
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

        self.airs_layers = torch.nn.ModuleList()
        self.weak_layer = AIRSLayer(weak_inchans, weak_outchans, weak_pools, weak_chans)

        for i in range(num_airs_layers):
            if i == 0:
                self.airs_layers.append(AIRSLayer(airs_inchans, airs_outchans, airs_pools, airs_chans))
            else:
                self.airs_layers.append(AIRSLayer(airs_outchans, airs_outchans, airs_pools, airs_chans))

    def forward(self, x: torch.Tensor, grappa: torch.Tensor, train_only_weak: bool = False) -> torch.Tensor:
        # x (b,c, h, w) undersampled coil image
        undersampled_img = self.rss_combine_torch(x)
        # undersampled_img_max = undersampled_img.max()
        max_val = self.max_normalize(x)
        x = x / max_val
        x = torch.view_as_real(x)
        x, b = self.chans_to_batch_dim(x)
        # for i in range(x.shape[0] // b):
        #     x[i] = self.weak_layer(x[i:i+1])
        if train_only_weak:
            x = self.forward_weak_layer(x)
        else:
            with torch.no_grad():
                x = self.forward_weak_layer(x)
        x = self.batch_chans_to_chan_dim(x, b)
        x = torch.view_as_complex(x)
        x = x * max_val
        x = self.rss_combine_torch(x)

        # short connect
        # x = x + undersampled_img

        if train_only_weak:
            x = torch.clamp(x.squeeze(1), 0, 0.0014163791202008724)
            return x

        grappa = grappa.unsqueeze(1)
        x = torch.cat([undersampled_img, grappa, x], dim=1)
        x = torch.complex(x, torch.zeros_like(x, dtype=torch.float32))
        x = torch.view_as_real(x)
        for i, layer in enumerate(self.airs_layers):
            if i <= self.disable_train_index:
                with torch.no_grad():
                    x = self.forward_cascade(layer, x, i)
            else:
                x = self.forward_cascade(layer, x, i)

        x = torch.abs(torch.view_as_complex(x))
        x = torch.clamp(x.squeeze(1), 0, 0.0014163791202008724)
        return x

    def max_normalize(self, x):
        b, c, h, w = x.shape
        x_abs = x.abs()
        x_abs = x_abs.view(b, c, h * w)
        max_val = torch.max(x_abs, dim=2).values.view(b, c, 1, 1)
        return max_val

    def forward_weak_layer(self, x):
        # x = self.complex_to_chan_dim(x)
        # x, mean, std = self.norm(x)
        # x = self.chan_complex_to_last_dim(x)
        x = self.weak_layer(x)
        # x = self.complex_to_chan_dim(x)
        # x = self.unnorm(x, mean, std)
        # x = self.chan_complex_to_last_dim(x)
        return x

    def forward_cascade(self, layer, x, i):
        # 여기 0으로 하면 input을  1로 하면 grappa를 함.
        add_later = x[:, :1]
        x = layer(x)
        x = add_later + x
        return x

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True, prior_trained_cascade_level: int = -1, load_weak_only: bool = False):
        self.weak_layer.load_state_dict(state_dict['weak_layer'])
        if load_weak_only:
            return
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

    def chans_to_batch_dim(self, x: torch.Tensor):
        b, c, h, w, two = x.shape
        assert two == 2
        return x.view(b * c, 1, h, w, 2), b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, _, h, w, two = x.shape
        assert two == 2
        c = bc // batch_size

        return x.view(batch_size, c, h, w, two)

    def rss_combine_torch(self, data, axis=1):
        return torch.sqrt(torch.square(torch.abs(data)).sum(axis, keepdim=True))

    def norm(self, x: torch.Tensor):
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)

        mean = x.mean(dim=2).view(b, 2, 1, 1)
        std = x.std(dim=2).view(b, 2, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    def unnorm(
            self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ):
        return x * std + mean

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()  # contiguous 메모리 배치를 바꾸어줌.

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w).contiguous()
