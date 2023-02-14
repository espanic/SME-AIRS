from SME import AdaptiveSensitivityModel
from airs_module import AIRSLayer
from utils.common.utils import ifftc_torch, complex_conj, complex_mul
from utils.common.fourier_transfroms import ifft2c_new, fft2c_new
from collections import OrderedDict
import torch
from plot_result import plot_image


class MyModel(torch.nn.Module):
    def __init__(self, airs_inchans: int = 2, airs_outchans: int = 2, airs_pools: int = 4, airs_chans: int = 64,
                 sme_inchans: int = 1,
                 sme_pools: int = 1, num_airs_layers: int = 1):
        super().__init__()
        self.air_inchans = airs_inchans
        self.air_outchans = airs_outchans
        self.airs_pools = airs_pools
        self.airs_chans = airs_chans
        self.sme_inchans = sme_inchans
        self.num_airs_layers = num_airs_layers

        self.airs_layers = torch.nn.ModuleList()
        self.regularization_params = torch.nn.ParameterList()
        self.sme = AdaptiveSensitivityModel(sme_inchans, sme_pools)
        for _ in range(num_airs_layers):
            self.airs_layers.append(AIRSLayer(airs_inchans, airs_outchans, airs_pools, airs_chans))
            self.regularization_params.append(torch.nn.Parameter(torch.ones(1)))

    def forward(self, input: torch.Tensor, acs: torch.Tensor) -> torch.Tensor:
        # input (1, c, h, w) shape kspace tensor

        input, mean, std = self.norm(input)
        # norm_factor = 384 * 384
        # input = input * norm_factor
        sens_maps = self.sme(input, acs)

        output = input
        for i, layer in enumerate(self.airs_layers):
            output = self.sens_reduce(output, sens_maps=sens_maps)
            output = layer(output)
            output = self.sens_expand(output, sens_maps)
            output = fft2c_new(output)
            output = torch.view_as_complex(output)

            # interleaved data consistency
            output = (input + output * self.regularization_params[i]) / (1 + self.regularization_params[i])
        # c개의 coil k-space
        # output = output / norm_factor
        output = ifftc_torch(output)

        output = self.unnorm(output, mean, std)

        # combine the result
        output = self.rss_combine_torch(output, 1)

        return output

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor):
        x = ifftc_torch(x)
        x = torch.view_as_real(x)
        return complex_mul(x, complex_conj(sens_maps)).sum(dim=1, keepdim=True)

    def sens_expand(self, x: torch.Tensor, sens_map):
        return complex_mul(x, complex_mul(x, sens_map))

    def rss_combine_torch(self, data, axis):
        return torch.sqrt(torch.square(torch.abs(data)).sum(axis))

    def norm(self, x):
        b, c, h, w = x.shape  # batch, height, width
        x = x.view(b, c, h * w)  # resize to (b, h*w)
        mean = torch.abs(x).mean(dim=2).view(b, c, 1, 1)
        std = torch.abs(x).std(dim=2).view(b, c, 1, 1)
        x = x.view(b, c, h, w)
        return (x - mean) / std, mean, std

    def unnorm(self, x, mean, std):
        return x * std + mean


class MyModel_V2(torch.nn.Module):
    def __init__(self, airs_inchans: int = 2,
                 airs_outchans: int = 2,
                 airs_pools: int = 4,
                 airs_chans: int = 128,
                 sme_inchans: int = 1,
                 sme_pools: int = 4,
                 num_airs_layers: int = 1,
                 sme_stateDict: OrderedDict = None,
                 airs_layers_stateDict: OrderedDict = None,
                 regularization_params_stateDict: OrderedDict = None,
                 disable_train_index: int = -1,
                 retrain: bool = False,
                 train: bool = True
                 ):
        super().__init__()
        self.air_inchans = airs_inchans
        self.air_outchans = airs_outchans
        self.airs_pools = airs_pools
        self.airs_chans = airs_chans
        self.sme_inchans = sme_inchans
        self.num_airs_layers = num_airs_layers
        self.disable_train_index = disable_train_index
        self.retrain = retrain

        self.sme = AdaptiveSensitivityModel(sme_inchans, sme_pools)
        self.airs_layers = torch.nn.ModuleList()
        self.regularization_params = torch.nn.ParameterList()

        if sme_stateDict is not None:
            self.sme.load_state_dict(sme_stateDict)

        if train:
            n = disable_train_index + 2 if self.retrain else disable_train_index + 1

            if airs_layers_stateDict is not None:
                for _ in range(n):
                    self.airs_layers.append(AIRSLayer(airs_inchans, airs_outchans, airs_pools, airs_chans))
                self.airs_layers.load_state_dict(airs_layers_stateDict)

            if regularization_params_stateDict is not None:
                for _ in range(n):
                    self.regularization_params.append(torch.nn.Parameter(torch.ones(1)))
                self.regularization_params.load_state_dict(regularization_params_stateDict)

            if airs_layers_stateDict is not None:
                if self.retrain:
                    n = 0
                else:
                    n = self.num_airs_layers - len(self.airs_layers)
            else:
                n = self.num_airs_layers

            for _ in range(n):
                self.airs_layers.append(AIRSLayer(airs_inchans, airs_outchans, airs_pools, airs_chans))
                self.regularization_params.append(torch.nn.Parameter(torch.ones(1)))
        else:
            for _ in range(num_airs_layers):
                self.airs_layers.append(AIRSLayer(airs_inchans, airs_outchans, airs_pools, airs_chans))
                self.regularization_params.append(torch.nn.Parameter(torch.ones(1)))
            self.airs_layers.load_state_dict(airs_layers_stateDict)
            self.regularization_params.load_state_dict(regularization_params_stateDict)

    def forward(self, input: torch.Tensor, acs: torch.Tensor) -> torch.Tensor:
        # input (1, c, h, w) shape kspace tensor

        # norm_factor = 384 * 384
        # input = input * norm_factor
        input, mean, std = self.norm(input)
        sens_maps = self.sme(input, acs)

        output = input
        for i, layer in enumerate(self.airs_layers):
            if i <= self.disable_train_index:
                with torch.no_grad():
                    output = self.forward_step(layer, input, output, sens_maps, i, mean, std)
            else:
                # c개의 coil k-space
                output = self.forward_step(layer, input, output, sens_maps, i, mean, std)

        output = ifftc_torch(output)

        output = self.unnorm(output, mean, std)

        # combine the result
        output = self.rss_combine_torch(output, 1)
        # clipping
        output = torch.clamp(output, 0, 0.0014163791202008724)
        return output

    def forward_step(self, layer, input, output, sens_maps, i, mean_input, std_input):
        output, mean, std = self.norm(output)
        output = self.sens_reduce(output, sens_maps=sens_maps)
        output = layer(output)
        output = self.sens_expand(output, sens_maps)
        output = fft2c_new(output)
        output = torch.view_as_complex(output)
        # interleaved data consistency
        output = self.unnorm(output, mean, std)
        output = (input + output * self.regularization_params[i]) / (1 + self.regularization_params[i])
        # output = self.unnorm(output, mean_input, std_input)
        return output

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor):
        x = ifftc_torch(x)
        x = torch.view_as_real(x)
        return complex_mul(x, complex_conj(sens_maps)).sum(dim=1, keepdim=True)

    def sens_expand(self, x: torch.Tensor, sens_map):
        return complex_mul(x, complex_mul(x, sens_map))

    def rss_combine_torch(self, data, axis):
        return torch.sqrt(torch.square(torch.abs(data)).sum(axis))

    def norm(self, x):
        b, c, h, w = x.shape  # batch, height, width
        x = x.view(b, c, h * w)  # resize to (b, h*w)
        mean = torch.abs(x).mean(dim=2).view(b, c, 1, 1)
        std = torch.abs(x).std(dim=2).view(b, c, 1, 1)
        x = x.view(b, c, h, w)
        return (x - mean) / std, mean, std

    def unnorm(self, x, mean, std):
        return x * std + mean


# includes cropping at the end and apply data consistency(not interleaved)
# may take training more longer

class MyModel_V3(torch.nn.Module):
    def __init__(self, airs_inchans: int = 2,
                 airs_outchans: int = 2,
                 airs_pools: int = 4,
                 airs_chans: int = 128,
                 sme_inchans: int = 1,
                 sme_pools: int = 4,
                 num_airs_layers: int = 1,
                 sme_stateDict: OrderedDict = None,
                 airs_layers_stateDict: OrderedDict = None,
                 regularization_params_stateDict: OrderedDict = None,
                 disable_train_index: int = -1,
                 retrain: bool = False,
                 train: bool = True
                 ):
        super().__init__()
        self.air_inchans = airs_inchans
        self.air_outchans = airs_outchans
        self.airs_pools = airs_pools
        self.airs_chans = airs_chans
        self.sme_inchans = sme_inchans
        self.num_airs_layers = num_airs_layers
        self.disable_train_index = disable_train_index
        self.retrain = retrain

        self.sme = AdaptiveSensitivityModel(sme_inchans, sme_pools)
        self.airs_layers = torch.nn.ModuleList()
        self.regularization_params = torch.nn.ParameterList()

        if sme_stateDict is not None:
            self.sme.load_state_dict(sme_stateDict)

        if train:
            n = disable_train_index + 2 if self.retrain else disable_train_index + 1

            if airs_layers_stateDict is not None:
                for _ in range(n):
                    self.airs_layers.append(AIRSLayer(airs_inchans, airs_outchans, airs_pools, airs_chans))
                self.airs_layers.load_state_dict(airs_layers_stateDict)

            if regularization_params_stateDict is not None:
                for _ in range(n):
                    self.regularization_params.append(torch.nn.Parameter(torch.ones(1)))
                self.regularization_params.load_state_dict(regularization_params_stateDict)

            if airs_layers_stateDict is not None:
                if self.retrain:
                    n = 0
                else:
                    n = self.num_airs_layers - len(self.airs_layers)
            else:
                n = self.num_airs_layers

            for _ in range(n):
                self.airs_layers.append(AIRSLayer(airs_inchans, airs_outchans, airs_pools, airs_chans))
                self.regularization_params.append(torch.nn.Parameter(torch.ones(1)))
        else:
            for _ in range(num_airs_layers):
                self.airs_layers.append(AIRSLayer(airs_inchans, airs_outchans, airs_pools, airs_chans))
                self.regularization_params.append(torch.nn.Parameter(torch.ones(1)))
            self.airs_layers.load_state_dict(airs_layers_stateDict)
            self.regularization_params.load_state_dict(regularization_params_stateDict)

    def forward(self, input: torch.Tensor, grappa: torch.Tensor, acs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # input (1, c, h, w) shape kspace tensor

        sens_maps = self.sme(input, acs)
        output = input
        for i, layer in enumerate(self.airs_layers):
            if i <= self.disable_train_index:
                with torch.no_grad():
                    output = self.forward_step(layer, input, output, mask, sens_maps, i)
            else:
                # c개의 coil k-space
                output = self.forward_step(layer, input, output, mask, sens_maps, i)

        output = ifftc_torch(output)

        # output = self.unnorm(output, mean, std)

        output = self.crop(output)

        # combine the result
        output = self.rss_combine_torch(output, 1)
        # clipping
        output = torch.clamp(output, 0, 0.0014163791202008724)
        return output

    def forward_step(self, layer, input, output, mask, sens_maps, i, mean_input, std_input):
        if i > 0:
            input, mean_input, std_input = self.norm(input)
        output = self.sens_reduce(output, sens_maps=sens_maps)
        output = layer(output)
        output = self.sens_expand(output, sens_maps)
        output = fft2c_new(output)
        output = torch.view_as_complex(output)
        # data consistency
        output = self.data_consistency(input, output, mask, i)
        output = self.unnorm(output, mean_input, std_input)
        return output

    def data_consistency(self, input, output, mask, i):
        dc_k = output * mask
        soft_dc = (dc_k - input) * self.regularization_params[i]
        return dc_k - soft_dc

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor):
        x = ifftc_torch(x)
        x = torch.view_as_real(x)
        return complex_mul(x, complex_conj(sens_maps)).sum(dim=1, keepdim=True)

    def sens_expand(self, x: torch.Tensor, sens_map):
        return complex_mul(x, complex_mul(x, sens_map))

    def rss_combine_torch(self, data, axis):
        return torch.sqrt(torch.square(torch.abs(data)).sum(axis))

    def norm(self, x):
        b, c, h, w = x.shape  # batch, height, width
        x = x.view(b, c, h * w)  # resize to (b, h*w)
        mean = torch.abs(x).mean(dim=2).view(b, c, 1, 1)
        std = torch.abs(x).std(dim=2).view(b, c, 1, 1)
        x = x.view(b, c, h, w)
        return (x - mean) / std, mean, std

    def unnorm(self, x, mean, std):
        return x * std + mean

    def crop(self, x):
        ch, cw = x.shape // 2
        return x[:, ch - 192: ch + 192, cw + 192: cw + 192]


# abs로 normalize말고 각 real, imag별 normalize를 해보자!
class MyModel_V4(torch.nn.Module):
    def __init__(self, airs_inchans: int = 2,
                 airs_outchans: int = 2,
                 airs_pools: int = 4,
                 airs_chans: int = 128,
                 sme_inchans: int = 1,
                 sme_pools: int = 4,
                 num_airs_layers: int = 1,
                 sme_stateDict: OrderedDict = None,
                 airs_layers_stateDict: OrderedDict = None,
                 regularization_params_stateDict: OrderedDict = None,
                 disable_train_index: int = -1,
                 retrain: bool = False,
                 train: bool = True
                 ):
        super().__init__()
        self.air_inchans = airs_inchans
        self.air_outchans = airs_outchans
        self.airs_pools = airs_pools
        self.airs_chans = airs_chans
        self.sme_inchans = sme_inchans
        self.num_airs_layers = num_airs_layers
        self.disable_train_index = disable_train_index
        self.retrain = retrain

        self.sme = AdaptiveSensitivityModel(sme_inchans, sme_pools)
        self.airs_layers = torch.nn.ModuleList()
        self.regularization_params = torch.nn.ParameterList()

        if sme_stateDict is not None:
            self.sme.load_state_dict(sme_stateDict)

        if train:
            n = disable_train_index + 2 if self.retrain else disable_train_index + 1

            if airs_layers_stateDict is not None:
                for _ in range(n):
                    self.airs_layers.append(AIRSLayer(airs_inchans, airs_outchans, airs_pools, airs_chans))
                self.airs_layers.load_state_dict(airs_layers_stateDict)

            if regularization_params_stateDict is not None:
                for _ in range(n):
                    self.regularization_params.append(torch.nn.Parameter(torch.ones(1)))
                self.regularization_params.load_state_dict(regularization_params_stateDict)

            if airs_layers_stateDict is not None:
                if self.retrain:
                    n = 0
                else:
                    n = self.num_airs_layers - len(self.airs_layers)
            else:
                n = self.num_airs_layers

            for _ in range(n):
                self.airs_layers.append(AIRSLayer(airs_inchans, airs_outchans, airs_pools, airs_chans))
                self.regularization_params.append(torch.nn.Parameter(torch.ones(1)))
        else:
            for _ in range(num_airs_layers):
                self.airs_layers.append(AIRSLayer(airs_inchans, airs_outchans, airs_pools, airs_chans))
                self.regularization_params.append(torch.nn.Parameter(torch.ones(1)))
            self.airs_layers.load_state_dict(airs_layers_stateDict)
            self.regularization_params.load_state_dict(regularization_params_stateDict)

    def forward(self, input: torch.Tensor, acs: torch.Tensor, grappa: torch.Tensor) -> torch.Tensor:
        # input (1, c, h, w) shape kspace tensor

        # 자체적인 normalization이 있음.
        sens_maps = self.sme(input, acs)
        input = torch.view_as_real(input)

        input, mean, std = self.norm(input)
        output = input
        for i, layer in enumerate(self.airs_layers):
            if i <= self.disable_train_index:
                with torch.no_grad():
                    output = self.forward_cascade(layer, input, output, sens_maps, i, mean, std)
            else:
                # c개의 coil k-space
                output = self.forward_cascade(layer, input, output, sens_maps, i, mean, std)

        output = ifft2c_new(output)
        output = torch.view_as_complex(output)

        # combine the result
        output = self.rss_combine_torch(output, 1)
        # clipping
        output = torch.clamp(output, 0, 0.0014163791202008724)
        return output

    def forward_cascade(self, layer, input, output, sens_maps, i, mean_input, std_input):
        if i > 0:
            output, mean, std = self.norm(output)
        else:
            mean, std = mean_input, std_input
        output = self.sens_reduce(output, sens_maps=sens_maps)
        output = layer(output)
        output = self.sens_expand(output, sens_maps)
        output = fft2c_new(output)
        # interleaved data consistency
        output = (input + output * self.regularization_params[i]) / (1 + self.regularization_params[i])
        # output = self.unnorm(output, mean_input, std_input)
        return output

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor):
        x = ifft2c_new(x)
        return complex_mul(x, complex_conj(sens_maps)).sum(dim=1, keepdim=True)

    def sens_expand(self, x: torch.Tensor, sens_map):
        return complex_mul(x, complex_mul(x, sens_map))

    def rss_combine_torch(self, data, axis):
        return torch.sqrt(torch.square(torch.abs(data)).sum(axis))

    def norm(self, x, meanStd=None):
        b, c, h, w, two = x.shape  # batch, height, width
        assert two == 2
        if meanStd is not None:
            mean, std = meanStd
            return (x - mean) / std, mean, std
        x = x.view(b, c, h * w, two)  # resize to (b, h*w)
        mean = torch.abs(x).mean(dim=2).view(b, c, 1, 1, two)
        std = torch.abs(x).std(dim=2).view(b, c, 1, 1, two)
        x = x.view(b, c, h, w, two)
        return (x - mean) / std, mean, std

    def unnorm(self, x, mean, std):
        return x * std + mean


# no sense estimation only airs model
class MyModel_V5(torch.nn.Module):
    def __init__(self, airs_inchans: int = 2,
                 airs_outchans: int = 2,
                 airs_pools: int = 4,
                 airs_chans: int = 128,
                 num_airs_layers: int = 1,
                 airs_layers_stateDict: OrderedDict = None,
                 regularization_params_stateDict: OrderedDict = None,
                 disable_train_index: int = -1,
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
        self.regularization_params = torch.nn.ParameterList()

        if train:
            n = disable_train_index + 2 if self.retrain else disable_train_index + 1

            if airs_layers_stateDict is not None:
                for _ in range(n):
                    self.airs_layers.append(AIRSLayer(airs_inchans, airs_outchans, airs_pools, airs_chans))
                self.airs_layers.load_state_dict(airs_layers_stateDict)

            if regularization_params_stateDict is not None:
                for _ in range(n):
                    self.regularization_params.append(torch.nn.Parameter(torch.ones(1)))
                self.regularization_params.load_state_dict(regularization_params_stateDict)

            if airs_layers_stateDict is not None:
                if self.retrain:
                    n = 0
                else:
                    n = self.num_airs_layers - len(self.airs_layers)
            else:
                n = self.num_airs_layers

            for _ in range(n):
                self.airs_layers.append(AIRSLayer(airs_inchans, airs_outchans, airs_pools, airs_chans))
                self.regularization_params.append(torch.nn.Parameter(torch.ones(1)))
        else:
            for _ in range(num_airs_layers):
                self.airs_layers.append(AIRSLayer(airs_inchans, airs_outchans, airs_pools, airs_chans))
                self.regularization_params.append(torch.nn.Parameter(torch.ones(1)))
            self.airs_layers.load_state_dict(airs_layers_stateDict)
            self.regularization_params.load_state_dict(regularization_params_stateDict)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input (b, 1, h, w) grappa image

        input = torch.complex(input, torch.zeros_like(input, dtype=torch.float32))
        input = torch.view_as_real(input)
        input = input.unsqueeze(1)
        # input, mean, std = self.norm(input)
        output = input
        for i, layer in enumerate(self.airs_layers):
            if i <= self.disable_train_index:
                with torch.no_grad():
                    output = self.forward_cascade(layer, output)
            else:
                # c개의 coil k-space
                output = self.forward_cascade(layer, output)

        # output = self.unnorm(output, mean, std)
        output = torch.abs(torch.view_as_complex(output))
        output = torch.clamp(output.squeeze(1), 0, 0.0014163791202008724)
        return output

    def forward_cascade(self, layer, output):
        add_later = output
        output = layer(output)
        output = add_later + output
        return output

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor):
        x = ifft2c_new(x)
        return complex_mul(x, complex_conj(sens_maps)).sum(dim=1, keepdim=True)

    def sens_expand(self, x: torch.Tensor, sens_map):
        return complex_mul(x, complex_mul(x, sens_map))

    def rss_combine_torch(self, data, axis):
        return torch.sqrt(torch.square(torch.abs(data)).sum(axis))

    def norm(self, x, meanStd=None):
        b, c, h, w, two = x.shape  # batch, height, width
        assert two == 2
        if meanStd is not None:
            mean, std = meanStd
            return (x - mean) / std, mean, std
        x = x.view(b, c, h * w, two)  # resize to (b, h*w)
        mean = torch.abs(x).mean(dim=2).view(b, c, 1, 1, two)
        std = torch.abs(x).std(dim=2).view(b, c, 1, 1, two)
        x = x.view(b, c, h, w, two)
        return (x - mean) / std, mean, std

    def unnorm(self, x, mean, std):
        return x * std + mean
