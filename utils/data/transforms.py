from utils.common.utils import ifftc, fftc
import torch
import numpy as np


def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    return torch.from_numpy(data)


class DataTransform:
    def __init__(self, isforward, max_key):
        self.isforward = isforward
        self.max_key = max_key

    def __call__(self, input, target, attrs, fname, slice):
        input = to_tensor(input)
        if not self.isforward:
            target = to_tensor(target)
            maximum = attrs[self.max_key]
        else:
            target = -1
            maximum = -1
        return input, target, maximum, fname, slice


class DataTransform2:
    def __init__(self, isforward, max_key):
        self.isforward = isforward
        self.max_key = max_key

    def __call__(self, input1, input2, target, attrs, fname, slice):
        input1 = to_tensor(input1)
        input2 = to_tensor(input2)
        if not self.isforward:
            target = to_tensor(target)
            maximum = attrs[self.max_key]
        else:
            target = -1
            maximum = -1
        return input1, input2, target, maximum, fname, slice


class CropAndTransform:

    def __init__(self, cropInput):
        self.cropInput = cropInput

    def __call__(self, input, grappa, mask, target, kfname, ifname, slice, attrs):
        # undersampling
        input = input * mask

        hw = target.shape[-2:]
        assert hw == (384, 384)

        # input to spatial domain
        input = ifftc(input)

        # cropping spatial domain
        if self.cropInput:
            input, mask_range = self.crop(input, hw)
        else:
            mask_range = 0, mask.shape[0]

        # input to frequency domain
        input = fftc(input).astype(np.complex64)
        input = to_tensor(input)

        maximum = attrs["max"]

        # input : 3차원의 텐서 channel, h, w (complex64)
        # target : 2차원 텐서 h, w (float32)
        # acs : 1차원 텐서 w (float32)
        return input, grappa, target, mask, self.get_acs(mask, mask_range), maximum, kfname, ifname, slice

    def crop(self, input, hw):
        h, w = hw
        center = input.shape[-2] // 2, input.shape[-1] // 2
        return input[..., center[0] - h // 2: center[0] + h // 2, center[1] - w // 2: center[1] + w // 2], (
            center[1] - w // 2, center[1] + w // 2)

    def get_acs(self, mask, mask_range):
        mask = mask[mask_range[0]: mask_range[1]]
        l = mask.shape[0]
        left, right = l // 2, l // 2 + 1
        while mask[left]:
            left -= 1
        while mask[right]:
            right += 1

        # left부터 right까지 모두 1임.
        acs = torch.zeros(l)
        acs[left:right + 1] = 1
        return acs


class CropAndTransform2:

    def __init__(self, cropInput, getKspace, getACS):
        self.cropInput = cropInput
        self.getKspace = getKspace
        self.getACS = getACS

    def __call__(self, input, grappa, mask, target, kfname, ifname, slices, attrs):
        # undersampling
        input = input * mask

        hw = target.shape[-2:]
        assert hw == (384, 384)

        # input to spatial domain
        if not self.getKspace:
            input = ifftc(input)

        # cropping spatial domain
        if self.cropInput:
            input, mask_range = self.crop(input, hw)
        else:
            mask_range = 0, mask.shape[0]

        input = input.astype(np.complex64)
        input = to_tensor(input)
        maximum = attrs["max"]

        # input : 3차원의 텐서 channel, h, w (complex64)
        # target : 2차원 텐서 h, w (float32)
        # acs : 1차원 텐서 w (float32)
        return input, grappa, target, mask, self.get_acs(mask, mask_range) if self.getACS else -1, maximum, kfname, ifname, slices

    def crop(self, input, hw):
        h, w = hw
        center = input.shape[-2] // 2, input.shape[-1] // 2
        return input[..., center[0] - h // 2: center[0] + h // 2, center[1] - w // 2: center[1] + w // 2], (
            center[1] - w // 2, center[1] + w // 2)

    def get_acs(self, mask, mask_range):
        mask = mask[mask_range[0]: mask_range[1]]
        l = mask.shape[0]
        left, right = l // 2, l // 2 + 1
        while mask[left]:
            left -= 1
        while mask[right]:
            right += 1

        # left부터 right 까지 모두 1임.
        acs = torch.zeros(l)
        acs[left:right + 1] = 1
        return acs
