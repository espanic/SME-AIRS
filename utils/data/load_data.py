import h5py
import torch

from utils.data.transforms import DataTransform, DataTransform2, CropAndTransform, CropAndTransform2
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from utils.common.utils import fftc
import numpy as np


# h5 데이터를 이미지로 변환


class SliceData(Dataset):
    def __init__(self, root, transform, input_key, target_key, forward=False, getKspace=False, targetKspace=False):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.examples = []
        self.getKspace = getKspace
        self.targetKspace = targetKspace

        files = list(Path(root).iterdir())
        for fname in sorted(files):
            num_slices = self._get_metadata(fname)

            self.examples += [
                (fname, slice_ind) for slice_ind in range(num_slices)
            ]

    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            num_slices = hf[self.input_key].shape[0]
        return num_slices

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, dataslice = self.examples[i]
        with h5py.File(fname, "r") as hf:
            input = hf[self.input_key][dataslice]
            if self.getKspace:
                input = fftc(input)
                input = input.astype(np.complex64)
            if self.forward:
                target = -1
            else:
                target = hf[self.target_key][dataslice]
                if self.targetKspace:
                    target = fftc(target)
                    target = target.astype(np.complex64)
            attrs = dict(hf.attrs)
        return self.transform(input, target, attrs, fname.name, dataslice)


class SliceData_Both(Dataset):
    def __init__(self, root, transform, input_key1, input_key2, target_key, forward=False, getKspace=False,
                 targetKspace=False):
        self.transform = transform
        self.input_key1 = input_key1
        self.input_key2 = input_key2
        self.target_key = target_key
        self.forward = forward
        self.examples = []
        self.getKspace = getKspace
        self.targetKspace = targetKspace

        files = list(Path(root).iterdir())
        for fname in sorted(files):
            num_slices = self._get_metadata(fname)

            self.examples += [
                (fname, slice_ind) for slice_ind in range(num_slices)
            ]

    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            num_slices = hf[self.input_key1].shape[0]
        return num_slices

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, dataslice = self.examples[i]
        with h5py.File(fname, "r") as hf:
            input1 = hf[self.input_key1][dataslice]
            input2 = hf[self.input_key2][dataslice]
            if self.getKspace:
                input1 = fftc(input1)
                input1 = input1.astype(np.complex64)
            if self.forward:
                target = -1
            else:
                target = hf[self.target_key][dataslice]
                if self.targetKspace:
                    target = fftc(target)
                    target = target.astype(np.complex64)
            attrs = dict(hf.attrs)
        return self.transform(input1, input2, target, attrs, fname.name, dataslice)


class SliceData_SMEAIRS(Dataset):
    def __init__(self, root, transform, input_key='kspace', target_key='image_label', mask_key='mask',
                 grappa_key='image_grappa'):
        self.kspace_root = root + 'kspace//'
        self.image_root = root + 'image/'
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.mask_key = mask_key
        self.grappa_key = grappa_key
        self.examples = []
        self.masks = []

        kspace_files = sorted(list(Path(self.kspace_root).iterdir()))
        image_files = sorted(list(Path(self.image_root).iterdir()))
        assert len(kspace_files) == len(image_files)
        for i, kfname, ifname in zip(range(len(kspace_files)), kspace_files, image_files):
            num_slices, mask = self._get_metadata(kfname)

            self.examples += [
                (kfname, ifname, slice_ind, i) for slice_ind in range(num_slices)  # ex : (brain1.
            ]
            self.masks.append(mask)

    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            num_slices = hf[self.input_key].shape[0]
            mask = hf[self.mask_key][:]
        return num_slices, mask

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        kfname, ifname, dataslice, mask_idx = self.examples[i]
        mask = self.masks[mask_idx]
        with h5py.File(kfname, "r") as hf:
            input = hf[self.input_key][dataslice]
        with h5py.File(ifname, "r") as hf:
            target = hf[self.target_key][dataslice]
            grappa = hf[self.grappa_key][dataslice]
            attrs = dict(hf.attrs)
        return self.transform(input, grappa, mask, target, kfname.name, ifname.name, dataslice, attrs)


class SliceData_v9(Dataset):
    def __init__(self, root, transform, input_key1='kspace', input_key2='image_grappa', target_key='image_label',
                 mask_key='mask',
                 grappa_key='image_grappa'):
        self.kspace_root = root + 'kspace/'
        self.image_root = root + 'image/'
        self.transform = transform
        self.input_key1 = input_key1
        self.input_key2 = input_key2
        self.target_key = target_key
        self.mask_key = mask_key
        self.examples = []
        self.masks = []

        kspace_files = sorted(list(Path(self.kspace_root).iterdir()))
        image_files = sorted(list(Path(self.image_root).iterdir()))
        assert len(kspace_files) == len(image_files)
        for i, kfname, ifname in zip(range(len(kspace_files)), kspace_files, image_files):
            num_slices, mask = self._get_metadata(kfname)

            self.examples += [
                (kfname, ifname, slice_ind, i) for slice_ind in range(num_slices)  # ex : (brain1.
            ]
            self.masks.append(mask)

    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            num_slices = hf[self.input_key1].shape[0]
            mask = hf[self.mask_key][:]
        return num_slices, mask

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        kfname, ifname, dataslice, mask_idx = self.examples[i]
        mask = self.masks[mask_idx]
        with h5py.File(kfname, "r") as hf:
            input = hf[self.input_key1][dataslice]
        with h5py.File(ifname, "r") as hf:
            target = hf[self.target_key][dataslice]
            grappa = hf[self.input_key2][dataslice]
            attrs = dict(hf.attrs)
        return self.transform(input, grappa, mask, target, kfname.name, ifname.name, dataslice, attrs)


def create_data_loaders(data_path, args, isforward=False, getKSpace=False, targetKSpace=False, shuffle=True,
                        getBoth=False):
    if not isforward:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1

    if getBoth:
        data_storage = SliceData_Both(
            root=data_path,
            transform=DataTransform2(isforward, max_key_),
            input_key1=args.input_key,
            input_key2='image_grappa',
            target_key=target_key_,
            forward=isforward,
            getKspace=getKSpace,
            targetKspace=targetKSpace
        )

    else:
        data_storage = SliceData(
            root=data_path,
            transform=DataTransform(isforward, max_key_),
            input_key=args.input_key,
            target_key=target_key_,
            forward=isforward,
            getKspace=getKSpace,
            targetKspace=targetKSpace
        )

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        shuffle=shuffle
    )
    return data_loader


def create_data_loader_SMEAIRS(data_path, batch_size, cropInput=True, shuffle=True):
    data_storage = SliceData_SMEAIRS(
        root=data_path,
        transform=CropAndTransform(cropInput),
    )

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=batch_size,
        shuffle=shuffle
    )
    return data_loader


if __name__ == '__main__':
    data_loader = create_data_loader_SMEAIRS('/root/input/train/', 1, shuffle=False)
    target_max = 0
    target_min = 0
    for i, data in enumerate(data_loader):
        # (1, c, h, w, 2), (1, h, w), (w,)
        if i % 500 == 0:
            print(i)
        input, target, mask, acs_mask, maximum, kfname, ifname, slice = data
        temp1 = target.max()
        temp2 = target.min()
        if target_max < temp1:
            target_max = temp1
        if target_min > temp2:
            target_min = temp2

        # assert input.shape[-3:-1] == target.shape[-2:] == (384, 384)

    print(target_max, target_min)


def create_data_loader_modelv9(data_path, args, isforward=False, getKSpace=False, cropInput=True, getACS = False, shuffle=True):
    data_storage = SliceData_v9(
        root=data_path,
        transform=CropAndTransform2(cropInput, getKSpace, getACS),
    )

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        shuffle=shuffle
    )
    return data_loader
