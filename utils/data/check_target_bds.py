import h5py
import numpy as np
import os
import glob

files = glob.glob('/root/input/train/image/*.h5')
print(len(files))
files =sorted(files)

target_min = 0
target_max = 0
j = 0
for i, file in enumerate(files):
    if i % 300 == 0:
        print(i)
        print(target_min, target_max)
    with h5py.File(file, 'r') as f:
        targets = f['image_label'][:]
        temp1 = targets.max()
        temp2 = targets.min()
        if target_max < temp1:
            target_max = temp1
        if target_min > temp2:
            target_min = temp2

print("result")
print("min {}, max {}".format(target_min, target_max))


# min 0, max 0.0014163791202008724