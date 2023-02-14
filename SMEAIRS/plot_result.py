import h5py
import matplotlib.pyplot as plt
from utils.common.utils import ssim_loss
import numpy as np
if __name__ == '__main__':
    with h5py.File('/root/myproject/result/SMEAIRS_Cas_not_norm_2/reconstructions_val/brain60.h5', 'r') as f:
        input = f['input'][:]
        recon = f['reconstruction'][:]
        target = f['target'][:]

    index = 0
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(input[index], cmap='gray')
    plt.title('input')
    plt.subplot(1, 3, 2)
    plt.imshow(recon[index], cmap='gray')
    plt.title('recon')
    plt.subplot(1, 3, 3)
    plt.imshow(target[index], cmap='gray')
    plt.title('target')
    print(ssim_loss(target[index], input[index]))
    print(ssim_loss(target[index], recon[index]))
    plt.show()


def plot_image(x):
    plt.figure()
    plt.imshow(x, cmap='gray')
    plt.show()


def plot_image_3(input, recon, target):
    plt.figure()
    plt.subplot(1,4, 1)
    plt.title('input')
    plt.imshow(input, cmap='gray')
    plt.subplot(1, 4, 2)
    plt.imshow(recon, cmap='gray')
    plt.subplot(1, 4, 3)
    plt.imshow(target, cmap='gray')
    plt.subplot(1, 4, 4)
    plt.imshow(np.abs(recon - target), cmap='gray')
    plt.show()
