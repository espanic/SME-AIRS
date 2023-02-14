import torch
import numpy as np
from utils.common.utils import ssim_loss, fftc, ifftc
from mymodel2 import MyModel_V6
from mymodel3 import MyModel_V7
from mymodel4 import MyModel_V8
from utils.data.load_data import create_data_loaders, create_data_loader_SMEAIRS
import os
import argparse
from glob import glob
from plot_result import plot_image_3, plot_image
from tqdm import tqdm

testdirs = glob('root')


def parse():
    parser = argparse.ArgumentParser(description='Custom model training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-v', '--data-path-val', type=str, default='/root/input/val/',
                        help='Directory of validation data')
    parser.add_argument('--input-key', type=str, default='image_input', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--model-path', type=str,
                        default='/root/myproject/trained/SMEAIRS_v6_input_grappa/checkpoints/level0/best_model.pt',
                        help='Name of max key in attributes')
    parser.add_argument('--model-version', type=str,
                        default='v6',
                        help='version of model')

    args = parser.parse_args()
    return args


def load_model(path, model_version):
    dicts = torch.load(path)
    modelDict = dicts['model']
    if model_version == 'v6':
        model = MyModel_V8(airs_inchans=4,
                           num_airs_layers=1,
                           # air_inchans_with_filtered=2,
                           airs_layers_stateDict=modelDict['airs_layers'],
                           # filtering=False,
                           adding_noise=True,
                           train=False)
    elif model_version == 'v7':
        model = MyModel_V7(sme_stateDict=modelDict['sme'], airs_layers_stateDict=modelDict['airs_layers'],
                           num_airs_layers=
                           1, train=False)

    return model


if __name__ == '__main__':
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    args = parse()
    if args.model_version == 'v6':
        dataloader = create_data_loaders('/root/input/leaderboard/image', args, shuffle=False, getBoth=True)
    if args.model_version == 'v7':
        dataloader = create_data_loader_SMEAIRS('/root/input/leaderboard/', 1, shuffle=False)
    model = load_model(args.model_path, args.model_version)
    model.to(device=device)
    total_loss = 0

    losses = []
    if args.model_version == 'v6':
        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader), desc="testing", total=len(dataloader)):
                input, grappa, target, maximum, fname, slice = data
                input = input.cuda(non_blocking=True)
                grappa = grappa.cuda(non_blocking=True)
                input = torch.cat([input.unsqueeze(1), grappa.unsqueeze(1)], dim=1)
                output = model(input)

                input = input.cpu().numpy()
                output = output.cpu().numpy()
                target = target.cpu().numpy()
                loss = ssim_loss(target, output, maximum.item())
                total_loss += loss
                losses.append(loss)
                if i % 50 == 0:
                    # plot_image_3(input[0], output[0], target[0])
                    print(f'{i} : {1 - loss}')
                    print(f'{i} : {1 - total_loss / (i + 1)}')
        print(1 - total_loss / len(dataloader))

    else:
        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader), desc="testing", total=len(dataloader)):

                input, grappa, target, mask, acs, maximum, _, _, _ = data
                input = input.cuda(non_blocking=True)
                grappa = grappa.cuda(non_blocking=True)
                acs = acs.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                maximum = maximum.cuda(non_blocking=True)
                output = model(input, grappa, acs)

                output = output.cpu().numpy()
                target = target.cpu().numpy()
                loss = ssim_loss(target, output, maximum.item())
                total_loss += loss
                if i % 50 == 0:
                    # plot_image_3(input[0], output[0], target[0])
                    print(f'{i} : {1 - loss}')
                    print(f'{i} : {1 - total_loss / (i + 1)}')
        print(1 - total_loss / len(dataloader))

