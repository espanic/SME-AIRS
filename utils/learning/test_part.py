import numpy as np
import torch

from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders, create_data_loader_modelv9
from SMEAIRS.mymodel2 import MyModel_V6
from SMEAIRS.mymodel5 import MyModel_V9
from SMEAIRS.plot_result import plot_image
from tqdm import tqdm


def test(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    inputs = defaultdict(dict)

    with torch.no_grad():
        if args.model_id == 'v6':
            for (input, grappa, _, _, fnames, slices) in tqdm(data_loader, desc="evaluate leaderboard",
                                                              total=len(data_loader)):
                input = input.cuda(non_blocking=True)
                grappa = grappa.cuda(non_blocking=True)
                input = torch.cat([input.unsqueeze(1), grappa.unsqueeze(1)], dim=1)
                output = model(input)

                for i in range(output.shape[0]):
                    reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                    inputs[fnames[i]][int(slices[i])] = input[i].cpu().numpy()
        elif args.model_id == 'v9':
            for (input, grappa, _, mask, _, _, _, fnames, slices) in tqdm(data_loader, desc="evaluate leaderboard",
                                                              total=len(data_loader)):
                input = input.cuda(non_blocking=True)
                grappa = grappa.cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True)
                output = model(input, grappa, mask)


                for i in range(output.shape[0]):
                    reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                    # inputs[fnames[i]][int(slices[i])] = input[i].cpu().numpy()
            inputs = None

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )

    if inputs is not None:
        for fname in inputs:
            inputs[fname] = np.stack(
                [out for _, out in sorted(inputs[fname].items())]
            )
    return reconstructions, inputs


def forward(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device ', torch.cuda.current_device())

    checkpoint = torch.load(args.exp_dir / 'best_model.pt', map_location='cpu')
    print(checkpoint['epoch'], checkpoint['best_val_loss'])

    if args.model_id == 'v6':
        model = MyModel_V6(num_airs_layers=1,
                           airs_layers_stateDict=checkpoint['model']['airs_layers'],
                           disable_train_index=-1,
                           airs_inchans=4,
                           air_inchans_with_filtered=2,
                           filtering=False,
                           train=False,
                           retrain=False)
        model.to(device=device)

        forward_loader = create_data_loaders(data_path=args.data_path, args=args, isforward=True, getBoth=True, shuffle=False)

    elif args.model_id == 'v9':
        model = MyModel_V9(airs_inchans=6)
        model.load_state_dict(checkpoint['model'],load_weak_only=False, prior_trained_cascade_level=0)
        model.to(device=device)

        forward_loader = create_data_loader_modelv9(args.data_path, args, shuffle=False)
    print('validating model named {}'.format(args.model_id))
    reconstructions, inputs = test(args, model, forward_loader)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)
