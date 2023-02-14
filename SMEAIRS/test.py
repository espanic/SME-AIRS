
from mymodel import MyModel, MyModel_V2
import torch
from utils.data.load_data import create_data_loader_SMEAIRS
import matplotlib.pyplot as plt
from utils.common.utils import ssim_loss
import numpy as np

if __name__ =='__main__':
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    model2 = True
    torch.cuda.set_device(device)
    if model2:
        state_dict = torch.load('/root/myproject/trained/SMEAIRS_Clipped/checkpoints/level1/best_model.pt')['model']
        model = MyModel_V2(num_airs_layers=2,
                           sme_stateDict=state_dict['sme'],
                           airs_layers_stateDict=state_dict['airs_layers'],
                           regularization_params_stateDict=state_dict['regularization_params'],
                           train=False
                           )
    else:
        model = MyModel(airs_chans=128, num_airs_layers=1)
        state_dict = torch.load('/root/myproject/result/SMEAIRS_norm_1layer/checkpoints/best_model.pt')['model']
        model.load_state_dict(state_dict)
    model.to(device=device)
    model.eval()
    data_loader = create_data_loader_SMEAIRS('/root/input/val/', 1, shuffle=False)
    with torch.no_grad():
        for data in data_loader:
            input, target, mask, acs_mask, maximum, kfname, ifname, slice = data
            input = input.cuda(non_blocking=True)
            acs_mask = acs_mask.cuda(non_blocking=True)

            output = model(input, acs_mask).cpu().numpy()[0]
            target = target.cpu().numpy()[0]
            output = np.clip(output, 0, 0.0014163791202008724)

            print(ssim_loss(target, output))
            plt.figure()
            plt.subplot(1, 3, 1)
            plt.imshow(np.clip(output, 0, target.max()), cmap='gray')
            plt.subplot(1, 3, 2)
            plt.imshow(target, cmap='gray')
            plt.subplot(1, 3, 3)
            plt.imshow(np.abs(target - output), cmap='gray')
            plt.show()
            print("a")







