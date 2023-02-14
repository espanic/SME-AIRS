import numpy as np
from utils.data.load_data import create_data_loaders
from utils.common.loss_function import SSIMLoss
from utils.common.utils import ifftc_torch, ssim_loss, save_reconstructions
from model_cascade import MyModel_Cascade
from collections import defaultdict, OrderedDict
import torch
import time
import os
import cv2
import shutil
from tqdm import tqdm
import pathlib
from utils.learning.test_part import test
from evaluation.leaderboard_eval import forward
from plot_result import plot_image


def train_cascade(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    print('Current cuda device: ', torch.cuda.current_device())

    train_loader = create_data_loaders(args.data_path_train + 'image/', args, getBoth=True)
    val_loader = create_data_loaders(args.data_path_val + 'image/', args, getBoth=True, shuffle=False)
    loss_type = SSIMLoss().to(device=device)

    if args.validate_only:
        print("start validation only")
        file = args.exp_dir / 'best_model.pt'
        dicts = torch.load(file)['model']
        model = MyModel_Cascade(num_airs_layers=args.train_cascade,
                                airs_chans=args.airs_chans,
                                airs_pools=4,
                                )

        model.to(device=device)
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate_both(args, model, val_loader,
                                                                                           loss_type)
        val_loss = val_loss / num_subjects
        print(
            'validation only! : '
            f'ValLoss = {val_loss:.4g}'
        )
        return

    startEpoch = 0
    model = MyModel_Cascade(num_airs_layers=args.train_cascade,
                            airs_chans=args.airs_chans,
                            airs_pools=4,
                            )

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.1
    if (epoch >= 5) else 1, last_epoch=-1)
    best_val_loss = 1.

    if args.retrain:
        dicts = torch.load(args.pretrained_dir, map_location='cpu')
        modelDict = dicts['model']
        # opDict = dicts['optimizer']
        # schDict = dicts['scheduler']
        # optimizer.load_state_dict(opDict)
        # scheduler.load_state_dict(schDict)
        model.load_state_dict(modelDict)
        # model.airs_layers.load_state_dict(modelDict['airs_layers'])
        startEpoch = dicts['epoch'] + 1
        best_val_loss = dicts['best_val_loss']
        print(startEpoch - 1, best_val_loss)

    args.air_inchans = model.air_inchans
    args.air_outchans = model.air_outchans
    args.air_chans = model.airs_chans
    args.airs_pools = model.airs_pools
    args.num_airs_layers = model.num_airs_layers

    f = os.path.join(args.config_dir, 'config_{}_{}.txt'.format(args.train_cascade, args.nick_name))
    g = os.path.join(args.config_dir, 'train_log_{}_{}.txt'.format(args.net_name, args.train_cascade))
    if not args.retrain:
        with open(f, 'w') as file:
            for arg in sorted(vars(args)):
                attr = getattr(args, arg)
                file.write('{} = {}\n'.format(arg, attr))
            file.write("\n")
        with open(g, 'w') as file:
            file.write("train log start\n")

    model.to(device=device)
    # optimizer_to(optimizer, device)
    # scheduler_to(scheduler, device)

    print("intial validation")
    # val_loss, num_subjects, _, _, _, val_time = validate_both(args, model, val_loader,
    #                                                           loss_type, None)
    # print(val_loss / num_subjects)

    for epoch in range(startEpoch, args.num_epochs):
        print(scheduler.get_last_lr())
        model, optimizer, train_loss, time = train_epoch_both(args, model, optimizer, loss_type, epoch, train_loader, g)
        print("total loss : {}, training time{}".format(train_loss, time))
        # scheduler.step()

        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate_both(args, model, val_loader,
                                                                                           loss_type, g)
        val_loss = val_loss / num_subjects

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g}'
            f'ValTime = {val_time:.4g}'
        )

        with open(g, 'a') as file:
            file.write(f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
                       f'ValLoss = {val_loss:.4g}'
                       f'ValTime = {val_time:.4g}')
            file.write('\n')

        with open(f, 'a') as file:
            file.write('epoch{} train_loss = {}\n'.format(epoch, train_loss))
        with open(f, 'a') as file:
            file.write('epoch{} val_loss = {}\n'.format(epoch, val_loss))
        with open(g, 'a') as file:
            file.write('epoch{} train_loss = {}\n'.format(epoch, train_loss))
        with open(g, 'a') as file:
            file.write('epoch{} val_loss = {}\n'.format(epoch, val_loss))

        save_model(args, args.exp_dir, epoch, optimizer, scheduler,
                   best_val_loss, is_new_best, model)

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            with open(g, 'a') as file:
                file.write("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")

            # save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)


def validate_both(args, model, data_loader, loss_type, g):
    model.eval()
    inputs = defaultdict(dict)
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start = time.perf_counter()

    print(f"validation data length : {len(data_loader)}")

    # metric_loss = 0
    total_loss = 0

    test_result = test_leaderboard(args, model)
    print("leaderboard result is : {}".format(test_result))
    if g is not None:
        with open(g, 'a') as file:
            file.write("leaderboard result is : {}\n".format(test_result))

    length = len(data_loader)
    with torch.no_grad():
        for i, data in tqdm(enumerate(data_loader), desc="validation precedure", mininterval=0.01, total=length):
            input, grappa, target, maximum, fnames, slices = data
            mask = create_mask(target)
            mask = torch.from_numpy(mask).cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)
            grappa = grappa.cuda(non_blocking=True)
            input = torch.cat([input.unsqueeze(1), grappa.unsqueeze(1)], dim=1)
            target = target.cuda(non_blocking=True)
            maximum = maximum.cuda(non_blocking=True)
            output = model(input, mask)
            # loss1 = loss_type(target * mask, output * mask, maximum)
            loss = loss_type(target * mask, output, maximum)
            total_loss += loss.item()
    return total_loss, length, reconstructions, targets, inputs, time.perf_counter() - start


# 0.01705492064356804

def train_epoch_both(args, model, optimizer, loss_type, epoch, data_loader, g):
    total_loss = 0
    start_iter = time.perf_counter()
    start = start_iter
    model.train()
    with open(g, 'a') as file:
        file.write(f'Epoch #{epoch:2d} ............... {args.net_name} : {args.train_cascade} ...............\n')
    print(f'Epoch #{epoch:2d} ............... {args.net_name} : {args.train_cascade} ...............')
    for i, data in tqdm(enumerate(data_loader), desc="train procedure ", mininterval=0.01, total=len(data_loader)):
        input, grappa, target, maximum, _, _ = data
        mask = create_mask(target)
        mask = torch.from_numpy(mask).cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        grappa = grappa.cuda(non_blocking=True)
        input = torch.cat([input.unsqueeze(1), grappa.unsqueeze(1)], dim=1)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)
        output = model(input, mask)
        assert output.shape == target.shape
        loss = loss_type(target * mask, output, maximum)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i % args.report_interval == 0:
            s = f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] ' \
                f'Iter = [{i:4d}/{len(data_loader):4d}] ' \
                f'Loss = {total_loss / (i + 1):.4g} ' \
                f'Current_Loss = {loss.item():.4g} ' \
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            print(s)
            with open(g, 'a') as file:
                file.write("{}\n".format(s))

        start_iter = time.perf_counter()

    total_loss = total_loss / len(data_loader)

    return model, optimizer, total_loss, time.perf_counter() - start


def rss_combine_torch(data, axis):
    return torch.sqrt(torch.square(torch.abs(data)).sum(axis))


def save_model(args, exp_dir, epoch, optimizer, scheduler, best_val_loss, is_new_best, model):
    # modelDict = OrderedDict()
    modelDict = model.state_dict()
    # modelDict['regularization_params'] = model.regularization_params.state_dict()

    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': modelDict,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def create_mask(target):
    target = target.numpy()
    mask = np.zeros(target.shape, dtype=np.float32)
    mask[target > 5e-5] = 1
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=15)
    mask = cv2.erode(mask, kernel, iterations=14)
    return mask


def test_leaderboard(args, model):
    model.eval()
    data_loader = create_data_loaders('/root/input/leaderboard/', args=args, isforward=True, getBoth=True)
    args.your_data_path = pathlib.Path('../result/model_cascade/reconstructions_forward/')
    args.model_id = 'v6'
    recons, _ = test(args, model, data_loader)
    save_reconstructions(recons, args.your_data_path)
    del recons
    args.output_key = 'reconstruction'
    args.leaderboard_data_path = args.leaderboard
    return forward(args)


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def scheduler_to(sched, device):
    for param in sched.__dict__.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
