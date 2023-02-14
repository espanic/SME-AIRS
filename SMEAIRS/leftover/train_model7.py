from SME import AdaptiveSensitivityModel
from utils.data.load_data import create_data_loader_SMEAIRS
from utils.common.loss_function import SSIMLoss
from utils.common.utils import ifftc_torch, ssim_loss, save_reconstructions
from mymodel3 import MyModel_V7
from collections import defaultdict, OrderedDict
import torch
import time
import os
import numpy as np
import shutil
from tqdm import tqdm


def train_model7(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())

    train_loader = create_data_loader_SMEAIRS(args.data_path_train, 1, shuffle=False)
    val_loader = create_data_loader_SMEAIRS(args.data_path_val, 1, shuffle=False)
    loss_type = SSIMLoss().to(device=device)

    if args.validate_only:
        print("start validation only")
        file = args.exp_dir / 'best_model.pt'
        dicts = torch.load(file)['model']
        model = MyModel_V7(num_airs_layers=args.train_cascade + 1,
                           airs_inchans=4,
                           airs_layers_stateDict=dicts['airs_layers'],
                           sme_stateDict=dicts['sme'],
                           disable_train_index=args.train_cascade - 1,
                           retrain=args.retrain,
                           train=False)
        model.to(device=device)
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader,
                                                                                      loss_type)
        val_loss = val_loss / num_subjects
        print(
            'validation only! : '
            f'ValLoss = {val_loss:.4g}'
        )
        return

    startEpoch = 0
    if args.train_cascade > 0 or args.retrain:
        dicts = torch.load(args.pretrained_dir)
        modelDict = dicts['model']
        opDict = dicts['optimizer']
        schDict = dicts['scheduler']
        model = MyModel_V7(num_airs_layers=args.train_cascade + 1,
                           airs_layers_stateDict=modelDict['airs_layers'],
                           disable_train_index=args.train_cascade - 1,
                           airs_inchans=4,
                           retrain=args.retrain)

    else:
        model = MyModel_V7(airs_inchans=4)
    model.to(device=device)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=1, last_epoch=-1)
    best_val_loss = 1.

    if args.retrain:
        optimizer.load_state_dict(opDict)
        scheduler.load_state_dict(schDict)
        startEpoch = dicts['epoch'] + 1
        best_val_loss = dicts['best_val_loss']

    args.air_inchans = model.air_inchans
    args.air_outchans = model.air_outchans
    args.air_chans = model.airs_chans
    args.num_airs_layers = model.num_airs_layers

    f = os.path.join(args.config_dir, 'config_{}.txt'.format(args.train_cascade))
    if not args.retrain:
        with open(f, 'w') as file:
            for arg in sorted(vars(args)):
                attr = getattr(args, arg)
                file.write('{} = {}\n'.format(arg, attr))
            file.write("\n")

    for epoch in range(startEpoch, args.num_epochs):
        model, optimizer, train_loss, time = train_epoch(args, model, optimizer, loss_type, epoch, train_loader)
        print("total loss : {}, training time{}".format(train_loss, time))
        scheduler.step()

        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader,
                                                                                      loss_type)
        val_loss = val_loss / num_subjects

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g}'
            f'ValTime = {val_time:.4g}'
        )

        with open(f, 'a') as file:
            file.write('epoch{} train_loss = {}\n'.format(epoch, train_loss))
        with open(f, 'a') as file:
            file.write('epoch{} val_loss = {}\n'.format(epoch, val_loss))

        save_model(args, args.exp_dir, epoch, optimizer, scheduler,
                   best_val_loss, is_new_best, model)

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            # save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)


def validate(args, model, data_loader, loss_type):
    model.eval()
    inputs = defaultdict(dict)
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start = time.perf_counter()

    print(f"validation data length : {len(data_loader)}")

    # metric_loss = 0
    total_loss = 0
    length = len(data_loader)
    with torch.no_grad():
        for i, data in tqdm(enumerate(data_loader), desc="validation precedure", mininterval=0.01, total=length):
            input, grappa, target, mask, acs, maximum, _, _, _ = data
            input = input.cuda(non_blocking=True)
            grappa = grappa.cuda(non_blocking=True)
            acs = acs.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            maximum = maximum.cuda(non_blocking=True)
            output = model(input,grappa, acs)
            loss = loss_type(target, output, maximum)
            total_loss += loss.item()
    return total_loss, length, reconstructions, targets, inputs, time.perf_counter() - start


def train_epoch(args, model, optimizer, loss_type, epoch, data_loader):
    total_loss = 0
    start_iter = time.perf_counter()
    start = start_iter
    print(f'Epoch #{epoch:2d} ............... {args.net_name} : {args.train_cascade} ...............')
    for i, data in tqdm(enumerate(data_loader), desc="train procedure ", mininterval=0.01, total=len(data_loader)):
        input, grappa, target, mask, acs, maximum, _, _, _ = data
        input = input.cuda(non_blocking=True)
        acs = acs.cuda(non_blocking=True)
        grappa = grappa.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)

        output = model(input, grappa, acs)
        assert output.shape == target.shape
        loss = loss_type(output, target, maximum)
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
        start_iter = time.perf_counter()

    total_loss = total_loss / len(data_loader)

    return model, optimizer, total_loss, time.perf_counter() - start


def rss_combine_torch(data, axis):
    return torch.sqrt(torch.square(torch.abs(data)).sum(axis))


def save_model(args, exp_dir, epoch, optimizer, scheduler, best_val_loss, is_new_best, model):
    modelDict = OrderedDict()
    modelDict['airs_layers'] = model.airs_layers.state_dict()
    modelDict['sme'] = model.sme.state_dict()

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
