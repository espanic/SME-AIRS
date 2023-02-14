import shutil
import numpy as np
import torch
import time
import os
from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions, ssim_loss, mse_loss, ssim_complex_loss
from utils.common.loss_function import SSIMLoss, MSEComplexLoss, SSIMComplexLoss
from utils.model.kikinet import Inet, Knet
from utils.model.unet import Unet
from torchsummary import torchsummary
from torch.utils.tensorboard import SummaryWriter
from utils.common.utils import ifftc_torch, fftc_torch
from torch.nn import functional


def train_epoch(args, epoch, model, data_loader, optimizer, loss_type):
    if args.model == 'kinet':
        kmodel = model[0]
        kmodel.eval()
        model = model[1]
        model.train()
    elif args.model == 'kiknet':
        kmodel1 = model[0]
        imodel1 = model[1]
        kmodel1.eval()
        imodel1.eval()
        model = model[2]
        model.train()

    elif args.model =='kikinet':
        kmodel1, imodel1, kmodel2, model = model
        kmodel1.eval()
        imodel1.eval()
        kmodel2.eval()
        model.train()

    else:
        model.train()

    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    for iter, data in enumerate(data_loader):
        input, target, maximum, _, _ = data
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)
        if args.model == 'kinet':
            with torch.no_grad():
                # kspace 가 나옴.
                input = kmodel(input)
                # complex image로 변환 함.
                input = ifftc_torch(input)
        elif args.model == 'kiknet':
            with torch.no_grad():
                input = kmodel1(input)
                input = ifftc_torch(input)
                input = imodel1(input)
                input = fftc_torch(input)
                # input = input.unsqueeze(1)

        elif args.model == 'kikinet':
            with torch.no_grad():
                input = kmodel1(input)
                input = ifftc_torch(input)
                input = imodel1(input)
                input = fftc_torch(input)
                input = kmodel2(input)
                input = ifftc_torch(input)

        output = model(input)

        if args.model == 'knet' or args.model == 'kiknet':
            loss = loss_type(output, target, args.batch_size)
        else:
            loss = loss_type(output, target, maximum)

        optimizer.zero_grad()
        if not torch.isfinite(loss):
            print('WARNING : non-finite loss, end training')
            continue
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()

    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch


def validate(args, model, data_loader):
    if args.model == 'kinet':
        kmodel = model[0]
        model = model[1]
        kmodel.eval()
        model.eval()
    elif args.model == 'kiknet':
        kmodel1 = model[0]
        imodel1 = model[1]
        model = model[2]
        kmodel1.eval()
        imodel1.eval()
        model.eval()
    elif args.model =='kikinet':
        kmodel1, imodel1, kmodel2, model = model
        kmodel1.eval()
        imodel1.eval()
        kmodel2.eval()
        model.eval()
    else:
        model.eval()

    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    inputs = defaultdict(dict)
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            input, target, _, fnames, slices = data
            input = input.cuda(non_blocking=True)

            if args.model == 'kinet':
                # kspace 가 나옴.
                input = kmodel(input)
                # complex image로 변환 함.
                input = ifftc_torch(input)

            elif args.model == 'kiknet':
                input = kmodel1(input)
                input = ifftc_torch(input)
                input = imodel1(input)
                input = fftc_torch(input)
            elif args.model == 'kikinet':
                input = kmodel1(input)
                input = ifftc_torch(input)
                input = imodel1(input)
                input = fftc_torch(input)
                input = kmodel2(input)
                input = ifftc_torch(input)


            output = model(input)
            if args.model == 'inet':
                output = torch.real(output)
            if args.model == 'kinet':
                input = torch.abs(input)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].numpy()
                inputs[fnames[i]][int(slices[i])] = input[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )
    for fname in inputs:
        inputs[fname] = np.stack(
            [out for _, out in sorted(inputs[fname].items())]
        )
        if args.model == 'unet':
            metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
        elif args.model == 'inet':
            # metric_loss = sum([mse_loss(targets[fname], reconstructions[fname])  for fname in reconstructions])
            metric_loss = sum([ssim_complex_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
        elif args.model == 'knet' or args.model == 'kiknet':
            metric_loss = sum([mse_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
        elif args.model == 'kinet' or 'kikinet':
            metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
            # metric_loss = sum([mse_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
    num_subjects = len(reconstructions)
    return metric_loss, num_subjects, reconstructions, targets, inputs, time.perf_counter() - start


def save_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())

    scheduler = None
    getKSpace = False
    targetKspace = False
    if args.model == 'unet':

        model = Unet(in_chans=args.in_chans, out_chans=args.out_chans)
        model.to(device=device)
        loss_type = SSIMLoss().to(device=device)
        optimizer = torch.optim.Adam(model.parameters(), args.lr)

    elif args.model == 'knet':
        num_layer = 20
        args.num_layer = num_layer
        model = Knet(in_chans=args.in_chans, out_chans=args.out_chans, num_layers=num_layer)
        model.to(device=device)
        loss_type = MSEComplexLoss().to(device=device)
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
        getKSpace = True
        targetKspace = True

    elif args.model == 'kinet':
        # train inet based on the output of the kmodel
        # load kmodel
        kmodel_state_dict = torch.load(os.path.join(args.model_dir, 'checkpoints', 'best_model.pt'))['model']
        kmodel = Knet(args.in_chans, args.out_chans, 20)
        kmodel.load_state_dict(kmodel_state_dict)
        kmodel.to(device=device)

        num_layer = 20
        args.num_layer = num_layer
        imodel = Inet(in_chans=args.in_chans, out_chans=args.out_chans, num_layers=num_layer)
        imodel.to(device=device)
        model = (kmodel, imodel)

        loss_type = SSIMLoss().to(device=device)
        optimizer = torch.optim.Adam(model[1].parameters(), args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=8, gamma=0.1, last_epoch=-1)
        getKSpace = True
        targetKspace = False

    elif args.model == 'kiknet':
        kmodel_state_dict = torch.load(os.path.join(args.model_dir, 'Knet_Use_Raw', 'checkpoints', 'best_model.pt'))[
            'model']
        kmodel1 = Knet(args.in_chans, args.out_chans, 20)
        kmodel1.load_state_dict(kmodel_state_dict)
        kmodel1.to(device=device)
        imodel1_state_dict = torch.load(os.path.join(args.model_dir, 'Inet_Use_Raw', 'checkpoints', 'best_model.pt'))[
            'model']
        imodel1 = Inet(args.in_chans, args.out_chans, 20)
        imodel1.load_state_dict(imodel1_state_dict)
        imodel1.to(device=device)

        num_layer = 20
        args.num_layer = num_layer
        kmodel2 = Knet(in_chans=args.in_chans, out_chans=args.out_chans, num_layers=num_layer)
        kmodel2.to(device=device)
        loss_type = MSEComplexLoss().to(device=device)
        optimizer = torch.optim.Adam(kmodel2.parameters(), args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.1, last_epoch=-1)
        getKSpace = True
        targetKspace = True
        model = (kmodel1, imodel1, kmodel2)

    elif args.model == 'kikinet':
        kmodel1_state_dict = torch.load(os.path.join(args.model_dir, 'Knet_Use_Raw', 'checkpoints', 'best_model.pt'))[
            'model']
        kmodel1 = Knet(args.in_chans, args.out_chans, 20)
        kmodel1.load_state_dict(kmodel1_state_dict)
        kmodel1.to(device=device)
        imodel1_state_dict = torch.load(os.path.join(args.model_dir, 'Inet_Use_Raw', 'checkpoints', 'best_model.pt'))[
            'model']
        imodel1 = Inet(args.in_chans, args.out_chans, 20)
        imodel1.load_state_dict(imodel1_state_dict)
        imodel1.to(device=device)

        kmodel2_state_dict = torch.load(os.path.join(args.model_dir, 'Knet_Use_Raw2', 'checkpoints', 'best_model.pt'))[
            'model']
        kmodel2 = Knet(in_chans=args.in_chans, out_chans=args.out_chans, num_layers=20)
        kmodel2.load_state_dict(kmodel2_state_dict)
        kmodel2.to(device=device)

        args.num_layer = 20
        imodel2 = Inet(in_chans=args.in_chans, out_chans=args.out_chans, num_layers=args.num_layer)
        imodel2.to(device=device)
        loss_type = SSIMLoss().to(device=device)
        optimizer = torch.optim.Adam(imodel2.parameters(), args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.1, last_epoch=-1)
        getKSpace = True
        targetKspace = False
        model = (kmodel1, imodel1, kmodel2, imodel2)


    # torchsummary.summary(model, (384, 384), batch_size=args.batch_size)

    # create config file
    f = os.path.join(args.config_dir, 'config.txt')
    b = os.path.join(args.config_dir, 'best_info.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))

    writer = SummaryWriter()

    best_val_loss = 1.
    start_epoch = 0

    train_loader = create_data_loaders(data_path=args.data_path_train, args=args, getKSpace=getKSpace,
                                       targetKSpace=targetKspace)
    val_loader = create_data_loaders(data_path=args.data_path_val, args=args, getKSpace=getKSpace,
                                     targetKSpace=targetKspace, shuffle=False)

    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')

        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, loss_type)

        writer.add_scalar("Loss/train", train_loss, epoch)
        scheduler.step()
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader)
        # ssim = ssim/num_subjects
        val_loss = val_loss / num_subjects
        writer.add_scalar("Loss/val", val_loss, epoch)
        # writer.add_image("Image/val", reconstructions['brain60.h5'][:1], epoch)
        # writer.add_image("Image/target", targets['brain60.h5'][:1], epoch)

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        if args.model == 'kinet':
            model_save = model[1]
        elif args.model == 'kiknet':
            model_save = model[2]
        elif args.model =='kikinet':
            model_save = model[3]
        else:
            model_save = model

        save_model(args, args.exp_dir, epoch + 1, model_save, optimizer,
                   best_val_loss, is_new_best)
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )
        # print("ssim =  {}".format(ssim))

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )
            with open(b, 'w') as file:
                file.write('{} = {}\n'.format('best_val_loss', best_val_loss))
                file.write('{} = {}\n'.format('final epoch', epoch))
                # file.write('{} = {}\n'.format('ssim', ssim))

        writer.flush()
