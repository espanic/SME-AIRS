from SME import AdaptiveSensitivityModel
from utils.data.load_data import create_data_loader_modelv9
from utils.common.loss_function import SSIMLoss
from final_model import FinalModel, FinalModel_v2
from coil_residual_model import Coil_Model
from collections import defaultdict, OrderedDict
import torch
import time
import os
import shutil
import pathlib
from tqdm import tqdm
from utils.learning.test_part import test, save_reconstructions
from evaluation.leaderboard_eval import forward
from plot_result import plot_image


def train_final(args):
    if args.model_version == 'v1':
        model_class = FinalModel
    elif args.model_version == 'v2':
        model_class = FinalModel_v2
    elif args.model_version == 'coil_residual':
        model_class = Coil_Model

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())
    torch.cuda.empty_cache()

    train_loader = create_data_loader_modelv9(args.data_path_train, args, cropInput=False, getKSpace=True, shuffle=True)
    val_loader = create_data_loader_modelv9(args.data_path_val, args, cropInput=False, getKSpace=True, shuffle=False)
    loss_type = SSIMLoss().to(device=device)

    if args.validate_only:
        print("start validation only")
        file = args.exp_dir / 'best_model.pt'
        dicts = torch.load(file, map_location='cpu')['model']
        model = model_class()
        model.load_state_dict(dicts)
        model.to(device=device)
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = _validate(args, model, val_loader,
                                                                                       loss_type)
        val_loss = val_loss / num_subjects
        print(
            'validation only! : '
            f'ValLoss = {val_loss:.4g}'
        )
        return

    # define model and load pretrained weights
    startEpoch = 0
    model = model_class()
    model.set_sme_requires_grad(False)
    if args.use_sme_pretrained:
        print("use sme pretrained")
        model.init_sens_map(torch.load(args.pretrained_sme_dir))
    if args.use_airs_pretrained:
        print("use airs pretrained")
        try:
            model.init_layer(torch.load(args.pretrained_airs_dir)['model']['airs_layers'])
        except NotImplementedError as e:
            temp = pathlib.WindowsPath
            pathlib.WindowsPath = pathlib.PosixPath
            model.init_layer(torch.load(args.pretrained_airs_dir)['model']['airs_layers'])
            pathlib.WindowsPath = temp

    model.to(device=device)

    def scheduler_lamda(epoch):
        if epoch >= 5:
            return 1
        else:
            return 1

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lr_lambda=scheduler_lamda,
                                                  last_epoch=-1)
    best_val_loss = 1.
    # args.num_airs_layers = model.num_airs_layers

    f = os.path.join(args.config_dir, 'config_{}_{}.txt'.format(args.net_name, args.train_cascade))
    g = os.path.join(args.config_dir, 'train_log_{}_{}.txt'.format(args.net_name, args.train_cascade))

    if args.retrain:
        print("retraining")
        file = args.exp_dir / 'best_model.pt'
        dicts = torch.load(file, map_location='cpu')
        model = model_class()
        model.load_state_dict(dicts['model'])
        model.to(device=device)
        optimizer.load_state_dict(dicts['optimizer'])
        scheduler.load_state_dict(dicts['scheduler'])
        startEpoch = dicts['epoch'] + 1
        best_val_loss = dicts['best_val_loss']
        print(startEpoch - 1, best_val_loss)
        del dicts
    else:
        with open(f, 'w') as file:
            for arg in sorted(vars(args)):
                attr = getattr(args, arg)
                file.write('{} = {}\n'.format(arg, attr))
            file.write("\n")
        with open(g, 'w') as file:
            file.write("---------------train log------------\n")

    for epoch in range(startEpoch, args.num_epochs):
        print("learning rate : {}".format(scheduler.get_last_lr()))
        with open(g, 'a') as file:
            file.write("learning rate : {}\n".format(scheduler.get_last_lr()))
        if epoch == 1:
            print("setting require grad on sme")
            model.set_sme_requires_grad(args.turn_on_sme)
        model, optimizer, train_loss, required_time = train_epoch(args, model, optimizer, loss_type, epoch,
                                                                  train_loader, g)
        print("total loss : {}, training time : {}".format(train_loss, required_time))

        val_loss, num_subjects, reconstructions, targets, inputs, val_time = _validate(args, model, val_loader,
                                                                                       loss_type, g)
        val_loss = val_loss / num_subjects
        scheduler.step()

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g}'
            f'ValTime = {val_time:.4g}'
        )

        with open(f, 'a') as file:
            file.write('epoch{} train_loss = {}\n'.format(epoch, train_loss))
            file.write('epoch{} val_loss = {}\n'.format(epoch, val_loss))
        with open(g, 'a') as file:
            file.write('epoch{} train_loss = {}\n'.format(epoch, train_loss))
            file.write('epoch{} val_loss = {}\n'.format(epoch, val_loss))

        save_model(args, args.exp_dir, epoch, optimizer, scheduler,
                   best_val_loss, is_new_best, model)

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            with open(g, 'a') as file:
                file.write("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
            # save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)


def _validate(args, model, data_loader, loss_type, g=None):
    model.eval()
    inputs = defaultdict(dict)
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start = time.perf_counter()

    print(f"validation data length : {len(data_loader)}")
    if g is not None:
        with open(g, 'a') as file:
            file.write(f"validation data length : {len(data_loader)} \n")

    # test_result = test_leaderboard(args, model)
    # print("leaderboard result is : {}".format(test_result))
    #
    # if g is not None:
    #     with open(g, 'a') as file:
    #         file.write("leaderboard result is : {}\n".format(test_result))

    total_loss = 0
    length = len(data_loader)
    with torch.no_grad():
        for i, data in tqdm(enumerate(data_loader), desc="validation precedure", mininterval=0.01, total=length):
            kspace, grappa, target, mask, _, maximum, _, _, _ = data
            kspace = kspace.cuda(non_blocking=True)
            grappa = grappa.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(kspace, grappa, mask)
            target = target.cuda(non_blocking=True)
            maximum = maximum.cuda(non_blocking=True)
            loss = loss_type(target, output, maximum)
            total_loss += loss.item()
    return total_loss, length, reconstructions, targets, inputs, time.perf_counter() - start


def train_epoch(args, model, optimizer, loss_type, epoch, data_loader, g):
    total_loss = 0
    start_iter = time.perf_counter()
    start = start_iter
    model.train()
    model.set_sme_requires_grad(False)
    s = f'Epoch #{epoch:2d} ............... {args.net_name} : {args.train_cascade} ...............'
    print(s)
    with open(g, 'a') as file:
        file.write('{}\n'.format(s))
    for i, data in tqdm(enumerate(data_loader), desc="train procedure ", mininterval=0.01, total=len(data_loader)):
        kspace, grappa, target, mask, _, maximum, _, _, _ = data
        kspace = kspace.cuda(non_blocking=True)
        grappa = grappa.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        output = model(kspace, grappa, mask)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)
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
            with open(g, 'a') as file:
                file.write('{}\n'.format(s))
        start_iter = time.perf_counter()

    total_loss = total_loss / len(data_loader)

    return model, optimizer, total_loss, time.perf_counter() - start


def save_model(args, exp_dir, epoch, optimizer, scheduler, best_val_loss, is_new_best, model):
    # modelDict = OrderedDict()

    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model_{}.pt'.format(epoch)
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model_{}.pt'.format(epoch), exp_dir / 'best_model.pt')

def test_leaderboard(args, model):
    model.eval()
    data_loader = create_data_loader_modelv9(args.leaderboard_data_path, args, cropInput=False, getKSpace=True, shuffle=False)
    args.your_data_path = pathlib.Path('../result/coil_residual/reconstructions_forward/')
    args.model_id = 'v9'
    recons, _ = test(args, model, data_loader)
    save_reconstructions(recons, args.your_data_path)
    # del recons
    args.output_key = 'reconstruction'
    return forward(args)
