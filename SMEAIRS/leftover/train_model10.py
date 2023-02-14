from SME import AdaptiveSensitivityModel
from utils.data.load_data import create_data_loader_modelv9
from utils.common.loss_function import SSIMLoss
from mymodel6 import MyModel_V10
from collections import defaultdict, OrderedDict
import torch
import time
import os
import shutil
from tqdm import tqdm
from utils.learning.test_part import test, save_reconstructions
from plot_result import plot_image


def train_model10(args):
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
        dicts = torch.load(file)['model']
        model = MyModel_V10(airs_inchans=4)
        model.to(device=device)
        model.load_state_dict(dicts)
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = _validate(args, model, val_loader,
                                                                                       loss_type)
        val_loss = val_loss / num_subjects
        print(
            'validation only! : '
            f'ValLoss = {val_loss:.4g}'
        )
        return


    startEpoch = 0
    model = MyModel_V10(airs_inchans=2)
    model.to(device=device)




    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    best_val_loss = 1.
    args.num_airs_layers = model.num_airs_layers

    if args.train_weak_only:
        f = os.path.join(args.config_dir, 'config_{}.txt'.format(args.weak_name))
    else:
        f = os.path.join(args.config_dir, 'config_{}_final_max_norm.txt'.format(args.train_cascade))

    if args.retrain:
        print("retraining : {}".format(args.weak_dir))
        if args.train_weak_only:
            dicts = torch.load(args.weak_dir)
            state_dict = dicts['model']
            optimizer.load_state_dict(dicts['optimizer'])
            startEpoch = dicts['epoch'] + 1
            model.load_state_dict(state_dict)
    else:
        with open(f, 'w') as file:
            for arg in sorted(vars(args)):
                attr = getattr(args, arg)
                file.write('{} = {}\n'.format(arg, attr))
            file.write("\n")

    for epoch in range(startEpoch, args.num_epochs):
        model, optimizer, train_loss, required_time = train_epoch(args, model, optimizer, loss_type, epoch,
                                                                  train_loader)
        print("total loss : {}, training time : {}".format(train_loss, required_time))

        val_loss, num_subjects, reconstructions, targets, inputs, val_time = _validate(args, model, val_loader,
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

        save_model(args, args.exp_dir, epoch, optimizer,
                   best_val_loss, is_new_best, model)

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            # save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)


def _validate(args, model, data_loader, loss_type):
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
            input, _, target, mask, _, maximum, _, _, _ = data
            input = input.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(input, mask)
            target = target.cuda(non_blocking=True)
            maximum = maximum.cuda(non_blocking=True)
            loss = loss_type(target, output, maximum)
            total_loss += loss.item()
    return total_loss, length, reconstructions, targets, inputs, time.perf_counter() - start


def train_epoch(args, model, optimizer, loss_type, epoch, data_loader):
    total_loss = 0
    start_iter = time.perf_counter()
    start = start_iter
    model.train()
    print(f'Epoch #{epoch:2d} ............... {args.net_name} : {args.train_cascade} ...............')

    for i, data in tqdm(enumerate(data_loader), desc="train procedure ", mininterval=0.01, total=len(data_loader)):
        input, _, target, mask, _, maximum, _, _, _ = data
        input = input.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        output = model(input, mask)
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
        start_iter = time.perf_counter()

    total_loss = total_loss / len(data_loader)

    return model, optimizer, total_loss, time.perf_counter() - start


def save_model(args, exp_dir, epoch, optimizer, best_val_loss, is_new_best, model):
    modelDict = OrderedDict()

    modelDict['airs_layers'] = model.airs_layers.state_dict()

    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': modelDict,
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')

# def test_leaderboard(args, model):
#     data_loader = create_data_loader_modelv9(args.leaderboard_data_path, args=args, shuffle=False)
#     recons, _ = test(args, model, data_loader)
#     save_reconstructions(recons, args.your_data_path)
#     del recons
#     args.output_key = 'reconstruction'
#     return forward(args)
