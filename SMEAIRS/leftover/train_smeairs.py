import argparse

from pathlib import Path
from SMEAIRS.train_part import train
from SMEAIRS.train_grappa_only import train_grappa
from train_grappa_with_filter import train_grappa_with_filter
from train_model7 import train_model7


def parse():
    parser = argparse.ArgumentParser(description='Custom model training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-e', '--num-epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('-l', '--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('-r', '--report-interval', type=int, default=300, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='SMEAIRS_v6_input_grappa_1', help='Name of network')
    parser.add_argument('-t', '--data-path-train', type=str, default='/root/input/train/',
                        help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=str, default='/root/input/val/',
                        help='Directory of validation data')

    parser.add_argument('--in-chans', type=int, default=1, help='Size of input channels for network')
    parser.add_argument('--out-chans', type=int, default=1, help='Size of output channels for network')
    parser.add_argument('--chans', type=int, default=18, help='the number of channels of the first convolution layer')
    parser.add_argument('--input-key', type=str, default='image_input', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--validate_only', type=bool, default=False, help='only validate validation data')
    parser.add_argument('--train-cascade', type=int, default=0, help='what level of cascade to train')
    parser.add_argument('--retrain', type=bool, default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()

    args.exp_dir = '../trained' / args.net_name / 'checkpoints' / 'level{}'.format(args.train_cascade)
    args.config_dir = '../trained' / args.net_name
    args.val_dir = '../trained' / args.net_name / 'reconstructions_val'
    args.main_dir = '../trained' / args.net_name / __file__
    if args.train_cascade > 0:
        args.pretrained_dir = '../trained' / args.net_name / 'checkpoints' / 'level{}'.format(args.train_cascade - 1) / \
                              'best_model.pt'
    if args.retrain:
        args.pretrained_dir = '../trained' / args.net_name / 'checkpoints' / 'level{}'.format(args.train_cascade) / \
                              'best_model.pt'

    if not args.validate_only:
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        args.val_dir.mkdir(parents=True, exist_ok=True)
    # train(args)
    # train_grappa(args)
    args.further_explain = 'grappa residual instead input residual'
    train_grappa_with_filter(args)

    # train_model7(args)

# validation result
# cascade 0 : 0.04893
# cascade 1 : #0.04764
# cascade 2 : #0.04762

# with raw input
# 0.04668
