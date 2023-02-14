import argparse
from pathlib import Path
from train_model9 import train_model9


def parse():
    parser = argparse.ArgumentParser(description='Custom model training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-e', '--num-epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('-l', '--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('-r', '--report-interval', type=int, default=300, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='model_v9', help='Name of network')
    parser.add_argument('-t', '--data-path-train', type=str, default='/root/input/train/',
                        help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=str, default='/root/input/val/',
                        help='Directory of validation data')

    parser.add_argument('--input-key', type=str, default='image_input', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--validate_only', type=bool, default=False, help='only validate validation data')
    parser.add_argument('--train_weak_only', type=bool, default=False, help='train only weak layer')
    parser.add_argument('--further_explain', type=str, default='max normalization version level 0',
                        help='explanation of model')
    parser.add_argument('--train-cascade', type=int, default=0, help='what level of cascade to train')
    parser.add_argument('--retrain', type=bool, default=True)
    parser.add_argument('--weak-name', type=str, default='weak_max_norm')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    if args.train_weak_only:
        args.exp_dir = '../trained' / args.net_name / 'checkpoints' / args.weak_name
    else:
        args.exp_dir = '../trained' / args.net_name / 'checkpoints' / 'level{}_v2'.format(args.train_cascade)
        args.weak_dir = '../trained' / args.net_name / 'checkpoints' / args.weak_name / 'best_model.pt'

    args.config_dir = '../trained' / args.net_name
    args.val_dir = '../trained' / args.net_name / 'reconstructions_val'
    args.main_dir = '../trained' / args.net_name / __file__
    if args.train_cascade > 0:
        args.pretrained_dir = '../trained' / args.net_name / 'checkpoints' / 'level{}_weak'.format(
            args.train_cascade - 1) / \
                              'best_model.pt'
    if args.retrain:
        args.pretrained_dir = '../trained' / args.net_name / 'checkpoints' / 'level{}'.format(args.train_cascade) / \
                              'best_model.pt'
        args.weak_dir = '../trained' / args.net_name / 'checkpoints' / args.weak_name / 'best_model.pt'

    if not args.validate_only:
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        args.val_dir.mkdir(parents=True, exist_ok=True)

    train_model9(args)
