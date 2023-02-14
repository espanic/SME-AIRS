import argparse
from pathlib import Path
from utils.learning.test_part import forward


def parse():
    parser = argparse.ArgumentParser(description='Test Unet on FastMRI challenge Images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU_NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-n', '--net_name', type=Path, default='SMEAIRS_v6_input_grappa', help='Name of network')
    parser.add_argument('-p', '--data_path', type=str, default='/root/input/leaderboard/image/',
                        help='Directory of test data')

    parser.add_argument('--in-chans', type=int, default=1, help='Size of input channels for network')
    parser.add_argument('--out-chans', type=int, default=1, help='Size of output channels for network')
    parser.add_argument("--input_key", type=str, default='image_input', help='Name of input key')
    parser.add_argument("--model_id", type=str, default='v6', help='model identification')
    parser.add_argument("--model_dir", type=str,
                        default='/root/myproject/trained/SMEAIRS_v6_input_grappa/checkpoints/level1/best_model.pt',
                        help='model identification')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    args.exp_dir = '../trained' / args.net_name  / 'checkpoints' / 'level0'
    args.forward_dir = '../result' / args.net_name / 'reconstructions_forward_level0'
    print(args.forward_dir)
    forward(args)
