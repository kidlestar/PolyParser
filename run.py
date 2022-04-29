# -*- coding: utf-8 -*-

import argparse
import os
from parser.cmds import Evaluate, Predict, Train
from parser.config import Config
import random
import torch
import time

import pdb
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create the Biaffine Parser model.'
    )
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    subcommands = {
        'evaluate': Evaluate(),
        'predict': Predict(),
        'train': Train()
    }
    for name, subcommand in subcommands.items():
        subparser = subcommand.add_subparser(name, subparsers)
        subparser.add_argument('--conf', '-c', default='config.ini',
                               help='path to config file')
        subparser.add_argument('--file', '-f', default='exp/ptb',
                                help='path to saved files')
        # subparser.add_argument('--preprocess', '-p', action='store_true',
        #                        help='whether to preprocess the data first')
        # subparser.add_argument('--seed', '-s', default=1, type=int,
        #                        help='seed for generating random numbers')
        # subparser.add_argument('--threads', '-t', default=16, type=int,
        #                        help='max num of threads')
        # subparser.add_argument('--tree', action='store_true',
        #                        help='whether to ensure well-formedness')
        # subparser.add_argument('--feat', default='tag',
        #                        choices=['tag', 'char', 'bert'],
        #                        help='choices of additional features')
    args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Override the default configs with parsed arguments",flush=True)
    args = Config(args.conf).update(vars(args))

    print(f"Set the max num of threads to {args.threads}")
    print(f"Set the seed for generating random numbers to {args.seed}")
    # print(f"Set the device with ID {args.device} visible")
    random.seed(time.time())
    tseed = random.randint(0,999999)
    torch.set_num_threads(args.threads)
    #random.seed(time.time())
    print(f"Set the seed for generating random numbers to {tseed}")
    torch.manual_seed(tseed)

    #torch.manual_seed(args.seed)
    
    args.fields = os.path.join(args.file, 'fields')
    args.model = os.path.join(args.file, 'model')
    print(args)

    print(f"Run the subcommand in mode {args.mode}")
    cmd = subcommands[args.mode]
    cmd(args)
