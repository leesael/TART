import argparse
import itertools
import json
import multiprocessing as mp
import os
import subprocess
from copy import copy

import pandas as pd
import torch
import tqdm as tqdm

import utils
from data import get_datasets


def run_command(args):
    command, gpu_list = args
    gpu_idx = int(mp.current_process().name.split('-')[-1]) - 1
    gpu = gpu_list[gpu_idx % len(gpu_list)]
    command += ['--gpu', str(gpu)]
    return subprocess.check_output(command)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, nargs='+', default=[0])
    parser.add_argument('--data', type=str, nargs='+', default=None)
    parser.add_argument('--workers', type=int, default=4)
    return parser.parse_known_args()


def summarize_results(out_list, data_path):
    # Write hyperparameters.
    out_path = out_list[0]['out_path']
    with open(os.path.join(out_path, 'parameters.json'), 'w') as f:
        out_dict = copy(out_list[0])
        del out_dict['gpu']
        del out_dict['data']
        del out_dict['fold']
        json.dump(out_dict, f, indent=4)

    # Write detailed results.
    values = []
    for out_dict in out_list:
        values.append([
            out_dict['data'],
            out_dict['fold'],
            out_dict['result']['trn_acc'],
            out_dict['result']['test_acc'],
            out_dict['result']['params'],
            out_dict['result']['time']
        ])
    df_all = pd.DataFrame(values, columns=['data', 'fold', 'trn_acc', 'test_acc', 'params', 'time'])
    df_all.to_csv(os.path.join(out_path, 'details.tsv'), sep='\t', index=False)

    df_out = utils.summarize_results(df_all, data_path)
    df_out.to_csv(os.path.join(out_path, 'summary.tsv'), sep='\t', index=False)
    print('\t'.join(str(e) for e in df_out.iloc[-1, 1:].values))


def main():
    data_path = '../data'
    args, unknown = parse_args()
    if torch.cuda.is_available() and not args.gpu:
        args.gpu = list(range(torch.cuda.device_count()))
    if args.data is None:
        args.data = get_datasets(data_path)

    fold_list = list(range(4))
    args_list = []
    for d, f in itertools.product(args.data, fold_list):
        command = ['python', 'main.py',
                   '--data', d,
                   '--fold', str(f)]
        args_list.append((command + unknown, args.gpu))

    out_list = []
    with mp.Pool(len(args.gpu) * args.workers) as pool:
        for out in tqdm.tqdm(pool.imap_unordered(run_command, args_list), total=len(args_list)):
            out_list.append(json.loads(out))

    summarize_results(out_list, data_path)


if __name__ == '__main__':
    main()
