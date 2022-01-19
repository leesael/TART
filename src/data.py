import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


DATA_PATH = '../data'


def get_stats(path=DATA_PATH):
    def to_group(v):
        if v >= 10000:
            return 'large'
        elif v >= 1000:
            return 'mid'
        else:
            return 'small'

    stat_path = os.path.join(DATA_PATH, 'stats.tsv')
    if not os.path.exists(stat_path):
        uci_path = os.path.join(path, 'uci')
        datasets = []
        for file in os.listdir(uci_path):
            if os.path.isdir(os.path.join(uci_path, file)):
                datasets.append(file)
        if 'molec-biol-protein-second' in datasets:
            datasets.remove('molec-biol-protein-second')

        df = []
        for dataset in datasets:
            trn_x, trn_y, test_x, test_y = read_data(path, dataset, fold=0)
            size = trn_x.shape[0] + test_x.shape[0]
            group = to_group(size)
            nx = trn_x.shape[1]
            ny = np.concatenate([trn_y, test_y]).max() + 1
            df.append((group, dataset, size, nx, ny))
        df = pd.DataFrame(df, columns=['group', 'data', 'size', 'nx', 'ny'])
        df.sort_values(by='size', ascending=False, inplace=True)
        df.to_csv(stat_path, index=False, sep='\t')
    return pd.read_csv(stat_path, sep='\t')


def get_datasets(path=DATA_PATH):
    return get_stats(path)['data']


def to_loader(x, y, batch_size, shuffle=False):
    x = torch.tensor(x)
    y_type = torch.long if y.dtype == np.int else torch.float
    y = torch.tensor(y, dtype=y_type)
    return DataLoader(TensorDataset(x, y), batch_size, shuffle)


def read_files(data, path=DATA_PATH):
    info = {}
    with open(os.path.join(path, data, f'{data}.txt')) as f:
        out = f.readlines()
    for e in out:
        key, val = [w.strip() for w in e.split('=')]
        try:
            val = int(val)
        except ValueError:
            pass
        info[key] = val

    df_list = []
    for i in range(info['n_arquivos']):
        file = os.path.join(path, data, info[f'fich{i + 1}'])
        df = pd.read_csv(file, sep='\t', index_col=0)
        df.reset_index(drop=True, inplace=True)
        df_list.append(df)
    return df_list


def read_kfold(file, fold):
    with open(file) as f:
        out = f.readlines()
    trn_idx = np.array([int(e) for e in out[2 * fold].split()])
    test_idx = np.array([int(e) for e in out[2 * fold + 1].split()])
    return trn_idx, test_idx


def read_stats(data, path=DATA_PATH):
    df_list = read_files(data, path)
    if len(df_list) == 1:
        df = df_list[0]
    elif len(df_list) == 2:
        df = pd.concat(df_list)
    else:
        raise ValueError(len(df_list))

    n_data = df.shape[0]
    n_features = df.shape[1] - 1
    n_labels = len(df.iloc[:, -1].unique())
    return n_data, n_features, n_labels


def read_data(path, data, fold):
    path = os.path.join(path, 'uci')
    df_list = read_files(data, path)
    if len(df_list) == 1:
        trn_idx, test_idx = read_kfold(os.path.join(path, data, 'conxuntos_kfold.dat'), fold)
        trn_x = df_list[0].iloc[trn_idx, :-1].values
        trn_y = df_list[0].iloc[trn_idx, -1].values
        test_x = df_list[0].iloc[test_idx, :-1].values
        test_y = df_list[0].iloc[test_idx, -1].values
    elif len(df_list) == 2:
        trn_x = df_list[0].iloc[:, :-1].values
        trn_y = df_list[0].iloc[:, -1].values
        test_x = df_list[1].iloc[:, :-1].values
        test_y = df_list[1].iloc[:, -1].values
    else:
        raise ValueError(len(df_list))

    trn_x = trn_x.astype(np.float32)
    test_x = test_x.astype(np.float32)
    return trn_x, trn_y, test_x, test_y


def main():
    path = '../data'
    data_list = []
    for dataset in get_datasets(path):
        data_list.append((dataset, *read_stats(dataset, path)))
    data_list = sorted(data_list, key=lambda x: x[1], reverse=True)
    for d in data_list:
        print('\t'.join(str(s) for s in d))


if __name__ == '__main__':
    main()
