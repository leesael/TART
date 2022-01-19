from collections import defaultdict
import numpy as np
import pandas as pd

import data


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ['true']:
        return True
    elif v.lower() in ['false']:
        return False
    else:
        raise ValueError()


def summarize_results(df_all, data_path):
    val_columns = ['trn_acc', 'test_acc', 'params']
    out_columns = ['data', 'trn_avg', 'trn_std', 'test_avg', 'test_std', 'param_avg', 'param_std']

    df_stats = data.get_stats(data_path)
    df_all = df_all.merge(df_stats, on='data')

    df1 = defaultdict(lambda: [])
    for data_, df_ in df_all.groupby('data'):
        avg = df_[val_columns].mean(axis=0)
        std = df_[val_columns].std(axis=0)
        for col in val_columns:
            df1[data_].extend([avg[col], std[col]])
    df1 = [[d, *df1[d]] for d in df_stats['data']]
    df1 = pd.DataFrame(df1, columns=out_columns)

    def summarize(df):
        values = []
        for fold_, df_ in df.groupby('fold'):
            values.append([df_[val].mean() for val in val_columns])
        values_avg = np.array(values).mean(axis=0)
        values_std = np.array(values).std(axis=0)
        return [v for p in zip(values_avg, values_std) for v in p]

    df2 = defaultdict(lambda: [])
    for group_, df_ in df_all.groupby('group'):
        df2[group_] = summarize(df_)
    df2['all'] = summarize(df_all)
    df2 = [[d, *df2[d]] for d in ['large', 'mid', 'small', 'all']]
    df2 = pd.DataFrame(df2, columns=out_columns)
    return pd.concat([df1, df2])
