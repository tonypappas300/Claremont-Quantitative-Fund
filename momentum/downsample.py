import numpy as np
import pandas as pd
from tqdm import tqdm

def downsample(seq, n=30, weekdays=None, cols=None):
    if weekdays is not None and n == 7:
        seq = seq[weekdays.values.tolist().index(2):-1-weekdays[::-1].values.tolist().index(2)]
        if cols is not None:
            cols = cols[weekdays.values.tolist().index(2):-1-weekdays[::-1].values.tolist().index(2)]
    else:
        seq = seq[len(seq) % n:]
        if cols is not None:
            cols = cols[len(seq) % n:]
    return np.prod(seq.reshape((-1, n)) + 1.0, axis=-1) - 1.0, cols if cols is None else cols[::n][:len(seq)//n]

def downsample_list(seqs, n=7, weekdays=None, cols=None):
    s1, _cols = downsample(seqs[0], n=n, weekdays=weekdays, cols=cols)
    return np.stack([s1] + [downsample(s, n=n, weekdays=weekdays)[0] for s in tqdm(seqs[1:])]), _cols

def downsample_df(df, n=7):
    weekdays = None
    df.columns = pd.to_datetime(df.columns, format="%Y%m%d")
    df_range = pd.date_range(df.columns.min(), df.columns.max())
    df[df_range[~(df_range.isin(df.columns))]] = 0.0
    if n == 7:
        weekdays = df.columns.weekday
    data, cols = downsample_list(df.values, n=n, weekdays=weekdays, cols=df.columns)
    return pd.DataFrame(data=data, columns=cols, index=df.index)

if __name__ == "__main__":
    print(downsample(np.array(list(range(100)))/100, n=30))
    df = pd.read_parquet("../data/full_rseqs.parquet").drop(columns=["permno"])
    df = df[df.columns.sort_values(ascending=True)]
    print(downsample_df(df, n=7))
    print(downsample_df(df, n=30))
    #print(pd.DataFrame(df.resample("W-WED", closed="left", axis=1)))