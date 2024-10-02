from multiprocessing import Pool, Manager, Array
import os

import pandas as pd
import numpy as np
from tqdm import tqdm
import ctypes

def wrap_applied_func(q, func):
    def f(n, *args):
        q.put((n, func(*args)))
    return f

def shared_df(df):
    df_dtypes_dict = dict(list(zip(df.columns, df.dtypes)))
    mparr = Array(ctypes.c_double, df.values.reshape(-1))
    return pd.DataFrame(np.frombuffer(mparr.get_obj()).reshape(df.shape), columns=df.columns).astype(df_dtypes_dict)

def delegate_apply(func, x):
    with Manager() as m:
        with Pool(processes=os.cpu_count() - 1) as p:
            q = m.Queue()
            #f = wrap_applied_func(q, func)
            l = []
            results = []
            for n, i in enumerate(x):
                results.append(None)
                l.append(p.apply_async(func, (q, n, i)))
            for _ in tqdm(list(enumerate(l))):
                try:
                    n, r = q.get()
                    results[n] = r
                except KeyboardInterrupt:
                    break
    return results