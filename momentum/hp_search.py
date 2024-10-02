from multiprocessing import Pool, Manager

import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from glob import glob

from backtest import r_seq_backtest

_hyperparameters = {
    "win_prop": (.1, [.1, .5]),
    "lose_prop": (.1, [.1, .5]),
    "hold_time": (1, [1, 180]),
    "window_size": (10, [3, 723])
}
mini_hyperparameters = {
    "win_prop": (.1, [.4, .5]),
    "lose_prop": (.1, [.4, .5]),
    "hold_time": (1, [1, 3]),
    "window_size": (10, [20, 50])
}
wh = {"window_size": (30, [10, 4*360]), "hold_time": (5, [1, 180])}

def _btinstance(i, data, space):
    """
    backtest instance
    Arguments:
        i: int = current search space index
        data: NDArray = price sequence data to pass to backtest
        space: {str: NDArray (n)} = hyperparameters with respective spaces
    Returns:
        float = result of backtest
    """
    return backtest(data, **{k: v[i] for k, v in space.items()}, progress_verbose=False)

def q_btinstance(i, data, space, q):
    """
    backtest instance outputting search space index and backtest result to multiprocessing queue
    Arguments:
        i: int = current search space index
        data: NDArray = price sequence data to pass to backtest
        space: {str: NDArray (n)} = hyperparameters with respective spaces
        q: multiprocessing.Queue = multiprocessing queue for output
    """
    sp = {k: v[i] for k, v in space.items()}
    rdf = r_seq_backtest(data, **sp, progress_verbose=False, annualize=False)
    for k, v in sp.items():
        rdf[k] = v
    q.put((i, rdf))

def search(data, hyperparameters, chunk=5):
    """
    hyperparameter search
    Arguments:
        data: NDArray = price sequence data to pass to backtest
        hyperparameters: {str: (float, [float, float])} = hyperparameter names mapped to incremental change in the search space and the interval to be searched
    Returns:
        pd.DataFrame = dataframe containing hyperparameters and results of backtest for each combination
    """
    hp = list(hyperparameters.keys())
    search_lines = {
        k: [
                v[1][0] + i * v[0] for i in range(
                    1 + int((v[1][1] - v[1][0])/v[0])
                )
            ]
        for k, v in hyperparameters.items()
    }

    search_space = {k: np.reshape(v, (-1,)) for k, v in zip(hp, np.meshgrid(*list(search_lines.values())))}
    space_size = len(search_space[list(search_space.keys())[0]])

    results = [0 for _ in range(space_size)]
    try:
        with Manager() as m:
            with Pool(processes=os.cpu_count() - 1) as p:
                q = m.Queue()
                l = []
                for i in range(space_size):
                    l.append(p.apply_async(q_btinstance, (i, data, search_space, q)))
                last_log = []
                for i in tqdm(range(space_size)):
                    n, r = q.get()
                    last_log.append(r)
                    if not chunk:
                        results[n] = r
                    elif (i + 1) % chunk == 0:
                        pd.concat(last_log, axis=0).set_index(list(search_space.keys()) + ["yyyymmdd"]).to_csv("./hpchunks/chunk_{}.csv".format(str((i + 1) // chunk)))
                        last_log = []
    except:
        results = list(filter(lambda x: id(x) != id(0), results))
    if not chunk:
        res = pd.concat(results, axis=0).set_index(list(search_space.keys()) + ["yyyymmdd"])
    else:
        res = pd.concat(last_log + [pd.read_csv(i) for i in glob("./hpchunks/chunk_*.csv")], axis=0).set_index(list(search_space.keys()) + ["yyyymmdd"])
    return res

if __name__ == "__main__":
    SAMPLE_SIZE = .5
    data = pd.read_csv("../data/full_rseqs.csv")
    if SAMPLE_SIZE is not None:
        u = data["permno"].unique()
        samp = np.random.choice(u, int(SAMPLE_SIZE * len(u)))
        data = data[data["permno"].isin(samp)]
    data = data.drop(columns=["permno"])
    res = search(data, {"window_size": (3, [180, 280]), "hold_time": (3, [50, 150])})
    res.to_csv("./hpoutput.csv")
    print(res)
    _res = pd.read_csv("./hpoutput.csv")
    _res = _res.set_index(list(filter(lambda x: x not in ["returns", "baseline"], _res.columns)))
    print(_res)