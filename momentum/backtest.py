import numpy as np
from strat import rseqs_portfolio, strict_data_portfolio
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

def backtest(data, window_size, hold_time, investment_level=1, win_prop=.1, lose_prop=.1, progress_verbose=True):
    """
    backtest price-momentum strategy
    Arguments:
        data: NDArray (m x n) = full price sequences to be backtested on with n closing prices for m companies
        window_size: int = number of closing prices preceeding current price to use in momentum calculation
        hold_time: int = predetermined amount of time to hold stock after making call
        investment_level: float = total absolute stake invested (should not affect final result)
        win_prop: float = portion of companies to buy
        lose_prop: float = portion of companies to short
        progress_verbose: bool = whether or not to print progress bar
    Returns:
        float = mean return on absolute investment of 1 dollar
    """
    data_length = data.shape[1]
    num_examples = data_length - window_size - 1 - hold_time
    returns = []
    r = range(num_examples)
    if progress_verbose:
        r = tqdm(r)
    for i in r:
        window = data[:, i: i + window_size]
        buy_p = data[:, i + window_size]
        mask = np.sum(np.concatenate([window, np.expand_dims(buy_p, -1)], axis=-1) == 0, axis=-1) <= 0

        window = window[mask]
        buy_p = buy_p[mask]
        sell_p = data[:, i + window_size + hold_time][mask]

        if np.sum(mask) <= 100:
            continue

        port = strict_data_portfolio(window, buy_p, invest_level=investment_level, win_prop=win_prop, lose_prop=lose_prop)
        
        spent = (buy_p * port)
        gain = (sell_p * port) - spent
        returns.append((np.sum(spent) - np.sum(gain))/np.sum(np.abs(spent)))

    return np.mean(returns)

def r_seq_backtest(data, window_size, hold_time, annualize=False, win_prop=.1, lose_prop=.1, progress_verbose=True, indices=None):
    """
    backtest price-momentum strategy using return sequences
    Arguments:
        data: NDArray (m x n) = full return sequences to be backtested on with n daily returns for m companies
        window_size: int = number of returns preceeding current return to use in momentum calculation
        hold_time: int = predetermined amount of time to hold stock after making call
        annualize: bool = whether to annualize return
        win_prop: float = portion of companies to buy
        lose_prop: float = portion of companies to short
        progress_verbose: bool = whether or not to print progress bar
    Returns:
        float = mean return on portfolio
    """
    an = (365/hold_time) if annualize else 1
    data_length = data.shape[1]
    num_examples = data_length - window_size - 1 - hold_time
    returns = []
    all_baseline = []
    columns = []
    index_baselines = {k: [] for k in indices.index} if indices is not None else {}
    r = range(num_examples)
    if progress_verbose:
        r = tqdm(r)
    for i in r:
        window = data.iloc[:, i: i + window_size]
        m = np.logical_not(np.isnan(np.sum(window.values, axis=-1)))
        window = window.values[m]
        hold = data.iloc[:, i + window_size: i + window_size + hold_time]
        hc = hold.columns
        hold = hold.values[m]

        if window.shape[0] <= 20:
            continue

        port = rseqs_portfolio(window, win_prop=win_prop, lose_prop=lose_prop)

        ret = np.prod(hold + 1.0, axis=-1) ** an
        ret[np.isnan(ret)] = 0.0
        ret[port < 0] = 2 - ret[port < 0]
        port = np.abs(port)
        returns.append(np.sum(port * ret)/np.sum(port))
        all_baseline.append(np.mean(ret))
        columns.append(data.columns[i + window_size])
        for k in index_baselines.keys():
            index_baselines[k].append(np.prod(1 + indices.loc[k, hc].values))
    
    return pd.DataFrame.from_dict({"returns": returns, "baseline": all_baseline, "yyyymmdd": columns} | index_baselines)

if __name__ == "__main__":
    df = pd.read_parquet("../data/full_rseqs.parquet").drop(columns=["permno"])
    idxdf = pd.read_parquet("../data/idx_rseqs.parquet", index_col=0).drop("date").astype(float)
    result = r_seq_backtest(df[df.columns.sort_values(ascending=True)], 210, 125, indices=idxdf).set_index("yyyymmdd")
    result.index = pd.to_datetime(result.index, format="%Y%m%d")
    result.to_csv("./boutput.csv", date_format='%Y%m%d')
    result.plot().set_yscale("log")
    plt.show()