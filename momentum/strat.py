import numpy as np

def returns(p_seqs):
    """
    returns based on input price sequences
    Arguments:
        p_seqs: NDArray (m x n) = n sequential closing prices for m companies
    Returns:
        NDArray (m x n - 1) = n sequential returns for m companies
    """
    d = p_seqs[:, 1:]
    return (d/p_seqs[:, :-1]) - 1

def strict_risk_adjusted_returns(p_seqs, return_stdev=False, verbose=True):
    """
    risk adjusted returns (mean return divided by standard deviation of returns)
    Arguments:
        p_seqs: NDArray (m x n) = n sequential closing prices for m companies
        return_stdev: bool = whether to return calculated standard deviations
        verbose: bool = whether to print warnings about standard deviations = 0
    Returns:
        NDArray (m) = risk adjusted returns for m companies
    """
    _returns = returns(p_seqs)
    mu_r = np.mean(_returns, axis=-1)
    stdev = np.std(_returns, axis=-1)
    m = stdev == 0
    if verbose and np.sum(mu_r[m]) > 0:
        print("WARNING: 0 stdev with nonzero return")
    stdev[m] = 1
    mu_r[m] = 0
    if return_stdev:
        return mu_r/stdev, stdev
    return mu_r/stdev

def r_seq_risk_adjusted(r_seqs, return_stdev=False, verbose=True):
    """
    risk adjusted returns (mean return divided by standard deviation of returns)
    Arguments:
        r_seqs: NDArray (m x n) = n sequential daily returns for m companies
        return_stdev: bool = whether to return calculated standard deviations
        verbose: bool = whether to print warnings about standard deviations = 0
    Returns:
        NDArray (m) = risk adjusted returns for m companies
    """
    mu_r = np.mean(r_seqs, axis=-1)
    stdev = np.std(r_seqs, axis=-1)
    m = stdev == 0
    if verbose and np.sum(mu_r[m]) > 0:
        print("WARNING: 0 stdev")
    stdev[m] = 1e-5
    if return_stdev:
        return mu_r/stdev, stdev
    return mu_r/stdev

def get_weights(winners, losers):
    """
    get portion of portfolio to be allocated to each holding
    Arguments:
        winners: NDArray (m_w) = indices of m_w winners (stocks to be bought)
        losers: NDArray (m_l) = indices of m_l losers (stocks to be shorted)
    Returns:
        NDArray (m_w + m_l) = portion of portfolio to be allocated to each of winners and losers
    """
    w = np.concatenate([np.ones(shape=(winners.shape[0],)), -np.ones(shape=(losers.shape[0],))], axis=0)
    return w/np.sum(np.abs(w))

def strict_data_portfolio(observation_window, buy_p, invest_level=1, win_prop=.1, lose_prop=.1):
    """
    price-momentum strategy implementation
    Arguments:
        observation_window: NDArray (m x w) = w closing prices for m companies
        buy_p: NDArray (m) = current price for m companies
        invest_level: float = total absolute investment
        win_prop: float = portion of companies to buy
        lose_prop: float = portion of companies to short
    Returns:
        NDArray (m) = quantity to hold for each of m companies
    """
    num_stocks = observation_window.shape[0]
    num_win = max(int(win_prop * num_stocks), 1)
    num_lose = max(int(lose_prop * num_stocks), 1)
    idxs = np.arange(num_stocks)

    z_scores = strict_risk_adjusted_returns(observation_window)

    sorted_z_scores = np.sort(z_scores)
    win_thresh = sorted_z_scores[-num_win]
    winners = idxs[z_scores >= win_thresh]
    lose_thresh = sorted_z_scores[num_lose - 1]
    losers = idxs[z_scores <= lose_thresh]

    portfolio = np.concatenate([winners, losers], axis=0)

    all_weights = get_weights(winners, losers)
    q = invest_level * all_weights / buy_p[portfolio]

    calls = np.zeros((num_stocks,))
    calls[portfolio] = q
    return calls

def rseqs_portfolio(observation_window, win_prop=.1, lose_prop=.1):
    """
    price-momentum strategy implementation using return sequences
    Arguments:
        observation_window: NDArray (m x w) = w daily returns for m companies
        win_prop: float = portion of companies to buy
        lose_prop: float = portion of companies to short
    Returns:
        NDArray (m) = quantity to hold for each of m companies
    """
    num_stocks = observation_window.shape[0]
    num_win = max(int(win_prop * num_stocks), 1)
    num_lose = max(int(lose_prop * num_stocks), 1)
    idxs = np.arange(num_stocks)

    z_scores = r_seq_risk_adjusted(observation_window)

    sorted_z_scores = np.sort(z_scores)
    win_thresh = sorted_z_scores[-num_win]
    winners = idxs[z_scores >= win_thresh]
    lose_thresh = sorted_z_scores[num_lose - 1]
    losers = idxs[z_scores <= lose_thresh]

    portfolio = np.concatenate([winners, losers], axis=0)

    q = get_weights(winners, losers)

    calls = np.zeros((num_stocks,))
    calls[portfolio] = q
    return calls