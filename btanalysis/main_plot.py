import sys
from datetime import datetime
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

#Sharpe, mean, geometric mean, volatility, alpha, beta, tracking error, realized volatility (30-day)

if __name__ == "__main__":
    BASELINE = "sprtrn"
    kill_bins = False
    if "kb" in sys.argv:
        kill_bins = True
        sys.argv = list(filter(lambda x: x != "kb", sys.argv))
    file = "./boutput.csv" if len(sys.argv) < 2 else sys.argv[1]
    period = None if len(sys.argv) < 3 else [datetime.strptime(i, "%Y%m%d") for i in sys.argv[2].split("-")]
    df = pd.read_csv(file, index_col=[0])
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    if period is not None:
        df = df.loc[(period[0] <= df.index) & (df.index <= period[-1])]
    else:
        period = (df[BASELINE].dropna().index.min(),)
        df = df.loc[period[0] <= df.index]
    for i in df.columns:
        if "cp" not in i:
            df[i + "_cp"] = np.cumsum(np.log10(df[i]))
    df["cur_max"] = [df["returns_cp"][:n+1].max() for n, _ in enumerate(df["returns_cp"].values)]
    df["uw"] = (10 ** df["returns_cp"] / 10 ** df["cur_max"]) - 1
    df["ro"] = df["returns"] - 1
    df["outperformance"] = (df["returns"] / df[BASELINE]) - 1

    cumulative_plot = plt.subplot2grid((4, 4), (0, 0), rowspan=2, colspan=4)
    df[filter(lambda x: "cp" in x, df.columns)].plot(ax=cumulative_plot)#.set_yscale("log")
    cumulative_plot.axhline(0, color="black", linewidth=1)
    cumulative_plot.fill_between(df.index[df["returns_cp"] >= 0], 0, df["returns_cp"][df["returns_cp"] >= 0], alpha=.2, color="green")
    cumulative_plot.fill_between(df.index[df["returns_cp"] <= 0], 0, df["returns_cp"][df["returns_cp"] <= 0], alpha=.2, color="red")
    cumulative_plot.set_title("Cumulative Return")

    underwater_plot = plt.subplot2grid((4, 4), (2, 0), rowspan=1, colspan=4)
    df["uw"].plot(ax=underwater_plot, linewidth=.5)
    underwater_plot.axhline(0, color="black")
    underwater_plot.fill_between(df.index, 0, df["uw"], alpha=.2)
    underwater_plot.set_ylim(-1.01, 0)
    underwater_plot.set_title("Underwater Plot")

    time_outperformance_plot = plt.subplot2grid((4, 4), (3, 0), rowspan=1)
    df[["outperformance"]].plot(ax=time_outperformance_plot, linewidth=.75)
    time_outperformance_plot.axhline(0, color="black", linewidth=1)
    time_outperformance_plot.set_title("Outperformance over Time")

    returns_plot = plt.subplot2grid((4, 4), (3, 1), rowspan=4)
    r_bins = None if kill_bins else [(i/50) - .5 for i in range(51)]
    df["ro"][df["ro"] > 0].hist(bins=r_bins, ax=returns_plot, color="green")
    df["ro"][df["ro"] <= 0].hist(bins=r_bins, ax=returns_plot, color="red")
    returns_plot.set_title("Returns")

    outperformance_plot = plt.subplot2grid((4, 4), (3, 2), rowspan=1)
    op_bins = None if kill_bins else [(i/50) - .5 for i in range(51)]
    df["outperformance"][df["outperformance"] > 0].hist(bins=op_bins, ax=outperformance_plot, color="green")
    df["outperformance"][df["outperformance"] <= 0].hist(bins=op_bins, ax=outperformance_plot, color="red")
    outperformance_plot.set_title("Performance over Baseline")

    pvb_plot = plt.subplot2grid((4, 4), (3, 3), rowspan=1)
    df.plot.scatter(BASELINE, "returns", c=pd.Series(["green", "red"])[(df[BASELINE] > df["returns"]).astype(int)], ax=pvb_plot, s=1)
    pvb_plot.axline((1, 1), slope=1, color="black")
    pvb_plot.set_title("Performance vs Baseline")

    plt.tight_layout()
    plt.subplots_adjust(wspace=.25, hspace=.6)
    plt.show()