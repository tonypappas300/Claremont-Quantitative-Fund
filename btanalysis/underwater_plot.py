import sys
from datetime import datetime
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

if __name__ == "__main__":
    file = "./boutput.csv" if len(sys.argv) < 2 else sys.argv[1]
    period = None if len(sys.argv) < 3 else [datetime.strptime(i, "%Y%m%d") for i in sys.argv[2].split("-")]
    df = pd.read_csv(file, index_col=[0])
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    if period is not None:
        df = df.loc[(period[0] <= df.index) & (df.index <= period[-1])]
        if "returns" in df.columns:
            df["returns_cp"] = np.cumprod(df["returns"])
    if "returns_cp" not in df.columns:
        df["returns_cp"] = np.cumprod(df["returns"])
    df["cur_max"] = [df["returns_cp"][:n+1].max() for n, _ in enumerate(df["returns_cp"].values)]
    df["uw"] = (df["returns_cp"] / df["cur_max"]) - 1
    df["uw"].plot(linewidth=.5)
    plt.axhline(0, color="black")
    plt.fill_between(df.index, 0, df["uw"], alpha=.2)
    plt.ylim(-1.01, 0)
    plt.show()