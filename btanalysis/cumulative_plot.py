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
        if "baseline" in df.columns:
            df["baseline_cp"] = np.cumprod(df["baseline"])
    cols = []
    if "returns_cp" in df.columns:
        cols.append("returns_cp")
    elif "returns" in df.columns:
        df["returns_cp"] = np.cumprod(df["returns"])
        cols.append("returns_cp")
    if "baseline_cp" in df.columns:
        cols.append("baseline_cp")
    elif "basline" in df.columns:
        df["baseline_cp"] = np.cumprod(df["baseline"])
        cols.append("baseline_cp")
    df[cols].plot().set_yscale("log")
    plt.show()