import sys
from datetime import datetime
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

if __name__ == "__main__":
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
    df["ro"] = df["returns"] - 1
    df.hist("ro", bins=None if kill_bins else [(i/20) - .5 for i in range(21)])
    plt.show()
    df["outperformance"] = (df["returns"] / df["baseline"]) - 1
    df.hist("outperformance", bins=None if kill_bins else [(i/20) - .5 for i in range(21)])
    plt.show()
    df[["outperformance"]].plot()
    plt.axhline(0, color="black")
    plt.show()
    df.plot.scatter("baseline", "returns", c=pd.Series(["green", "red"])[(df["baseline"] > df["returns"]).astype(int)])
    plt.axline((1, 1), slope=1, color="black") 
    plt.show()