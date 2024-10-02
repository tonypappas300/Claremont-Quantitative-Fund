import matplotlib.pyplot as plt
from sys import argv

import pandas as pd
import numpy as np

if __name__ == "__main__":
    MEASURE = ("Cumulative", lambda r, h, w: (10 ** np.mean(np.log10(np.power(np.mean(np.reshape(r if len(r)%h.values[0] == 0 else r[:-(len(r)%h.values[0])], (-1, h.values[0])), axis=-1), 1/h.values[0])))) ** 365)#np.sum(np.log10(r)))#lambda x: x.mean()
    MIN_CUT = .95
    X = []
    Y = []
    _Z = []
    Z = []
    df = pd.read_csv("./hpoutput.csv" if len(argv) < 2 else argv[1]).dropna()
    df["an_returns"] = df["returns"] ** (365/df["hold_time"])
    unique_window = df["window_size"].unique()
    unique_hold = df["hold_time"].unique()
    wdfs = {i: df[df["window_size"] == i] for i in unique_window}
    for i, _df in wdfs.items():
        for h in _df["hold_time"].unique():
            _d = _df[_df["hold_time"] == h]
            X.append(_d["window_size"].values[0])
            Y.append(_d["hold_time"].values[0])
            _Z.append(MEASURE[1](_d["returns"], _d["hold_time"], _d["window_size"]))
            Z.append(MEASURE[1](_d["returns"]/_d["baseline"], _d["hold_time"], _d["window_size"]))
    X = np.array(X)
    Y = np.array(Y)
    _Z = np.array(_Z)
    Z = np.array(Z)

    #ax = plt.figure().add_subplot(projection='3d')
    #ax.plot_trisurf(X, Y, Z, cmap="coolwarm")
    #ax.set(xlabel='Window Size', ylabel='Hold Time', zlabel='Return Over Baseline')
    #plt.show()

    xi = np.sort(np.unique(X))
    yi = np.sort(np.unique(Y))
    zi = []
    _zi = []
    xi, yi = np.meshgrid(xi, yi)
    for _x, _y in zip(np.reshape(xi, (-1,)), np.reshape(yi, (-1,))):
        idx = (X == _x) & (Y == _y)
        sidx = np.sum(idx)
        if sidx == 0:
            zi.append(1.0)
            _zi.append(1.0)
        elif sidx == 1:
            zi.append(MIN_CUT if (c := np.mean(Z[idx])) < MIN_CUT else c)
            _zi.append(MIN_CUT if (c := np.mean(_Z[idx])) < MIN_CUT else c)
        else:
            print(sidx, idx)
            zi.append(1.0)
            _zi.append(1.0)
    zi = np.array(zi)
    zi = np.reshape(zi, np.shape(xi))
    _zi = np.array(_zi)
    _zi = np.reshape(_zi, np.shape(xi))

    ax = plt.figure().add_subplot(projection='3d')

    ax.plot_surface(xi, yi, zi, cmap="coolwarm")

    ax.contourf(xi, yi, zi, zdir='z', offset=np.min(zi) - .025, cmap='coolwarm')
    ax.contourf(xi, yi, zi, zdir='x', offset=np.min(X) - 75, cmap='coolwarm')
    ax.contourf(xi, yi, zi, zdir='y', offset=np.max(Y) + 25, cmap='coolwarm')

    ax.set(xlabel='Window Size', ylabel='Hold Time', zlabel='Return Over Baseline', xlim=(np.min(X) - 75, np.max(X) + 75), ylim=(np.min(Y) - 25, np.max(Y) + 25), zlim=(np.min(zi) - .025, np.max(Z) + .025))

    plt.show()

    ax = plt.figure().add_subplot(projection='3d')

    ax.plot_surface(xi, yi, _zi, cmap="coolwarm")

    ax.contourf(xi, yi, _zi, zdir='z', offset=np.min(_zi) - .025, cmap='coolwarm')
    ax.contourf(xi, yi, _zi, zdir='x', offset=np.min(X) - 75, cmap='coolwarm')
    ax.contourf(xi, yi, _zi, zdir='y', offset=np.max(Y) + 25, cmap='coolwarm')

    ax.set(xlabel='Window Size', ylabel='Hold Time', zlabel= ('Average' if MEASURE[0] == "mean" else MEASURE[0].title()) + ' Return', xlim=(np.min(X) - 75, np.max(X) + 75), ylim=(np.min(Y) - 25, np.max(Y) + 25), zlim=(np.min(_zi) - .025, np.max(_Z) + .025))

    plt.show()