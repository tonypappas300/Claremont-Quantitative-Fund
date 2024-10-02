import matplotlib.pyplot as plt
from functools import partial

import pandas as pd

from delegate_apply import delegate_apply, shared_df

def get_XYZ(df, q, n, _id):
    _d = df[(df["window_size"] == _id[0]) & (df["hold_time"] == _id[1])]
    q.put((n, (_d["window_size"].values[0], ["hold_time"].values[0], _d["returns"].mean())))

if __name__ == "__main__":
    df = pd.read_csv("./hpoutput.csv")
    #df["id"] = df["window_size"].astype(str) + "|" + df["hold_time"].astype(str)
    unique_ids = [[str(n) for n in i.split("|")] for i in (df["window_size"].astype(str) + "|" + df["hold_time"].astype(str)).unique().tolist()]
    #df.drop("id")
    df = shared_df(df)
    X, Y, Z = zip(*delegate_apply(partial(get_XYZ, df), unique_ids))
    #X = []
    #Y = []
    #Z = []
    #for i in tqdm(unique_ids):
    #    _d = df[df["id"] == i]
    #    X.append(_d["window_size"].values[0])
    #    Y.append(_d["hold_time"].values[0])
    #    Z.append(_d["returns"].mean())

    ax = plt.figure().add_subplot(projection='3d')

    # Plot the 3D surface
    ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8, alpha=0.3)

    # Plot projections of the contours for each dimension.  By choosing offsets
    # that match the appropriate axes limits, the projected contours will sit on
    # the 'walls' of the graph
    ax.contourf(X, Y, Z, zdir='z', offset=-100, cmap='coolwarm')
    ax.contourf(X, Y, Z, zdir='x', offset=-40, cmap='coolwarm')
    ax.contourf(X, Y, Z, zdir='y', offset=40, cmap='coolwarm')

    ax.set(xlim=(-40, 40), ylim=(-40, 40), zlim=(-100, 100),
        xlabel='X', ylabel='Y', zlabel='Z')

    plt.show()