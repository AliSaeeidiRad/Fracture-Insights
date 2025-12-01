import os
import glob

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from pc import Curve


def plot_figure_density(curves_dir):
    num_curves = 10
    list_files = glob.glob(os.path.join(curves_dir, "*.txt"))[:num_curves]

    curves_list = [Curve(curve).curve().to_numpy() for curve in list_files]

    total_points = np.sum([c.shape[0] for c in curves_list])
    print(f"Total points in all curves: {total_points}")

    curves = np.concatenate(curves_list)
    print(f"Concatenated curves shape: {curves.shape}")

    kde = gaussian_kde(curves.T)

    x = np.linspace(curves[:, 0].min(), curves[:, 0].max(), 100)
    y = np.linspace(curves[:, 1].min(), curves[:, 1].max(), 100)
    z = np.linspace(curves[:, 2].min(), curves[:, 2].max(), 100)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    positions = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()])
    density = kde(positions)
    density = density.reshape(100, 100, 100)

    fig = plt.figure(dpi=1200)

    ax = fig.add_subplot(111, projection="3d")

    for curve_i in curves_list:
        ax.plot(curve_i[:, 0], curve_i[:, 1], curve_i[:, 2], c="b", linewidth=0.7)

    slices = [0, 50, -1]
    for sl in slices:
        kde_slice = density[:, :, sl]
        mappable = ax.contourf(
            xx[:, :, sl],
            yy[:, :, sl],
            kde_slice,
            zdir="z",
            offset=z[sl],
            cmap="viridis",
            alpha=0.5,
        )
    cbar = fig.colorbar(mappable, ax=ax, shrink=1.0)
    cbar.set_label("Density")
    cbar.set_ticks([])
    cbar.ax.text(
        2.0,
        0,
        "low",
        transform=cbar.ax.transAxes,
        va="center",
        ha="center",
        fontsize=10,
        color="black",
    )
    cbar.ax.text(
        2.0,
        1,
        "high",
        transform=cbar.ax.transAxes,
        va="center",
        ha="center",
        fontsize=10,
        color="black",
    )

    ax.text(
        x.max() + 12,
        y.max() - 37,
        z[0],
        "First Slice",
        color="black",
        fontsize=12,
        ha="center",
        va="bottom",
    )
    ax.text(
        x.max() + 13,
        y.max() - 37,
        z[50],
        "Middle Slice",
        color="black",
        fontsize=12,
        ha="center",
        va="bottom",
    )
    ax.text(
        x.max() + 12,
        y.max() - 37,
        z[-1],
        "Last Slice",
        color="black",
        fontsize=12,
        ha="center",
        va="bottom",
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.view_init(azim=50, elev=30)

    fig.savefig("figure_density.tif", format="tif")
