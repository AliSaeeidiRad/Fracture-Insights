import os
import glob

import matplotlib.pyplot as plt

from pc import Curve


def plot_figure_curves(curves_dir):
    list_files = glob.glob(os.path.join(curves_dir, "*.txt"))

    curves_list = [Curve(curve).curve().to_numpy() for curve in list_files]

    fig = plt.figure(dpi=500)
    
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    
    for i in range(len(curves_list)):
        ax.plot(curves_list[i][:, 0], curves_list[i][:, 1], curves_list[i][:, 2])
    
    ax.view_init(azim=70, elev=20)
    
    fig.savefig("figure_curves.pdf", format="pdf")