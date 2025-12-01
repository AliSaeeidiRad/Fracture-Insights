import os
import glob

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from curve import Curve
from analysis import Procrustes

if __name__ == "__main__":
    output_dir = "Output\Cluster_Result_*"

    for result in glob.glob(output_dir):
        if os.path.isdir(result):
            all_curves = []
            for cluster in glob.glob(os.path.join(result, "*")):
                _curves = [{f"curve" : Curve(curve).curve().to_numpy(), "cluster" : cluster.split("_")[-1]} for curve in glob.glob(os.path.join(cluster, "*.txt"))]
                all_curves += _curves
            if "procrustes" in result: alpha = 1.00
            else: alpha = 0.00
            if "frechet" in result: beta = 1.00
            else: beta = 0.00
            pr = Procrustes([item["curve"] for item in all_curves])
            pr.analysis(alpha=alpha, beta=beta)
            matrix = pr.similarity_matrix
            matrix = (matrix - np.min(matrix))/(np.max(matrix) - np.min(matrix))
            fig = plt.figure(dpi=300, figsize=(10, 10))
            ax = fig.add_subplot(1, 1, 1)
            sns.heatmap(
                matrix, 
                annot=False, 
                fmt=".2f", 
                cmap="YlGnBu", 
                cbar_kws={'label': 'Similarity'},
                xticklabels=[int(item["cluster"]) for item in all_curves],
                yticklabels=[int(item["cluster"]) for item in all_curves],
                vmin=np.min(matrix), 
                vmax=np.max(matrix),
                ax=ax,
                square=True,
            )
            for i in range(pr.t):
                print(f"Group {i+1} : ")
            fig.savefig(f"{result.split(os.sep)[-1]}.tiff", format="tiff")
        else:
            pass