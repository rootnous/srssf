# =============================================================================
# SRSF Supplementary Plotting Script -- Per-Class F1 Heatmap and SHAP
# Feature-Importance Heatmap
# =============================================================================
#
# This file accompanies the manuscript submitted, and the main
# SRSF benchmark pipeline (see the companion training/evaluation script).
# It generates two of the qualitative diagnostic figures reported in the
# paper's Results section, directly from the CSV artifacts produced by the
# main pipeline:
#
#   1. Per-class F1-score heatmap (function `plot_perclass_f1_heatmap`):
#        Consolidates mean per-class F1-score (%) for all fifteen compared
#        methods across all four benchmark datasets (Indian Pines, Pavia
#        University, Salinas, KSC) into a single compact 2x2 grid of
#        heatmaps, with actual class names (not bare numeric IDs) on the
#        y-axis and method names on the x-axis. Expects the schema produced
#        by `run_per_class()` / `results_per_class.csv` in the main
#        pipeline (columns: Dataset, Run, Method, Class, TestSupport,
#        Accuracy, Precision, Recall, F1).
#
#   2. SHAP feature-importance heatmap (bottom "# 2. SHAP HEATMAP" section):
#        Aggregates the mean absolute SHAP value per feature, per method,
#        and per dataset from the `shap_summary_<Dataset>_main.csv` files
#        written by `save_shap_plot()` / `aggregate_explainability()` in
#        the main pipeline, and renders them as a single combined heatmap
#        via seaborn.
#
# ------------------------------------------------------------------------
# Inputs expected (produced by the main SRSF pipeline)
# ------------------------------------------------------------------------
#   results_per_class.csv                     (per-class metrics, all runs)
#   shap_summary_IndianPines_main.csv
#   shap_summary_Salinas_main.csv
#   shap_summary_PaviaU_main.csv
#   shap_summary_KSC_main.csv
# 
#
# ------------------------------------------------------------------------
# Requirements
# ------------------------------------------------------------------------
# Python >= 3.9 with: numpy, pandas, matplotlib, seaborn
#
# ------------------------------------------------------------------------
# Usage
# ------------------------------------------------------------------------
# Standalone (per-class heatmap only):
#   python <this_file>.py
#   (equivalently: python plot_perclass_heatmap.py results_per_class.csv
#    perclass_f1_heatmap.pdf, matching the docstring embedded below)
#
# The SHAP heatmap section at the bottom of this file is written as a
# sequential script block (not wrapped in a function) and executes
# immediately when this file is run, reading the four
# `shap_summary_*_main.csv` files from the current working directory and
# writing `shap_heatmap.pdf`.
#
# ------------------------------------------------------------------------
# Outputs
# ------------------------------------------------------------------------
#   perclass_f1_heatmap.pdf   (Section 1: per-class F1 heatmap, 2x2 grid)
#   shap_heatmap.pdf          (Section 2: aggregated SHAP feature heatmap)
#
# ------------------------------------------------------------------------
# License and citation
# ------------------------------------------------------------------------
# Released for academic/research use alongside the associated 
# manuscript and its companion main pipeline script. If you use this code,
# please cite the paper (see the manuscript for the full citation) and
# this repository. 
# The License is on the GitHub repo.
# =============================================================================

"""
1. per-class F1-score heatmap.

y-axis now shows actual class names (e.g., "Corn-notill",
"Meadows") instead of bare numeric class IDs, using the same CLASS_NAMES
mapping already defined in the main pipeline file. Colorbar remains in its
own reserved GridSpec column so it never overlaps the heatmap panels.

Produces ONE figure: a 2x2 grid of small heatmaps (one per dataset),
each showing mean per-class F1-score (%) across accepted seeds, with
class NAMES on the y-axis and methods on the x-axis.

Standalone usage:
    python plot_perclass_heatmap.py results_per_class.csv perclass_f1_heatmap.pdf
"""


import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

ALL_DATASETS = ["IndianPines", "PaviaU", "Salinas", "KSC"]

METHOD_ORDER = [
    "Raw+SVM", "PCA+SVM", "KPCA+SVM", "ICA+SVM", "SpaBS+SVM", "MVPCA+SVM",
    "EMP+SVM", "EAP+SVM", "Gabor+SVM", "FastPCA+RF", "EMP+RF", "SVM+RF",
    "1D-CNN", "SRSF(Ours)", "SRSF-SelectedInput",
]

METHOD_ABBR = {
    "Raw+SVM": "Raw", "PCA+SVM": "PCA", "KPCA+SVM": "KPCA", "ICA+SVM": "ICA",
    "SpaBS+SVM": "SpaBS", "MVPCA+SVM": "MVPCA", "EMP+SVM": "EMP",
    "EAP+SVM": "EAP", "Gabor+SVM": "Gabor", "FastPCA+RF": "FPCA-RF",
    "EMP+RF": "EMP-RF", "SVM+RF": "SVM-RF", "1D-CNN": "1D-CNN",
    "SRSF(Ours)": "SRSF", "SRSF-SelectedInput": "SRSF-SI",
}

# Same class-name mapping already defined in the main pipeline file
# (1d_leakage_free_seed_filtered.py). Kept identical here so the standalone
# script produces labels consistent with the embedded version.
CLASS_NAMES = {
    "IndianPines": {
        1: "Alfalfa", 2: "Corn-notill", 3: "Corn-mintill", 4: "Corn",
        5: "Grass-pasture", 6: "Grass-trees", 7: "Grass-pasture-mowed",
        8: "Hay-windrowed", 9: "Oats", 10: "Soybean-notill",
        11: "Soybean-mintill", 12: "Soybean-clean", 13: "Wheat",
        14: "Woods", 15: "Bldg-Grass-Tree-Drives", 16: "Stone-Steel-Towers",
    },
    "PaviaU": {
        1: "Asphalt", 2: "Meadows", 3: "Gravel", 4: "Trees",
        5: "Painted metal sheets", 6: "Bare Soil", 7: "Bitumen",
        8: "Self-Blocking Bricks", 9: "Shadows",
    },
    "Salinas": {
        1: "Brocoli_green_weeds_1", 2: "Brocoli_green_weeds_2", 3: "Fallow",
        4: "Fallow_rough_plow", 5: "Fallow_smooth", 6: "Stubble", 7: "Celery",
        8: "Grapes_untrained", 9: "Soil_vinyard_develop",
        10: "Corn_senesced_green_weeds", 11: "Lettuce_romaine_4wk",
        12: "Lettuce_romaine_5wk", 13: "Lettuce_romaine_6wk",
        14: "Lettuce_romaine_7wk", 15: "Vinyard_untrained",
        16: "Vinyard_vertical_trellis",
    },
    "KSC": {
        1: "Scrub", 2: "Willow swamp", 3: "CP hammock", 4: "Slash pine",
        5: "Oak/Broadleaf", 6: "Hardwood", 7: "Swamp", 8: "Graminoid marsh",
        9: "Spartina marsh", 10: "Cattail marsh", 11: "Salt marsh",
        12: "Mud flats", 13: "Water",
    },
}


def _class_label(dname, class_id, max_len=18):
    """Return a readable, length-capped class name for axis labels."""
    name = CLASS_NAMES.get(dname, {}).get(int(class_id), "Class %d" % int(class_id))
    if len(name) > max_len:
        name = name[:max_len - 1] + "\u2026"  # ellipsis
    return name


def plot_perclass_f1_heatmap(df, out_path="perclass_f1_heatmap.pdf",
                              datasets=ALL_DATASETS,
                              method_order=METHOD_ORDER,
                              metric="F1", figsize=(8.4, 6.8)):
    """
    df: DataFrame with columns [Dataset, Run, Method, Class, F1, ...]
        (exactly the schema produced by run_per_class() /
        results_per_class.csv).
    Saves one compact 2x2-panel heatmap figure to out_path and returns
    the path. Y-axis ticks show actual class names (via CLASS_NAMES)
    instead of bare numeric IDs. Colorbar sits in its own reserved
    GridSpec column so it never overlaps any heatmap panel.
    """
    methods = [m for m in method_order if m in df["Method"].unique()]
    agg = (df.groupby(["Dataset", "Method", "Class"])[metric]
             .mean()
             .reset_index())

    present = [d for d in datasets if d in agg["Dataset"].unique()]
    if not present:
        return None

    n_cols = 2
    n_rows = int(np.ceil(len(present) / n_cols))
    norm = Normalize(vmin=0, vmax=100)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        n_rows, n_cols + 1,
        width_ratios=[1.0] * n_cols + [0.05],
        wspace=0.75, hspace=0.65,
        left=0.16, right=0.88, top=0.90, bottom=0.14,
        figure=fig,
    )

    axes = []
    im = None
    for idx, dname in enumerate(present):
        r, c = divmod(idx, n_cols)
        ax = fig.add_subplot(gs[r, c])
        axes.append(ax)

        sub = agg[agg["Dataset"] == dname]
        classes = sorted(sub["Class"].unique())
        mat = np.full((len(classes), len(methods)), np.nan)
        for j, m in enumerate(methods):
            for i, cls in enumerate(classes):
                val = sub[(sub["Method"] == m) & (sub["Class"] == cls)][metric]
                if len(val):
                    mat[i, j] = val.values[0]

        im = ax.imshow(mat, aspect="auto", cmap="viridis", norm=norm)
        ax.set_title(dname, fontsize=8, pad=2)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(
            [_class_label(dname, cls) for cls in classes],
            fontsize=4.8,
        )
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([METHOD_ABBR.get(m, m) for m in methods],
                            fontsize=5, rotation=90)
        ax.tick_params(length=2, pad=1)
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)

    cax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Mean F1-score (%)", fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    fig.text(0.5, 0.03, "Method", ha="center", fontsize=7)
    fig.suptitle("Per-class F1-score across methods (mean over accepted seeds)",
                 fontsize=8, y=0.985)

    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


if __name__ == "__main__":

    csv_path = "/content/results_per_class.csv"

    out_path = "perclass_f1_heatmap.pdf"
    df = pd.read_csv(csv_path)
    saved = plot_perclass_f1_heatmap(df, out_path=out_path)
    print("Saved:", saved)


# 2. SHAP HEATMAP

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load CSVs for each dataset
indian = pd.read_csv("shap_summary_IndianPines_main.csv")
salinas = pd.read_csv("shap_summary_Salinas_main.csv")
pavia = pd.read_csv("shap_summary_PaviaU_main.csv")
ksc = pd.read_csv("shap_summary_KSC_main.csv")

# Add dataset labels
indian["dataset"] = "Indian Pines"
salinas["dataset"] = "Salinas"
pavia["dataset"] = "Pavia"
ksc["dataset"] = "KSC"

# Combine
df = pd.concat([indian, salinas, pavia, ksc])

# Example: heatmap of mean SHAP per feature across datasets
pivot = df.pivot_table(index="feature", columns=["dataset","method"], values="mean_abs_shap")
sns.heatmap(pivot, cmap="viridis")
plt.title("Feature importance across HSI datasets")
plt.savefig("shap_heatmap.pdf", dpi=300, bbox_inches="tight")  # PDF for publication
plt.show()



"""# New Section"""