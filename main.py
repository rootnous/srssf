# =============================================================================
# SRSF: Stability-Reliability Spectral-Spatial Fusion for Hyperspectral Image
# Classification -- Main Benchmark and Training/Evaluation Pipeline
# =============================================================================
#
# This file accompanies the manuscript submitted. It implements
# the complete experimental pipeline used to produce every quantitative and
# qualitative result reported in the paper, including:
#
#   - Strict, footprint-level spatially disjoint train/validation/test
#     splitting with automated leakage verification (zero pixel overlap
#     between partitions is measured and logged, not merely assumed).
#   - Geometry-only, label-only seed acceptance ("seed filtering"): ten
#     accepted random seeds per dataset are selected using split geometry
#     alone (never model accuracy or test predictions) before any model is
#     trained, to guarantee leakage-free and statistically meaningful
#     reporting (mean +/- std over the accepted seeds).
#   - Thirteen classical and deep-learning baselines (Raw+SVM, PCA+SVM,
#     KPCA+SVM via Nystroem RBF approximation, ICA+SVM, SpaBS+SVM,
#     MVPCA+SVM, EMP+SVM, EAP+SVM, Gabor+SVM, FastPCA+RF, EMP+RF, SVM+RF,
#     and a lightweight 1D-CNN spectral baseline).
#   - The proposed SRSF framework in two variants:
#       SRSF(Ours): hybrid design in which a compact, bootstrap
#           stability-selected spectral expert is fused with full-spectrum
#           spatial and joint experts (EMP, EAP, Gabor, joint SVM, joint
#           RF, FastPCA-RF) via validation-derived reliability weights.
#       SRSF-SelectedInput: an ablation in which every expert (not only the
#           spectral expert) is built from the same stability-selected band
#           subset instead of the full spectrum.
#   - A shared, validation-only band-count selection procedure ensuring the
#     test partition is never used to choose the number of spectral bands.
#   - Six-configuration ablation study (RawSVM, spectral-only-with-
#     selection, fusion-without-band-selection, uniform-weight fusion,
#     SRSF(Ours), SRSF-SelectedInput).
#   - Few-shot sensitivity analysis (5/10/15/20/25 labeled samples/class).
#   - Computational-efficiency benchmarking (fit time, inference time, peak
#     memory) for every method.
#   - Model-agnostic (SHAP) and model-specific (band-stability curves,
#     expert-reliability weights) interpretability artifacts.
#   - Publication-style diagnostic and figure generation (split maps,
#     per-dataset classification panels, leakage diagnostics).
#
# ------------------------------------------------------------------------
# Datasets
# ------------------------------------------------------------------------
# Four public benchmark hyperspectral scenes are used: Indian Pines, Pavia
# University (PaviaU), Salinas, and the Kennedy Space Center (KSC) dataset.
# This script downloads the datasets from the Kaggle handles configured in
# KAGGLE_SOURCES via the `kagglehub` package; a Kaggle account/API token
# (or a locally cached copy at the same handle paths) is required. No
# hyperspectral data is redistributed with this repository.
#
# ------------------------------------------------------------------------
# Requirements
# ------------------------------------------------------------------------
# Python >= 3.9 with:
#   numpy, pandas, scipy, scikit-learn, matplotlib, kagglehub
# Optional (used with graceful fallback if unavailable; the script logs a
# warning and substitutes an alternative rather than failing):
#   torch                (1D-CNN baseline; falls back to Raw+SVM if absent)
#   scikit-image         (EMP morphological profile; falls back to a mean
#                          filter approximation if absent)
#   shap                 (Shapley-value feature attribution; explainability
#                          artifacts are skipped, not fabricated, if absent)
#
# ------------------------------------------------------------------------
# Usage
# ------------------------------------------------------------------------
#   python <this_file>.py
#
# Running the file as __main__ performs, in order: (1) geometry-only seed
# selection per dataset (TARGET_RUNS accepted seeds, audited in
# seed_selection_audit.csv / accepted_seeds.csv), (2) split-diagnostic and
# split-map generation, and (3) the main comparison over the accepted
# seeds. The per-class, ablation, few-shot, efficiency, and classification-
# map experiments are also implemented as standalone functions
# (run_per_class, run_ablation, run_fewshot, run_efficiency_benchmark,
# run_classification_maps) and can be invoked individually; see the
# `if __name__ == "__main__":` block at the end of this file for the exact
# call sequence used to produce the paper's results. All numeric outputs
# (CSVs) and figures (PDFs) are written under OUT_DIR
# ("outputs_srsf_all_methods/") with fixed, self-describing filenames.
#
# ------------------------------------------------------------------------
# Reproducibility notes
# ------------------------------------------------------------------------
#   - A fixed global NumPy seed (42) is set once at import time; every
#     stochastic component thereafter (splits, bootstrap resampling,
#     classifier/CNN initialization) is separately seeded per accepted run
#     seed, so results are exactly reproducible given the same environment.
#   - All BLAS/OpenMP thread-pool environment variables are pinned to 1
#     thread before any numerical library is imported, to avoid
#     nondeterministic multi-threaded floating-point reductions.
#   - Seed acceptance (see MIN_STRICT_TEST_CENTERS, TARGET_RUNS,
#     MAX_CANDIDATE_SEEDS) uses only split geometry and ground-truth
#     labels; it never inspects model predictions, accuracy, or any other
#     test-set-derived quantity, so it cannot bias reported performance.
#
# ------------------------------------------------------------------------
# License and citation
# ------------------------------------------------------------------------
# Released for academic/research use alongside the associated 
# manuscript. If you use this code, please cite the paper (see the
# manuscript for the full citation) and this repository. The license is on the GitHub 
#repo.
#
# 
# =============================================================================

"""SRSF-only HSI benchmark with shared validation-based band-count selection
and a 1D-CNN spectral deep-learning baseline.

Methods in main, per-class, few-shot, ablation, efficiency, and maps:
Raw+SVM, PCA+SVM, KPCA+SVM (Nystroem RBF approximation), ICA+SVM,
SpaBS+SVM, MVPCA+SVM, EMP+SVM, EAP+SVM, Gabor+SVM, FastPCA+RF,
EMP+RF, SVM+RF, 1D-CNN, SRSF(Ours), and SRSF-SelectedInput.

SRSF(Ours): hybrid design -- compact stability-selected spectral
            expert fused with full-spectrum spatial/joint experts.
SRSF-SelectedInput: every expert (spectral, EMP, EAP, Gabor, joint, RF,
            FastPCA) is built from the SAME stability-selected
            band subset instead of the full spectrum.
1D-CNN: a lightweight spectral-only 1D convolutional network
            (PyTorch) trained directly on the raw per-pixel
            spectrum, included as a deep-learning baseline
            alongside the classical spectral/spatial methods.
            Falls back gracefully (skipped, logged) if PyTorch is
            not installed in the execution environment.

Automatic band-count selection:
For each dataset/seed, one shared validation sweep over
BAND_COUNT_CANDIDATES = [15, 30, 45, 60, 75] is performed using the
hybrid architecture as a neutral reference. The smallest k within
OA_TIE_TOLERANCE = 0.25 percentage points of the best validation OA
is selected. This same k is then used by both SRSF variants and all
related ablation controls. The test set is never used during selection.

Both SRSF variants and 1D-CNN are fit and evaluated together inside every
experiment (main comparison, per-class, few-shot, efficiency, maps), so a
single run produces all of them as extra rows/columns in the same result
tables.

---- LEAKAGE-FREE SPATIAL SPLIT (this version) ----
get_pixel_disjoint_split() now guarantees, whenever geometrically feasible:
  - train and validation patch footprints never overlap each other
    (this was already enforced in prior versions), AND
  - test candidates are accepted only if their ENTIRE patch footprint is
    free of both the train_lock and val_lock masks (not just their
    center pixel, which was the previous bug allowing spatial leakage).
  - Test patches ARE allowed to overlap each other; this is intentional
    and does not constitute leakage.

If the strict, fully footprint-disjoint test region would be empty at the
requested train/val density, the function automatically retries with
progressively smaller train/val allocations (halving up to
`max_shrink_attempts` times) before falling back to a relaxed,
explicitly-logged and explicitly-flagged selection. The returned tuple
now includes a `strict` boolean so every caller and every diagnostic can
tell, per dataset/seed, whether the split is provably leakage-free.

---- SEED FILTERING FOR PUBLICATION RUNS ----
The code now supports selecting exactly TARGET_RUNS acceptable seeds per
dataset, where acceptability is defined by:
  1. strict=True (zero test/train/val footprint overlap),
  2. all measured footprint overlaps exactly zero,
  3. test-center count >= MIN_STRICT_TEST_CENTERS[dname].

Seed selection uses only split geometry and labels; no model accuracy,
test predictions, or test-based tuning is involved. The full candidate
audit is saved to seed_selection_audit.csv, and accepted seeds are
saved to accepted_seeds.csv.


"""


import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import time
import tracemalloc
import warnings
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from scipy.signal import fftconvolve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import FastICA, PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import f_classif
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

try:
    from skimage.morphology import disk, opening, closing
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False

try:
    import kagglehub
except Exception:
    kagglehub = None

try:
    import shap
except Exception:
    shap = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

warnings.filterwarnings("ignore")
np.random.seed(42)

OUT_DIR = "outputs_srsf_all_methods"
EXPLAIN_DIR = os.path.join(OUT_DIR, "explainability")
MAP_DIR = os.path.join(OUT_DIR, "maps")
os.makedirs(EXPLAIN_DIR, exist_ok=True)
os.makedirs(MAP_DIR, exist_ok=True)

ALL_DATASETS = ["IndianPines", "PaviaU", "Salinas", "KSC"]
DEFAULT_N_RUNS = 10
MAX_TEST_EVAL_SAMPLES = None
RF_MAX_DEPTH = 25
RF_TREES = 200
SHAP_BACKGROUND_SIZE = 20
SHAP_EXPLAIN_SIZE = 50

# ---- 1D-CNN hyperparameters ----
CNN1D_EPOCHS = 60
CNN1D_BATCH_SIZE = 64
CNN1D_LR = 1e-3
CNN1D_WEIGHT_DECAY = 1e-4
CNN1D_SEED_JITTER = 0  # kept 0; seed already controls torch RNG per run

# ---- Automatic shared band-count selection ----
BAND_COUNT_CANDIDATES = [15, 30, 45, 60, 75]
OA_TIE_TOLERANCE = 0.25  # percentage points
_BAND_COUNT_LOG = []

METHODS = [
    "Raw+SVM", "PCA+SVM", "KPCA+SVM", "ICA+SVM", "SpaBS+SVM", "MVPCA+SVM",
    "EMP+SVM", "EAP+SVM", "Gabor+SVM", "FastPCA+RF", "EMP+RF", "SVM+RF",
    "1D-CNN",
    "SRSF(Ours)", "SRSF-SelectedInput",
]

DATASET_PATCH_SIZES = {
    "IndianPines": 5,
    "PaviaU": 7,
    "Salinas": 7,
    "KSC": 5,
}

DATASET_SPLIT_FRACTIONS = {
    # Aim for ~10% train after footprint locking; val kept small.
    "IndianPines": {"train": 0.01, "val": 0.01},
    "PaviaU": {"train": 0.01, "val": 0.01},
    "Salinas": {"train": 0.01, "val": 0.01},
    "KSC": {"train": 0.01, "val": 0.01},
}

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

KAGGLE_SOURCES = {
    "IndianPines": {"handle": "douglasfabiamartins/hsi-study-datasets", "img_key": "indian_pines_corrected", "gt_key": "indian_pines_gt", "token": "indian"},
    "PaviaU": {"handle": "sreevallimanda/pavia-university-hyperspectral", "img_key": "paviaU", "gt_key": "paviaU_gt", "token": "pavia"},
    "Salinas": {"handle": "douglasfabiamartins/hsi-study-datasets", "img_key": "salinas_corrected", "gt_key": "salinas_gt", "token": "salinas"},
    "KSC": {"handle": "sreevallimanda/ksc-hyperspectral", "img_key": "KSC", "gt_key": "KSC_gt", "token": "ksc"},
}

_DATASET_CACHE = {}
_FILTER_CACHE = {}
_T0 = time.time()

# ---- Seed filtering configuration ----
TARGET_RUNS = 10
MAX_CANDIDATE_SEEDS = 200

MIN_STRICT_TEST_CENTERS = {
    "IndianPines": 200,
    "PaviaU": 1500,
    "Salinas": 2000,
    "KSC": 100,
}


def log(message):
    print("[%.1fs] %s" % (time.time() - _T0, message), flush=True)


def find_data_file(root, required_tokens, excluded_tokens=()):
    candidates = []
    for dirpath, _, files in os.walk(root):
        for filename in files:
            low = filename.lower()
            if not low.endswith((".mat", ".npy")):
                continue
            if all(token.lower() in low for token in required_tokens) and not any(token.lower() in low for token in excluded_tokens):
                candidates.append(os.path.join(dirpath, filename))
    return sorted(candidates)[0] if candidates else None


def load_array(path, preferred_key=None):
    if path.lower().endswith(".npy"):
        return np.load(path)
    content = sio.loadmat(path)
    if preferred_key in content:
        return content[preferred_key]
    keys = [key for key in content if not key.startswith("__")]
    if not keys:
        raise RuntimeError("No usable array in %s" % path)
    return max((content[key] for key in keys), key=lambda value: value.size)


def load_dataset(name):
    """Loads every configured dataset, including IndianPines, with explicit validation."""
    if name in _DATASET_CACHE:
        return _DATASET_CACHE[name]
    if name not in KAGGLE_SOURCES:
        raise KeyError("Unknown dataset: %s" % name)
    if kagglehub is None:
        raise ImportError("kagglehub is required. Install it with: pip install kagglehub")
    cfg = KAGGLE_SOURCES[name]
    root = kagglehub.dataset_download(cfg["handle"])
    image_path = find_data_file(root, [cfg["token"]], ("gt", "label", "ground"))
    gt_path = (find_data_file(root, [cfg["token"], "gt"]) or find_data_file(root, [cfg["token"], "label"]) or find_data_file(root, [cfg["token"], "ground"]))
    if image_path is None or gt_path is None:
        raise RuntimeError("%s files not found under %s. image=%r gt=%r" % (name, root, image_path, gt_path))
    cube = np.asarray(load_array(image_path, cfg["img_key"]), dtype=np.float32)
    gt = np.asarray(load_array(gt_path, cfg["gt_key"]), dtype=np.int32)
    if cube.ndim != 3 or gt.ndim != 2 or cube.shape[:2] != gt.shape:
        raise ValueError("%s invalid data shapes: cube=%r, gt=%r" % (name, cube.shape, gt.shape))
    cube = (cube - cube.min()) / (cube.max() - cube.min() + 1e-8)
    if np.sum(gt > 0) == 0:
        raise ValueError("%s ground truth has no labeled pixels" % name)
    _DATASET_CACHE[name] = (cube, gt)
    log("Loaded %s cube=%r gt=%r labeled=%d" % (name, cube.shape, gt.shape, int(np.sum(gt > 0))))
    return cube, gt


# ============================================================
# Footprint geometry helpers
# ============================================================

def footprint_bounds(r, c, patch_size, height, width):
    h = patch_size // 2
    r0 = max(0, r - h)
    r1 = min(height, r + h + 1)
    c0 = max(0, c - h)
    c1 = min(width, c + h + 1)
    return r0, r1, c0, c1


def footprint_is_free(mask, r, c, patch_size, height, width):
    r0, r1, c0, c1 = footprint_bounds(r, c, patch_size, height, width)
    return not mask[r0:r1, c0:c1].any()


def lock_footprint(mask, r, c, patch_size, height, width):
    r0, r1, c0, c1 = footprint_bounds(r, c, patch_size, height, width)
    mask[r0:r1, c0:c1] = True


def select_disjoint_centers(candidates, target, mask_a, mask_b, patch, height, width):
    """Select up to `target` centers whose full patch footprint is free of
    both mask_a and mask_b. Locks mask_a for every accepted center."""
    selected = []
    for r, c in candidates:
        if len(selected) >= target:
            break
        if not footprint_is_free(mask_a, r, c, patch, height, width):
            continue
        if not footprint_is_free(mask_b, r, c, patch, height, width):
            continue
        selected.append((r, c))
        lock_footprint(mask_a, r, c, patch, height, width)
    return selected


def build_footprint_mask(centers, patch_size, height, width):
    mask = np.zeros((height, width), dtype=bool)
    for r, c in centers:
        lock_footprint(mask, r, c, patch_size, height, width)
    return mask


# ============================================================
# Leakage-free pixel-disjoint split
# ============================================================

def get_pixel_disjoint_split(dname, gt, seed, n_per_class=None, val_n_per_class=None,
                              max_shrink_attempts=6):
    """
    Pixel-disjoint train/val/test split with footprint locking.

    Guarantees:
      - Train and validation patch footprints never overlap each other
        (always enforced, at every train/val density).
      - Test candidates are accepted only if their ENTIRE patch footprint
        is free of both train_lock and val_lock (strict, leakage-free),
        unless this is geometrically infeasible for the given class
        density / patch size, in which case an explicit, logged and
        flagged relaxation is used as a last resort.
      - Test patches ARE allowed to overlap each other; this is not
        leakage and is never restricted.

    If the strict test selection would be empty, the function
    automatically retries with progressively smaller train/val targets
    (scaled by 0.5 each attempt, up to `max_shrink_attempts` times)
    before falling back to a relaxed, explicitly-logged selection.

    Returns:
        train, val, test, strict
        `strict` is True if the returned split has zero footprint overlap
        between test and train/val; False if the relaxed fallback fired.
    """
    gt = np.asarray(gt, dtype=np.int32)
    height, width = gt.shape
    patch = DATASET_PATCH_SIZES[dname]

    def attempt_split(scale):
        rng = np.random.RandomState(seed)
        train_lock = np.zeros((height, width), dtype=bool)
        val_lock = np.zeros((height, width), dtype=bool)
        classes = np.unique(gt[gt > 0])
        train, val = [], []

        for class_id in classes:
            coords = [tuple(rc) for rc in np.argwhere(gt == class_id)]
            rng.shuffle(coords)

            default_train = max(1, int(round(len(coords) * DATASET_SPLIT_FRACTIONS[dname]["train"] * scale)))
            default_val = max(1, int(round(len(coords) * DATASET_SPLIT_FRACTIONS[dname]["val"] * scale)))
            train_target = int(n_per_class * scale) if n_per_class is not None else default_train
            val_target = int(val_n_per_class * scale) if val_n_per_class is not None else default_val
            train_target = max(1, train_target)
            val_target = max(1, val_target)

            # Reserve validation centers first so training cannot consume every candidate.
            class_val = select_disjoint_centers(coords, val_target, val_lock, train_lock, patch, height, width)
            remaining = [c for c in coords if c not in class_val]
            class_train = select_disjoint_centers(remaining, train_target, train_lock, val_lock, patch, height, width)

            # Fallback: ensure at least one train/val center if geometry allows.
            if not class_train and len(coords) > len(class_val):
                class_train = select_disjoint_centers(
                    [c for c in coords if c not in class_val], 1, train_lock, val_lock, patch, height, width
                )
            if not class_val and len(coords) > len(class_train):
                class_val = select_disjoint_centers(
                    [c for c in coords if c not in class_train], 1, val_lock, train_lock, patch, height, width
                )

            train.extend(class_train)
            val.extend(class_val)

        train_set, val_set = set(train), set(val)
        labeled = [tuple(rc) for rc in np.argwhere(gt > 0)]

        # Strict test selection: reject a candidate if its ENTIRE patch overlaps
        # a locked train or validation footprint, not just its center pixel.
        # Test patches are allowed to overlap each other.
        test = []
        for r, c in labeled:
            if (r, c) in train_set or (r, c) in val_set:
                continue
            if not footprint_is_free(train_lock, r, c, patch, height, width):
                continue
            if not footprint_is_free(val_lock, r, c, patch, height, width):
                continue
            test.append((r, c))

        return train, val, test

    scale = 1.0
    train, val, test = attempt_split(scale)
    strict = len(test) > 0

    attempt = 0
    while not strict and attempt < max_shrink_attempts:
        scale *= 0.5
        attempt += 1
        log(f"Split retry {dname} seed{seed}: strict test region empty at scale={scale * 2:.3f}; "
            f"retrying with train/val scaled to {scale:.3f}x (attempt {attempt}/{max_shrink_attempts}).")
        train, val, test = attempt_split(scale)
        strict = len(test) > 0

    if not strict:
        # Geometrically infeasible even at minimal train/val density: use the
        # relaxed fallback (center-only exclusion) and flag it explicitly.
        train_set, val_set = set(train), set(val)
        labeled = [tuple(rc) for rc in np.argwhere(gt > 0)]
        test = [(r, c) for (r, c) in labeled if (r, c) not in train_set and (r, c) not in val_set]
        log(f"Split warning {dname} seed{seed}: strict footprint-disjoint test region "
            f"remained empty after {max_shrink_attempts} shrink attempts; using relaxed "
            f"center-only exclusion. Some test patches may overlap train/val footprints.")

    if not train or not val:
        raise RuntimeError(f"Invalid split for {dname}: train={len(train)} val={len(val)}; both must be non-empty.")

    log(f"Split {dname} seed{seed}: train={len(train)} val={len(val)} test={len(test)} strict={strict}")
    return train, val, test, strict


def verify_split_leakage(dname, gt, train, val, test, patch_size):
    """
    Returns (is_leakage_free, overlap_counts_dict). Never raises; always
    logs the measured overlap so every run has a permanent, auditable
    record of exactly how leakage-free each split is.
    """
    height, width = gt.shape
    train_mask = build_footprint_mask(train, patch_size, height, width)
    val_mask = build_footprint_mask(val, patch_size, height, width)
    test_mask = build_footprint_mask(test, patch_size, height, width)

    train_val = int(np.logical_and(train_mask, val_mask).sum())
    train_test = int(np.logical_and(train_mask, test_mask).sum())
    val_test = int(np.logical_and(val_mask, test_mask).sum())

    log(f"Leakage check {dname}: train-val={train_val} px, "
        f"train-test={train_test} px, val-test={val_test} px "
        f"(all must be 0 for a leakage-free split)")

    is_clean = (train_val == 0 and train_test == 0 and val_test == 0)
    return is_clean, {"train_val": train_val, "train_test": train_test, "val_test": val_test}


def run_split_leakage_diagnostics(datasets=ALL_DATASETS, seed=0):
    """Run the split + leakage check for every dataset and persist a CSV
    record. Never raises: logs and records every case, strict or relaxed."""
    rows = []
    for dname in datasets:
        cube, gt = load_dataset(dname)
        train, val, test, strict = get_pixel_disjoint_split(dname, gt, seed)
        is_clean, overlaps = verify_split_leakage(dname, gt, train, val, test, DATASET_PATCH_SIZES[dname])

        rows.append({
            "Dataset": dname, "Seed": seed, "Strict": strict, "LeakageFree": is_clean,
            "Train": len(train), "Val": len(val), "Test": len(test),
            "TrainValOverlapPx": overlaps["train_val"],
            "TrainTestOverlapPx": overlaps["train_test"],
            "ValTestOverlapPx": overlaps["val_test"],
        })

        if not is_clean:
            log(f"NOTE: {dname} seed {seed} uses relaxed test selection; "
                f"see overlap counts above. This is reported, not silently hidden.")

    frame = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, "dataset_split_diagnostic.csv")
    frame.to_csv(path, index=False)
    log("Split leakage diagnostic saved to: %s" % path)
    return frame


# Fixed, dataset-independent color scheme for split zones.
SPLIT_ZONE_COLORS = {
    "background": "#f5f5f5",
    "train": "#2ca02c",
    "val": "#1f77b4",
    "test": "#d62728",
    "locked": "#cccccc",
}
SPLIT_ZONE_ORDER = ["background", "locked", "train", "val", "test"]
SPLIT_ZONE_INDEX = {name: i for i, name in enumerate(SPLIT_ZONE_ORDER)}


def _build_split_zone_map(gt, train, val, test, train_lock, val_lock):
    """
    Build a categorical zone map with fixed semantics:
    0 = background (unlabeled)
    1 = locked footprint (excluded, not train/val/test center)
    2 = train center
    3 = val center
    4 = test center
    """
    height, width = gt.shape
    zone = np.full((height, width), SPLIT_ZONE_INDEX["background"], dtype=np.int8)

    locked_only = (train_lock | val_lock) & (gt > 0)
    zone[locked_only] = SPLIT_ZONE_INDEX["locked"]

    for r, c in test:
        zone[r, c] = SPLIT_ZONE_INDEX["test"]
    for r, c in val:
        zone[r, c] = SPLIT_ZONE_INDEX["val"]
    for r, c in train:
        zone[r, c] = SPLIT_ZONE_INDEX["train"]

    return zone


def _split_zone_cmap():
    colors = [SPLIT_ZONE_COLORS[name] for name in SPLIT_ZONE_ORDER]
    cmap = ListedColormap(colors)
    bounds = np.arange(-0.5, len(SPLIT_ZONE_ORDER) + 0.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def plot_dataset_split_map(dname, gt, train, val, test, patch_size, seed, strict=None, save=True):
    """
    Save a single split-map figure for one dataset using a fixed color scheme
    that is identical across all datasets:
    green = train center
    blue  = validation center
    red   = test center
    gray  = locked footprint (excluded from centers)
    white = unlabeled background
    """
    height, width = gt.shape

    train_lock = np.zeros((height, width), dtype=bool)
    val_lock = np.zeros((height, width), dtype=bool)

    for r, c in train:
        r0, r1, c0, c1 = footprint_bounds(r, c, patch_size, height, width)
        train_lock[r0:r1, c0:c1] = True
    for r, c in val:
        r0, r1, c0, c1 = footprint_bounds(r, c, patch_size, height, width)
        val_lock[r0:r1, c0:c1] = True

    zone_map = _build_split_zone_map(gt, train, val, test, train_lock, val_lock)
    cmap, norm = _split_zone_cmap()

    total_labeled = max(1, int(np.sum(gt > 0)))
    train_pct = 100.0 * len(train) / total_labeled
    val_pct = 100.0 * len(val) / total_labeled
    test_pct = 100.0 * len(test) / total_labeled

    plt.figure(figsize=(6, 6))
    plt.imshow(zone_map, cmap=cmap, norm=norm)

    handles = [
        mpatches.Patch(color=SPLIT_ZONE_COLORS["train"], label="Train center (%.1f%%)" % train_pct),
        mpatches.Patch(color=SPLIT_ZONE_COLORS["val"], label="Val center (%.1f%%)" % val_pct),
        mpatches.Patch(color=SPLIT_ZONE_COLORS["test"], label="Test center (%.1f%%)" % test_pct),
        mpatches.Patch(color=SPLIT_ZONE_COLORS["locked"], label="Locked footprint (patch=%dpx)" % patch_size),
        mpatches.Patch(color=SPLIT_ZONE_COLORS["background"], label="Background"),
    ]
    plt.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False, fontsize=8)

    strict_label = "" if strict is None else (" [leakage-free]" if strict else " [relaxed test]")
    plt.title("%s pixel-disjoint split (seed=%d)%s" % (dname, seed, strict_label))
    plt.axis("off")
    plt.tight_layout()

    if save:
        path = os.path.join(OUT_DIR, "split_map_%s_seed%d.pdf" % (dname, seed))
        plt.savefig(path, bbox_inches="tight")
        log("Saved split map: %s" % path)
    plt.close()


def run_split_maps(datasets=ALL_DATASETS, seed=0):
    """Generate one split map per dataset with a shared fixed color scheme."""
    for dname in datasets:
        cube, gt = load_dataset(dname)
        train, val, test, strict = get_pixel_disjoint_split(dname, gt, seed)
        patch_size = DATASET_PATCH_SIZES[dname]
        plot_dataset_split_map(dname, gt, train, val, test, patch_size, seed, strict=strict)


def stratified_subsample_coords(coords, gt, max_total, seed=0):
    if max_total is None or len(coords) <= max_total:
        return list(coords)
    labels = np.asarray([gt[r, c] for r, c in coords])
    rng = np.random.RandomState(seed)
    selected = []
    for class_id in np.unique(labels):
        indices = np.flatnonzero(labels == class_id)
        take = max(1, int(round(max_total * len(indices) / len(coords))))
        selected.extend([coords[i] for i in rng.choice(indices, min(take, len(indices)), replace=False)])
    return selected


def coords_to_labels(gt, coords):
    return np.asarray([gt[r, c] for r, c in coords], dtype=np.int32)


def extract_spectral_vectors(cube, coords):
    return np.asarray([cube[r, c, :] for r, c in coords], dtype=np.float32)


def build_gray_and_flat(cube):
    return cube.reshape(-1, cube.shape[-1]), cube.mean(axis=-1)


def emp_stack(gray):
    if not HAS_SKIMAGE:
        return np.stack([uniform_filter(gray, size) for size in (3, 5, 7)], axis=-1)
    features = []
    for radius in (1, 2, 3):
        se = disk(radius)
        features.extend([opening(gray, se), closing(gray, se)])
    return np.stack(features, axis=-1)


def eap_stack(gray):
    features = []
    for threshold in (0.1, 0.3, 0.5, 0.7):
        binary = (gray > threshold).astype(np.float32)
        features.extend([uniform_filter(binary, 5), uniform_filter(1 - binary, 5)])
    return np.stack(features, axis=-1)


def gabor_stack(gray):
    features = []
    xx, yy = np.meshgrid(np.linspace(-1, 1, 9), np.linspace(-1, 1, 9))
    for frequency in (0.1, 0.25, 0.4):
        for theta in (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4):
            xr = xx * np.cos(theta) + yy * np.sin(theta)
            yr = -xx * np.sin(theta) + yy * np.cos(theta)
            kernel = np.exp(-(xr ** 2 + yr ** 2) / 0.5) * np.cos(2 * np.pi * frequency * xr)
            features.append(fftconvolve(gray, kernel, mode="same"))
    return np.stack(features, axis=-1)


def get_cached_filters(dname, gray):
    if dname not in _FILTER_CACHE:
        t0 = time.time()
        _FILTER_CACHE[dname] = (emp_stack(gray), eap_stack(gray), gabor_stack(gray))
        log("Image filters cached for %s (%.1fs)" % (dname, time.time() - t0))
    return _FILTER_CACHE[dname]


def at_coords(feature_map, coords):
    result = np.asarray([feature_map[r, c] for r, c in coords], dtype=np.float32)
    return result.reshape(-1, 1) if result.ndim == 1 else result


def compute_metrics(y_true, y_pred, labels=None):
    labels = np.asarray(labels if labels is not None else np.unique(np.concatenate([y_true, y_pred])))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    return {"OA": accuracy_score(y_true, y_pred), "AA": float(np.mean(per_class)), "Kappa": cohen_kappa_score(y_true, y_pred, labels=labels), "PerClassAcc": per_class}


def safe_components(n, rows, cols):
    return max(1, min(int(n), int(rows), int(cols)))


def build_all_method_features(X_train, y_train, X_test, EMP_train, EMP_test, EAP_train, EAP_test, Gabor_train, Gabor_test, seed):
    """Build matching train/test matrices for every requested classical method."""
    n = safe_components(30, X_train.shape[0], X_train.shape[1])
    pca = PCA(n_components=n, random_state=seed)
    pca_train, pca_test = pca.fit_transform(X_train), pca.transform(X_test)
    landmarks = safe_components(min(300, X_train.shape[0]), X_train.shape[0], X_train.shape[0])
    nys = Nystroem(kernel="rbf", gamma=1.0 / max(1, X_train.shape[1]), n_components=landmarks, random_state=seed)
    nys_train, nys_test = nys.fit_transform(X_train), nys.transform(X_test)
    kpca = PCA(n_components=safe_components(30, nys_train.shape[0], nys_train.shape[1]), random_state=seed)
    kpca_train, kpca_test = kpca.fit_transform(nys_train), kpca.transform(nys_test)
    ica = FastICA(n_components=n, whiten="unit-variance", random_state=seed, max_iter=1000, tol=1e-3)
    try:
        ica_train, ica_test = ica.fit_transform(X_train), ica.transform(X_test)
    except Exception:
        ica_train, ica_test = pca_train, pca_test
    variance_order = np.argsort(X_train.var(axis=0))[::-1][:n]
    fastpca = PCA(n_components=n, svd_solver="randomized", random_state=seed)
    fast_train, fast_test = fastpca.fit_transform(X_train), fastpca.transform(X_test)
    mvpca_scores = np.sum(np.abs(fastpca.components_[:min(10, len(fastpca.components_))]), axis=0)
    mvpca_order = np.argsort(mvpca_scores)[::-1][:n]
    return {
        "Raw+SVM": (X_train, X_test),
        "PCA+SVM": (pca_train, pca_test),
        "KPCA+SVM": (kpca_train, kpca_test),
        "ICA+SVM": (ica_train, ica_test),
        "SpaBS+SVM": (X_train[:, variance_order], X_test[:, variance_order]),
        "MVPCA+SVM": (X_train[:, mvpca_order], X_test[:, mvpca_order]),
        "EMP+SVM": (np.concatenate([X_train, EMP_train], axis=1), np.concatenate([X_test, EMP_test], axis=1)),
        "EAP+SVM": (np.concatenate([X_train, EAP_train], axis=1), np.concatenate([X_test, EAP_test], axis=1)),
        "Gabor+SVM": (np.concatenate([X_train, Gabor_train], axis=1), np.concatenate([X_test, Gabor_test], axis=1)),
        "FastPCA+RF": (fast_train, fast_test),
        "EMP+RF": (np.concatenate([X_train, EMP_train], axis=1), np.concatenate([X_test, EMP_test], axis=1)),
        "SVM+RF": (X_train, X_test),
    }


def fit_one_classical_method(method, X_train, y_train, X_test, seed):
    scaler = StandardScaler().fit(X_train)
    train_scaled, test_scaled = scaler.transform(X_train), scaler.transform(X_test)
    if method in ("FastPCA+RF", "EMP+RF"):
        model = RandomForestClassifier(n_estimators=RF_TREES, max_depth=RF_MAX_DEPTH, random_state=seed, n_jobs=1)
    elif method == "SVM+RF":
        svm = SVC(kernel="rbf", C=100.0, gamma="scale", probability=True, random_state=seed)
        rf = RandomForestClassifier(n_estimators=RF_TREES, max_depth=RF_MAX_DEPTH, random_state=seed, n_jobs=1)
        model = VotingClassifier(estimators=[("svm", svm), ("rf", rf)], voting="soft")
    else:
        model = SVC(kernel="rbf", C=100.0, gamma="scale", probability=True, random_state=seed)
    model.fit(train_scaled, y_train)
    return model.predict(test_scaled), {"model": model, "scaler": scaler, "X_train": X_train, "X_test": X_test}


# ============================================================
# 1D-CNN spectral baseline (PyTorch)
# ============================================================
if HAS_TORCH:
    class Spectral1DCNN(nn.Module):
        """Compact 1D-CNN operating directly on the raw per-pixel spectrum."""

        def __init__(self, n_bands, n_classes):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(1),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(64, n_classes),
            )

        def forward(self, x):
            x = self.features(x)
            return self.classifier(x)


def fit_predict_1dcnn(X_train, y_train, X_test, seed):
    """Fits a lightweight spectral 1D-CNN and returns test predictions.

    Falls back to a RawSVM prediction (logged) if PyTorch is unavailable,
    so downstream experiment loops never crash on a missing dependency.
    """
    if not HAS_TORCH:
        log("1D-CNN skipped: PyTorch not installed; falling back to Raw+SVM for this call.")
        return fit_one_classical_method("Raw+SVM", X_train, y_train, X_test, seed)

    torch.manual_seed(seed)
    np.random.seed(seed)

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    classes = np.unique(y_train)
    class_to_index = {int(c): i for i, c in enumerate(classes)}
    y_train_idx = np.asarray([class_to_index[int(c)] for c in y_train], dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = torch.from_numpy(X_train_scaled).unsqueeze(1)  # (N, 1, B)
    y_train_t = torch.from_numpy(y_train_idx)
    X_test_t = torch.from_numpy(X_test_scaled).unsqueeze(1)

    dataset = TensorDataset(X_train_t, y_train_t)
    batch_size = min(CNN1D_BATCH_SIZE, max(1, len(dataset)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = Spectral1DCNN(n_bands=X_train_scaled.shape[1], n_classes=len(classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CNN1D_LR, weight_decay=CNN1D_WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(CNN1D_EPOCHS):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_t.to(device))
        pred_idx = torch.argmax(test_logits, dim=1).cpu().numpy()

    pred = classes[pred_idx]
    payload = {"model": model, "scaler": scaler, "X_train": X_train, "X_test": X_test}
    return pred, payload


class StabilityReliabilitySpectralSpatialFusion:
    """SRSF with spectral, EMP, EAP, Gabor, joint and RF experts.

    select_input_mode:
        False (default, "SRSF(Ours)"): the compact stability-selected band
            subset feeds only the standalone "spectral" expert; every other
            expert (EMP/EAP/Gabor/joint/RF/FastPCA) uses the full spectrum.
        True ("SRSF-SelectedInput"): the SAME selected band subset feeds
            every expert, so the whole spectral-spatial pipeline operates on
            a compact representation instead of the raw full spectrum.
    """

    def __init__(self, n_bands=30, n_bootstraps=10, bootstrap_fraction=0.80,
                 stability_lambda=0.50, corr_threshold=0.98,
                 reliability_temperature=0.05, random_state=42,
                 select_input_mode=False):
        self.n_bands = int(n_bands)
        self.n_bootstraps = int(n_bootstraps)
        self.bootstrap_fraction = float(bootstrap_fraction)
        self.stability_lambda = float(stability_lambda)
        self.corr_threshold = float(corr_threshold)
        self.reliability_temperature = float(reliability_temperature)
        self.random_state = int(random_state)
        self.select_input_mode = bool(select_input_mode)
        self.selected_band_indices_ = None
        self.band_mean_scores_ = None
        self.band_std_scores_ = None
        self.band_stability_scores_ = None
        self.classes_ = None
        self.expert_names_ = None
        self.experts_ = None
        self.global_weights_ = None
        self.class_weights_ = None
        self.pca_model_ = None

    def _select_bands(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).reshape(-1)
        if len(X) != len(y):
            raise ValueError("SRSF band-selection row mismatch")
        rng = np.random.RandomState(self.random_state)
        score_list = []
        for _ in range(max(1, self.n_bootstraps)):
            indices = []
            for class_id in np.unique(y):
                class_indices = np.flatnonzero(y == class_id)
                take = max(1, min(len(class_indices), int(np.ceil(len(class_indices) * self.bootstrap_fraction))))
                indices.extend(rng.choice(class_indices, take, replace=False))
            scores, _ = f_classif(X[indices], y[indices])
            score_list.append(np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0))
        matrix = np.asarray(score_list, dtype=np.float64)
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        stable = mean - self.stability_lambda * std
        ranked = np.argsort(stable)[::-1]
        selected = []
        target = min(self.n_bands, X.shape[1])
        for index in ranked:
            index = int(index)
            if len(selected) >= target:
                break
            if not selected:
                selected.append(index)
                continue
            corr = np.corrcoef(X[:, index], X[:, selected], rowvar=False)[0, 1:]
            if np.max(np.abs(np.nan_to_num(corr, nan=1.0))) < self.corr_threshold:
                selected.append(index)
        for index in ranked:
            index = int(index)
            if len(selected) >= target:
                break
            if index not in selected:
                selected.append(index)
        self.selected_band_indices_ = np.asarray(selected, dtype=np.int64)
        self.band_mean_scores_ = mean.astype(np.float32)
        self.band_std_scores_ = std.astype(np.float32)
        self.band_stability_scores_ = stable.astype(np.float32)

    def _feature_blocks(self, X, EMP, EAP, Gabor, fit_pca=False):
        X = np.asarray(X, dtype=np.float32)
        EMP = np.asarray(EMP, dtype=np.float32)
        EAP = np.asarray(EAP, dtype=np.float32)
        Gabor = np.asarray(Gabor, dtype=np.float32)
        if not (len(X) == len(EMP) == len(EAP) == len(Gabor)):
            raise ValueError("SRSF feature row mismatch")
        if self.selected_band_indices_ is None:
            raise RuntimeError("SRSF bands have not been selected")

        X_selected = X[:, self.selected_band_indices_]
        # In select_input_mode, every spectral-derived block uses the SAME
        # compact selected band subset instead of the full spectrum.
        X_spatial_input = X_selected if self.select_input_mode else X

        if fit_pca or self.pca_model_ is None:
            self.pca_model_ = PCA(
                n_components=safe_components(30, X_spatial_input.shape[0], X_spatial_input.shape[1]),
                svd_solver="randomized",
                random_state=self.random_state,
            )
            fast = self.pca_model_.fit_transform(X_spatial_input)
        else:
            fast = self.pca_model_.transform(X_spatial_input)

        joint = np.concatenate([X_spatial_input, EMP, EAP, Gabor], axis=1)
        return {
            "spectral": X_selected,
            "emp": np.concatenate([X_spatial_input, EMP], axis=1),
            "eap": np.concatenate([X_spatial_input, EAP], axis=1),
            "gabor": np.concatenate([X_spatial_input, Gabor], axis=1),
            "joint": joint,
            "rf": joint,
            "fastpca_rf": fast,
        }

    def _fit_experts(self, X, EMP, EAP, Gabor, y, fit_pca=False):
        y = np.asarray(y).reshape(-1)
        blocks = self._feature_blocks(X, EMP, EAP, Gabor, fit_pca=fit_pca)
        experts = {}
        for name, features in blocks.items():
            if name in ("rf", "fastpca_rf"):
                scaler = StandardScaler().fit(features) if name == "rf" else None
                model = RandomForestClassifier(
                    n_estimators=RF_TREES,
                    max_depth=RF_MAX_DEPTH,
                    random_state=self.random_state,
                    n_jobs=1,
                )
            else:
                scaler = StandardScaler().fit(features)
                model = SVC(
                    kernel="rbf",
                    C=100.0,
                    gamma="scale",
                    probability=True,
                    random_state=self.random_state,
                )
            train_features = features if scaler is None else scaler.transform(features)
            model.fit(train_features, y)
            experts[name] = {"model": model, "scaler": scaler}
        return experts

    def _expert_probabilities(self, experts, X, EMP, EAP, Gabor):
        blocks = self._feature_blocks(X, EMP, EAP, Gabor, fit_pca=False)
        class_to_index = {int(c): i for i, c in enumerate(self.classes_)}
        outputs = {}
        for name, payload in experts.items():
            features = blocks[name]
            if payload["scaler"] is not None:
                features = payload["scaler"].transform(features)
            model = payload["model"]
            local = model.predict_proba(features)
            full = np.zeros((len(features), len(self.classes_)), dtype=np.float64)
            for j, class_id in enumerate(model.classes_):
                key = int(class_id)
                if key not in class_to_index:
                    raise ValueError(
                        "SRSF expert '%s' produced unknown class %r; global classes=%r"
                        % (name, class_id, self.classes_)
                    )
                full[:, class_to_index[key]] = local[:, j]
            outputs[name] = full
        return outputs

    def fit(self, X_train, y_train, X_val, y_val,
            EMP_train, EMP_val, EAP_train, EAP_val,
            Gabor_train, Gabor_val):
        X_train = np.asarray(X_train, dtype=np.float32)
        X_val = np.asarray(X_val, dtype=np.float32)
        y_train = np.asarray(y_train).reshape(-1)
        y_val = np.asarray(y_val).reshape(-1)
        if len(X_train) != len(y_train) or len(X_val) != len(y_val):
            raise ValueError("SRSF spectral/label mismatch")
        self.classes_ = np.unique(y_train)
        if len(self.classes_) < 2:
            raise ValueError("SRSF requires at least two training classes")
        self._select_bands(X_train, y_train)
        self.pca_model_ = None
        self.experts_ = self._fit_experts(X_train, EMP_train, EAP_train, Gabor_train, y_train, fit_pca=True)
        validation_probabilities = self._expert_probabilities(self.experts_, X_val, EMP_val, EAP_val, Gabor_val)
        self.expert_names_ = list(validation_probabilities)
        scores = np.asarray([
            accuracy_score(y_val, self.classes_[np.argmax(validation_probabilities[name], axis=1)])
            for name in self.expert_names_
        ], dtype=np.float64)
        logits = scores / max(self.reliability_temperature, 1e-8)
        logits -= logits.max()
        weights = np.exp(logits)
        weights /= weights.sum() + 1e-12
        self.global_weights_ = weights
        self.class_weights_ = np.tile(weights, (len(self.classes_), 1))
        return self

    def refit_development(self, X_trainval, y_trainval,
                           EMP_trainval, EAP_trainval, Gabor_trainval):
        y_trainval = np.asarray(y_trainval).reshape(-1)
        if len(X_trainval) != len(y_trainval):
            raise ValueError("SRSF development spectral/label mismatch")

        # Allow development to include classes not seen during fit.
        # Refit only on the intersection of known classes.
        known = set(self.classes_)
        mask = np.array([c in known for c in y_trainval])
        if not mask.any():
            # No known-class samples in development; skip refit.
            return self

        X_dev = X_trainval[mask]
        y_dev = y_trainval[mask]
        EMP_dev = EMP_trainval[mask]
        EAP_dev = EAP_trainval[mask]
        Gabor_dev = Gabor_trainval[mask]

        self.pca_model_ = None
        self.experts_ = self._fit_experts(X_dev, EMP_dev, EAP_dev, Gabor_dev, y_dev, fit_pca=True)
        return self

    def predict_proba(self, X, EMP, EAP, Gabor):
        if self.experts_ is None or self.classes_ is None:
            raise RuntimeError("SRSF must be fitted before prediction")
        probabilities = self._expert_probabilities(self.experts_, X, EMP, EAP, Gabor)
        output = np.zeros_like(next(iter(probabilities.values())), dtype=np.float64)
        for index, name in enumerate(self.expert_names_):
            output += probabilities[name] * self.class_weights_[:, index][None, :]
        return output / np.maximum(output.sum(axis=1, keepdims=True), 1e-12)

    def predict(self, X, EMP, EAP, Gabor):
        return self.classes_[np.argmax(self.predict_proba(X, EMP, EAP, Gabor), axis=1)]


def choose_shared_band_count(data, seed,
                              candidates=BAND_COUNT_CANDIDATES,
                              tie_tolerance=OA_TIE_TOLERANCE):
    """Select one k using validation OA, shared by both SRSF variants.

    The test set is never used during selection. Each candidate is fitted on
    the outer training split and evaluated only on the held-out validation
    split. The smallest k within tie_tolerance of the best validation OA is
    selected.
    """
    validation_oa = {}

    for k in candidates:
        candidate = StabilityReliabilitySpectralSpatialFusion(
            n_bands=int(k),
            random_state=seed,
            select_input_mode=False,
        )
        candidate.fit(
            data["Xtr"], data["ytr"],
            data["Xva"], data["yva"],
            data["Etr"], data["Eva"],
            data["Atr"], data["Ava"],
            data["Gtr"], data["Gva"],
        )
        validation_pred = candidate.predict(
            data["Xva"], data["Eva"], data["Ava"], data["Gva"]
        )
        validation_oa[int(k)] = 100.0 * accuracy_score(
            data["yva"], validation_pred
        )

    best_oa = max(validation_oa.values())
    eligible = [
        k for k, oa in validation_oa.items()
        if oa >= best_oa - tie_tolerance
    ]
    chosen_k = min(eligible)

    row = {
        "Dataset": data.get("dname", ""),
        "Seed": int(seed),
        "ChosenBandCount": int(chosen_k),
        "BestValidationOA": float(best_oa),
        "TieTolerance": float(tie_tolerance),
    }
    for k in candidates:
        row["ValOA_k%d" % int(k)] = float(validation_oa[int(k)])
    _BAND_COUNT_LOG.append(row)

    log(
        "[Shared band count] seed=%d: chosen k=%d; validation OA=%s"
        % (seed, chosen_k, {k: round(v, 2) for k, v in validation_oa.items()})
    )

    return int(chosen_k), validation_oa


def fit_srsf_with_shared_k(data, seed, chosen_k, select_input_mode):
    """Fit either architecture using the exact same selected band count."""
    model = StabilityReliabilitySpectralSpatialFusion(
        n_bands=int(chosen_k),
        random_state=seed,
        select_input_mode=bool(select_input_mode),
    )
    model.fit(
        data["Xtr"], data["ytr"],
        data["Xva"], data["yva"],
        data["Etr"], data["Eva"],
        data["Atr"], data["Ava"],
        data["Gtr"], data["Gva"],
    )
    model.refit_development(
        np.concatenate([data["Xtr"], data["Xva"]]),
        np.concatenate([data["ytr"], data["yva"]]),
        np.concatenate([data["Etr"], data["Eva"]]),
        np.concatenate([data["Atr"], data["Ava"]]),
        np.concatenate([data["Gtr"], data["Gva"]]),
    )
    return model


def save_band_count_log():
    if not _BAND_COUNT_LOG:
        return None
    frame = pd.DataFrame(_BAND_COUNT_LOG)
    path = os.path.join(OUT_DIR, "results_band_count_selection.csv")
    frame.to_csv(path, index=False)
    log("Saved shared band-count log: %s" % path)
    return frame


def save_srsf_domain_plots(model, dname, experiment, seed, tag="srsf"):
    band = pd.DataFrame({
        "band_index": np.arange(len(model.band_mean_scores_)),
        "mean_score": model.band_mean_scores_,
        "std_score": model.band_std_scores_,
        "stability_score": model.band_stability_scores_,
        "selected": np.isin(np.arange(len(model.band_mean_scores_)), model.selected_band_indices_).astype(int),
    })
    expert = pd.DataFrame({
        "expert_name": model.expert_names_,
        "global_weight": model.global_weights_,
        "class_mean_weight": model.class_weights_.mean(axis=0),
    })
    stem = "%s_%s_seed%d" % (dname, experiment, seed)
    band.to_csv(os.path.join(EXPLAIN_DIR, "%s_band_stability_%s.csv" % (tag, stem)), index=False)
    expert.to_csv(os.path.join(EXPLAIN_DIR, "%s_expert_weights_%s.csv" % (tag, stem)), index=False)

    plt.figure(figsize=(12, 4))
    plt.plot(band.band_index, band.stability_score, lw=1)
    selected = band[band.selected == 1]
    plt.scatter(selected.band_index, selected.stability_score, color="red", s=18)
    plt.xlabel("Spectral band")
    plt.ylabel("Stability score")
    plt.title("%s stable bands (k=%d): %s" % (tag.upper(), len(model.selected_band_indices_), stem))
    plt.tight_layout()
    plt.savefig(os.path.join(EXPLAIN_DIR, "%s_band_stability_%s.pdf" % (tag, stem)))
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.bar(expert.expert_name, expert.global_weight)
    plt.xticks(rotation=40, ha="right")
    plt.ylabel("Reliability weight")
    plt.title("%s expert reliability: %s" % (tag.upper(), stem))
    plt.tight_layout()
    plt.savefig(os.path.join(EXPLAIN_DIR, "%s_expert_weights_%s.pdf" % (tag, stem)))
    plt.close()


def save_shap_plot(method, payload, dname, experiment, seed):
    """Produces a per-method SHAP top-feature PDF; failures are logged without stopping experiments."""
    filename = "shap_summary_%s_%s_seed%d_%s" % (dname, experiment, seed, method.replace("+", "_").replace("(", "").replace(")", "").replace("-", "_"))
    if shap is None:
        log("SHAP unavailable; install with: pip install shap")
        return
    if method == "1D-CNN":
        log("SHAP skipped for 1D-CNN (PyTorch model); tabular SHAP explainer is not applicable here.")
        return
    X_train, X_test = payload["X_train"], payload["X_test"]
    background = X_train[:min(SHAP_BACKGROUND_SIZE, len(X_train))]
    explain = X_test[:min(SHAP_EXPLAIN_SIZE, len(X_test))]
    scaler, model = payload["scaler"], payload["model"]

    def probability_function(values):
        return model.predict_proba(scaler.transform(values))

    try:
        explainer = shap.Explainer(probability_function, background)
        values = np.asarray(explainer(explain).values)
        if values.ndim == 3:
            values = values.mean(axis=2)
        importance = np.mean(np.abs(values), axis=0)
        frame = pd.DataFrame({"feature": np.arange(len(importance)), "mean_abs_shap": importance, "method": method, "dataset": dname, "experiment": experiment, "seed": seed})
        frame.to_csv(os.path.join(EXPLAIN_DIR, filename + ".csv"), index=False)
        top = frame.nlargest(min(20, len(frame)), "mean_abs_shap").iloc[::-1]
        plt.figure(figsize=(8, 6))
        plt.barh(top.feature.astype(str), top.mean_abs_shap)
        plt.xlabel("Mean absolute SHAP value")
        plt.ylabel("Feature index")
        plt.title("SHAP summary: %s | %s" % (dname, method))
        plt.tight_layout()
        plt.savefig(os.path.join(EXPLAIN_DIR, filename + ".pdf"))
        plt.close()
    except Exception as exc:
        log("SHAP failed for %s / %s: %s" % (dname, method, exc))


def aggregate_explainability(experiment_name, datasets=ALL_DATASETS, runs=None):
    """
    Aggregate band-stability, expert-weights, and SHAP across runs.
    Produces one CSV + one PDF per dataset+experiment, for both SRSF tags.
    """
    if runs is None:
        runs = list(range(DEFAULT_N_RUNS))

    for dname in datasets:
        for tag in ("srsf", "srsfsel"):
            # ---- Band stability ----
            band_frames = []
            for seed in runs:
                path = os.path.join(EXPLAIN_DIR, "%s_band_stability_%s_%s_seed%d.csv" % (tag, dname, experiment_name, seed))
                if os.path.exists(path):
                    band_frames.append(pd.read_csv(path))
            if band_frames:
                band_all = pd.concat(band_frames, ignore_index=True)
                agg = band_all.groupby("band_index").agg(
                    mean_score=("mean_score", "mean"),
                    std_score=("std_score", "mean"),
                    stability_score=("stability_score", "mean"),
                    selected=("selected", lambda x: (x > 0).mean()),
                ).reset_index()
                stem = "%s_%s" % (dname, experiment_name)
                agg.to_csv(os.path.join(EXPLAIN_DIR, "%s_band_stability_%s.csv" % (tag, stem)), index=False)

                plt.figure(figsize=(12, 4))
                plt.plot(agg.band_index, agg.stability_score, lw=1)
                sel = agg[agg["selected"] > 0.5]
                plt.scatter(sel.band_index, sel.stability_score, color="red", s=18)
                plt.xlabel("Spectral band")
                plt.ylabel("Mean stability score")
                plt.title("%s band stability (aggregated): %s" % (tag.upper(), stem))
                plt.tight_layout()
                plt.savefig(os.path.join(EXPLAIN_DIR, "%s_band_stability_%s.pdf" % (tag, stem)))
                plt.close()

            # ---- Expert weights ----
            expert_frames = []
            for seed in runs:
                path = os.path.join(EXPLAIN_DIR, "%s_expert_weights_%s_%s_seed%d.csv" % (tag, dname, experiment_name, seed))
                if os.path.exists(path):
                    expert_frames.append(pd.read_csv(path))
            if expert_frames:
                expert_all = pd.concat(expert_frames, ignore_index=True)
                agg_exp = expert_all.groupby("expert_name").agg(
                    global_weight=("global_weight", "mean"),
                    class_mean_weight=("class_mean_weight", "mean"),
                ).reset_index()
                stem = "%s_%s" % (dname, experiment_name)
                agg_exp.to_csv(os.path.join(EXPLAIN_DIR, "%s_expert_weights_%s.csv" % (tag, stem)), index=False)

                plt.figure(figsize=(8, 4))
                plt.bar(agg_exp.expert_name, agg_exp.global_weight)
                plt.xticks(rotation=40, ha="right")
                plt.ylabel("Mean reliability weight")
                plt.title("%s expert weights (aggregated): %s" % (tag.upper(), stem))
                plt.tight_layout()
                plt.savefig(os.path.join(EXPLAIN_DIR, "%s_expert_weights_%s.pdf" % (tag, stem)))
                plt.close()

        # ---- SHAP summary (per dataset+experiment, aggregated across runs & methods) ----
        shap_frames = []
        for seed in runs:
            for method in METHODS:
                safe = method.replace("+", "_").replace("(", "").replace(")", "").replace("-", "_")
                path = os.path.join(EXPLAIN_DIR, "shap_summary_%s_%s_seed%d_%s.csv" % (dname, experiment_name, seed, safe))
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    df["method"] = method
                    shap_frames.append(df)

        if shap_frames:
            shap_all = pd.concat(shap_frames, ignore_index=True)
            agg_shap = (
                shap_all
                .groupby(["feature", "method"])["mean_abs_shap"]
                .mean()
                .reset_index()
            )
            stem = "%s_%s" % (dname, experiment_name)
            agg_shap.to_csv(os.path.join(EXPLAIN_DIR, "shap_summary_%s.csv" % stem), index=False)

            top = agg_shap.nlargest(20, "mean_abs_shap").iloc[::-1]
            plt.figure(figsize=(8, 6))
            plt.barh(top.feature.astype(str), top.mean_abs_shap)
            plt.xlabel("Mean absolute SHAP value (avg over runs & methods)")
            plt.ylabel("Feature index")
            plt.title("Aggregated SHAP summary: %s" % stem)
            plt.tight_layout()
            plt.savefig(os.path.join(EXPLAIN_DIR, "shap_summary_%s.pdf" % stem))
            plt.close()


def prepare_run_data(dname, cube, gt, gray, seed, n_per_class=None):
    train, val, test_all, strict = get_pixel_disjoint_split(
        dname, gt, seed, n_per_class=n_per_class,
        val_n_per_class=n_per_class if n_per_class is not None else None
    )
    test = stratified_subsample_coords(test_all, gt, MAX_TEST_EVAL_SAMPLES, seed)
    emp, eap, gabor = get_cached_filters(dname, gray)
    data = {}
    for prefix, coords in (("tr", train), ("va", val), ("te", test)):
        data["X" + prefix] = extract_spectral_vectors(cube, coords)
        data["E" + prefix] = at_coords(emp, coords)
        data["A" + prefix] = at_coords(eap, coords)
        data["G" + prefix] = at_coords(gabor, coords)
        data["y" + prefix] = coords_to_labels(gt, coords)
    data["train"], data["val"], data["test"] = train, val, test
    data["strict_split"] = strict
    return data


def evaluate_all_methods_one_run(dname, cube, gt, gray, seed,
                                  n_per_class=None, experiment="main",
                                  make_shap=True):
    data = prepare_run_data(dname, cube, gt, gray, seed, n_per_class)
    data["dname"] = dname

    features = build_all_method_features(
        data["Xtr"], data["ytr"], data["Xte"],
        data["Etr"], data["Ete"],
        data["Atr"], data["Ate"],
        data["Gtr"], data["Gte"], seed,
    )

    results = {}

    for method, (X_train, X_test) in features.items():
        pred, payload = fit_one_classical_method(
            method, X_train, data["ytr"], X_test, seed
        )
        results[method] = (data["yte"], pred)
        if make_shap:
            save_shap_plot(method, payload, dname, experiment, seed)

    # 1D-CNN spectral baseline: trained directly on raw per-pixel spectrum
    # (same spectral input as Raw+SVM), independent of the SRSF band-count
    # selection logic below.
    cnn_pred, cnn_payload = fit_predict_1dcnn(
        data["Xtr"], data["ytr"], data["Xte"], seed
    )
    results["1D-CNN"] = (data["yte"], cnn_pred)
    if make_shap:
        save_shap_plot("1D-CNN", cnn_payload, dname, experiment, seed)

    # One shared band-count choice for both SRSF architectures.
    chosen_k, _ = choose_shared_band_count(data, seed)

    hybrid = fit_srsf_with_shared_k(
        data, seed, chosen_k, select_input_mode=False
    )
    results["SRSF(Ours)"] = (
        data["yte"],
        hybrid.predict(data["Xte"], data["Ete"], data["Ate"], data["Gte"]),
    )
    save_srsf_domain_plots(hybrid, dname, experiment, seed, tag="srsf")

    selected_input = fit_srsf_with_shared_k(
        data, seed, chosen_k, select_input_mode=True
    )
    results["SRSF-SelectedInput"] = (
        data["yte"],
        selected_input.predict(
            data["Xte"], data["Ete"], data["Ate"], data["Gte"]
        ),
    )
    save_srsf_domain_plots(
        selected_input, dname, experiment, seed, tag="srsfsel"
    )

    return results, data, hybrid


def run_main_comparison(datasets=ALL_DATASETS, n_runs=DEFAULT_N_RUNS,
                        seeds_per_dataset=None):
    """Run the main comparison on explicit accepted seeds.

    If seeds_per_dataset is omitted, this preserves the old behavior and uses
    seeds 0..n_runs-1. For publication runs, pass the result of
    select_good_seeds().
    """
    if seeds_per_dataset is None:
        seeds_per_dataset = {
            dname: list(range(n_runs))
            for dname in datasets
        }

    rows = []
    for dname in datasets:
        cube, gt = load_dataset(dname)
        _, gray = build_gray_and_flat(cube)
        values = {method: [] for method in METHODS}

        for run_index, seed in enumerate(seeds_per_dataset.get(dname, [])):
            try:
                results, _, _ = evaluate_all_methods_one_run(
                    dname, cube, gt, gray, seed, experiment="main"
                )
                for method, (y_true, pred) in results.items():
                    values[method].append(compute_metrics(y_true, pred))
            except Exception as exc:
                log(
                    f"Main comparison failed {dname} seed={seed} "
                    f"(accepted run {run_index + 1}): {exc}"
                )

        for method, metrics in values.items():
            if metrics:
                rows.append({
                    "Dataset": dname,
                    "Method": method,
                    "OA_mean": 100 * np.mean([m["OA"] for m in metrics]),
                    "OA_std": 100 * np.std([m["OA"] for m in metrics]),
                    "AA_mean": 100 * np.mean([m["AA"] for m in metrics]),
                    "AA_std": 100 * np.std([m["AA"] for m in metrics]),
                    "Kappa_mean": 100 * np.mean([m["Kappa"] for m in metrics]),
                    "Kappa_std": 100 * np.std([m["Kappa"] for m in metrics]),
                    "N_runs": len(metrics),
                })

    frame = pd.DataFrame(rows)
    frame.to_csv(os.path.join(OUT_DIR, "results_main_comparison.csv"), index=False)
    return frame


def run_per_class(datasets=ALL_DATASETS, n_runs=DEFAULT_N_RUNS,
                  seeds_per_dataset=None):
    if seeds_per_dataset is None:
        seeds_per_dataset = {dname: list(range(n_runs)) for dname in datasets}

    rows = []
    for dname in datasets:
        cube, gt = load_dataset(dname)
        _, gray = build_gray_and_flat(cube)
        classes = np.unique(gt[gt > 0])
        for seed in seeds_per_dataset.get(dname, []):
            results, _, _ = evaluate_all_methods_one_run(dname, cube, gt, gray, seed, experiment="perclass", make_shap=False)
            for method, (y_true, pred) in results.items():
                precision, recall, f1, _ = precision_recall_fscore_support(y_true, pred, labels=classes, zero_division=0)
                cm = confusion_matrix(y_true, pred, labels=classes)
                accuracy = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
                support = cm.sum(axis=1)
                for i, class_id in enumerate(classes):
                    rows.append({
                        "Dataset": dname, "Run": seed, "Method": method, "Class": int(class_id),
                        "TestSupport": int(support[i]),
                        "Accuracy": 100 * accuracy[i], "Precision": 100 * precision[i],
                        "Recall": 100 * recall[i], "F1": 100 * f1[i],
                    })
    frame = pd.DataFrame(rows)
    frame.to_csv(os.path.join(OUT_DIR, "results_per_class.csv"), index=False)
    return frame


def run_fewshot(datasets=ALL_DATASETS, samples_per_class=(5, 10, 15, 20, 25), n_runs=DEFAULT_N_RUNS,
                seeds_per_dataset=None):
    if seeds_per_dataset is None:
        seeds_per_dataset = {dname: list(range(n_runs)) for dname in datasets}

    rows = []
    for dname in datasets:
        cube, gt = load_dataset(dname)
        _, gray = build_gray_and_flat(cube)
        for shots in samples_per_class:
            values = {method: [] for method in METHODS}
            for seed in seeds_per_dataset.get(dname, []):
                try:
                    results, _, _ = evaluate_all_methods_one_run(dname, cube, gt, gray, seed, n_per_class=shots, experiment="fewshot_%d" % shots, make_shap=False)
                    for method, (y_true, pred) in results.items():
                        values[method].append(compute_metrics(y_true, pred))
                except Exception as exc:
                    log("Few-shot failed %s shots=%d seed=%d: %s" % (dname, shots, seed, exc))
            for method, metrics in values.items():
                if metrics:
                    rows.append({
                        "Dataset": dname, "SamplesPerClass": shots, "Method": method,
                        "OA_mean": 100 * np.mean([m["OA"] for m in metrics]), "OA_std": 100 * np.std([m["OA"] for m in metrics]),
                        "AA_mean": 100 * np.mean([m["AA"] for m in metrics]), "AA_std": 100 * np.std([m["AA"] for m in metrics]),
                        "Kappa_mean": 100 * np.mean([m["Kappa"] for m in metrics]), "Kappa_std": 100 * np.std([m["Kappa"] for m in metrics]),
                        "N_runs": len(metrics),
                    })
    frame = pd.DataFrame(rows)
    frame.to_csv(os.path.join(OUT_DIR, "results_fewshot.csv"), index=False)
    return frame


ABLATION_CONFIGS = [
    "RawSVM",
    "SRSF_SpectralOnly_WithSelection",
    "SRSF_FusionNoBandSelection",
    "SRSF_UniformFusion",
    "SRSF(Ours)",
    "SRSF-SelectedInput",
]


def evaluate_ablation_one_run(dname, cube, gt, gray, seed):
    data = prepare_run_data(dname, cube, gt, gray, seed)
    data["dname"] = dname
    ytr, yv, yte = data["ytr"], data["yva"], data["yte"]
    Xtr, Xv, Xte = data["Xtr"], data["Xva"], data["Xte"]
    Etr, Ev, Ete = data["Etr"], data["Eva"], data["Ete"]
    Atr, Av, Ate = data["Atr"], data["Ava"], data["Ate"]
    Gtr, Gv, Gte = data["Gtr"], data["Gva"], data["Gte"]

    results = {}

    raw_pred, _ = fit_one_classical_method(
        "Raw+SVM", Xtr, ytr, Xte, seed
    )
    results["RawSVM"] = (yte, raw_pred)

    # One shared band-count choice for both architectures and controls.
    chosen_k, _ = choose_shared_band_count(data, seed)

    hybrid = fit_srsf_with_shared_k(
        data, seed, chosen_k, select_input_mode=False
    )
    results["SRSF(Ours)"] = (
        yte, hybrid.predict(Xte, Ete, Ate, Gte)
    )

    selected_input = fit_srsf_with_shared_k(
        data, seed, chosen_k, select_input_mode=True
    )
    results["SRSF-SelectedInput"] = (
        yte, selected_input.predict(Xte, Ete, Ate, Gte)
    )

    selected_bands = hybrid.selected_band_indices_
    spectral_pred, _ = fit_one_classical_method(
        "Raw+SVM", Xtr[:, selected_bands], ytr,
        Xte[:, selected_bands], seed
    )
    results["SRSF_SpectralOnly_WithSelection"] = (yte, spectral_pred)

    # Full-spectrum control: the selected-band spectral expert is replaced
    # by an all-band spectral expert; the rest of the hybrid architecture
    # remains unchanged.
    no_sel_model = StabilityReliabilitySpectralSpatialFusion(
        n_bands=chosen_k, random_state=seed, select_input_mode=False
    )
    no_sel_model.fit(
        Xtr, ytr, Xv, yv, Etr, Ev, Atr, Av, Gtr, Gv
    )
    no_sel_model.selected_band_indices_ = np.arange(Xtr.shape[1])
    no_sel_model.refit_development(
        np.concatenate([Xtr, Xv]), np.concatenate([ytr, yv]),
        np.concatenate([Etr, Ev]), np.concatenate([Atr, Av]),
        np.concatenate([Gtr, Gv]),
    )
    results["SRSF_FusionNoBandSelection"] = (
        yte, no_sel_model.predict(Xte, Ete, Ate, Gte)
    )

    uniform_model = StabilityReliabilitySpectralSpatialFusion(
        n_bands=chosen_k, random_state=seed, select_input_mode=False
    )
    uniform_model.fit(
        Xtr, ytr, Xv, yv, Etr, Ev, Atr, Av, Gtr, Gv
    )
    n_experts = len(uniform_model.expert_names_)
    uniform_model.global_weights_ = np.full(
        n_experts, 1.0 / n_experts
    )
    uniform_model.class_weights_ = np.tile(
        uniform_model.global_weights_,
        (len(uniform_model.classes_), 1),
    )
    uniform_model.refit_development(
        np.concatenate([Xtr, Xv]), np.concatenate([ytr, yv]),
        np.concatenate([Etr, Ev]), np.concatenate([Atr, Av]),
        np.concatenate([Gtr, Gv]),
    )
    results["SRSF_UniformFusion"] = (
        yte, uniform_model.predict(Xte, Ete, Ate, Gte)
    )

    return results


def run_ablation(datasets=ALL_DATASETS, n_runs=DEFAULT_N_RUNS,
                 seeds_per_dataset=None):
    if seeds_per_dataset is None:
        seeds_per_dataset = {dname: list(range(n_runs)) for dname in datasets}

    rows = []
    for dname in datasets:
        cube, gt = load_dataset(dname)
        _, gray = build_gray_and_flat(cube)
        values = {config: [] for config in ABLATION_CONFIGS}
        for seed in seeds_per_dataset.get(dname, []):
            results = evaluate_ablation_one_run(dname, cube, gt, gray, seed)
            for config in ABLATION_CONFIGS:
                y_true, y_pred = results[config]
                values[config].append(100 * accuracy_score(y_true, y_pred))
        for config, scores in values.items():
            rows.append({
                "Dataset": dname,
                "Config": config,
                "OA_mean": np.mean(scores),
                "OA_std": np.std(scores),
                "N_runs": len(scores),
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(os.path.join(OUT_DIR, "results_ablation.csv"), index=False)
    return frame


def measure_efficiency(fit_fn, predict_fn):
    tracemalloc.start()
    t0 = time.time()
    fit_fn()
    fit_time = time.time() - t0
    t1 = time.time()
    predict_fn()
    infer_time = time.time() - t1
    memory = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
    tracemalloc.stop()
    return fit_time, infer_time, memory


def run_efficiency_benchmark(datasets=ALL_DATASETS, n_runs=DEFAULT_N_RUNS,
                             seeds_per_dataset=None):
    if seeds_per_dataset is None:
        seeds_per_dataset = {dname: list(range(n_runs)) for dname in datasets}

    rows = []
    for dname in datasets:
        cube, gt = load_dataset(dname)
        _, gray = build_gray_and_flat(cube)
        for seed in seeds_per_dataset.get(dname, []):
            data = prepare_run_data(dname, cube, gt, gray, seed)
            data["dname"] = dname
            features = build_all_method_features(
                data["Xtr"], data["ytr"], data["Xte"],
                data["Etr"], data["Ete"],
                data["Atr"], data["Ate"],
                data["Gtr"], data["Gte"], seed,
            )
            for method, (Xtr, Xte) in features.items():
                holder = {}
                fit_time, infer_time, memory = measure_efficiency(
                    lambda: holder.setdefault("value", fit_one_classical_method(method, Xtr, data["ytr"], Xte, seed)),
                    lambda: holder["value"][0],
                )
                rows.append({"Dataset": dname, "Run": seed, "Method": method, "FitTime": fit_time, "InferTime": infer_time, "PeakMemoryMB": memory})

            # 1D-CNN efficiency timing (uses raw spectral train/test, same as Raw+SVM).
            holder = {}

            def fit_1dcnn():
                holder["value"] = fit_predict_1dcnn(
                    data["Xtr"], data["ytr"], data["Xte"], seed
                )

            def infer_1dcnn():
                return holder["value"][0]

            fit_time, infer_time, memory = measure_efficiency(fit_1dcnn, infer_1dcnn)
            rows.append({"Dataset": dname, "Run": seed, "Method": "1D-CNN", "FitTime": fit_time, "InferTime": infer_time, "PeakMemoryMB": memory})

            # Shared band-count selection for both SRSF variants.
            chosen_k, _ = choose_shared_band_count(data, seed)

            for method_name, select_input_mode in (
                ("SRSF(Ours)", False),
                ("SRSF-SelectedInput", True),
            ):
                holder = {}

                def fit_srsf(select_input_mode=select_input_mode):
                    holder["model"] = fit_srsf_with_shared_k(
                        data, seed, chosen_k, select_input_mode
                    )

                def infer_srsf():
                    return holder["model"].predict(
                        data["Xte"], data["Ete"], data["Ate"], data["Gte"]
                    )

                fit_time, infer_time, memory = measure_efficiency(fit_srsf, infer_srsf)
                rows.append({
                    "Dataset": dname, "Run": seed, "Method": method_name,
                    "FitTime": fit_time, "InferTime": infer_time,
                    "PeakMemoryMB": memory, "ChosenBandCount": chosen_k,
                })

    frame = pd.DataFrame(rows)
    frame.to_csv(os.path.join(OUT_DIR, "results_efficiency.csv"), index=False)
    return frame


def _dataset_class_names(dname, class_ids):
    """Return readable class labels; falls back safely when names are absent."""
    names = globals().get("CLASS_NAMES", {}).get(dname, {})
    if isinstance(names, dict):
        return [str(names.get(int(cid), "Class %d" % int(cid))) for cid in class_ids]
    if isinstance(names, (list, tuple)):
        return [str(names[int(cid)]) if int(cid) < len(names) else "Class %d" % int(cid) for cid in class_ids]
    return ["Class %d" % int(cid) for cid in class_ids]


def _make_dataset_cmap(class_ids):
    """Create a stable categorical colormap for the dataset's class IDs."""
    base = plt.get_cmap("tab20")
    colors = [base(i % base.N) for i in range(len(class_ids))]
    return ListedColormap(colors), BoundaryNorm(np.arange(-0.5, len(class_ids) + 0.5, 1), len(class_ids))


def _prediction_to_index_map(prediction_map, class_ids):
    """Map original labels to compact 0..N-1 categorical color indices."""
    index_map = np.full(prediction_map.shape, -1, dtype=np.int16)
    class_to_index = {int(cid): i for i, cid in enumerate(class_ids)}
    for class_id, index in class_to_index.items():
        index_map[prediction_map == class_id] = index
    return index_map


def _make_rgb_image(cube):
    """Create a display RGB image from three percentile-normalized bands."""
    height, width, bands = cube.shape
    if bands >= 3:
        band_indices = [
            min(bands - 1, int(0.70 * bands)),
            min(bands - 1, int(0.45 * bands)),
            min(bands - 1, int(0.15 * bands)),
        ]
    else:
        band_indices = [0, 0, 0]

    rgb = np.zeros((height, width, 3), dtype=np.float32)
    for channel, band_index in enumerate(band_indices):
        band = cube[:, :, band_index]
        low, high = np.percentile(band, (2, 98))
        rgb[:, :, channel] = np.clip((band - low) / (high - low + 1e-8), 0, 1)
    return rgb


def _save_dataset_panel(dname, cube, gt, prediction_maps, seed):
    """Save one panel containing RGB, gray, GT, and every method map."""
    method_names = list(prediction_maps.keys())
    class_ids = np.unique(gt[gt > 0]).astype(int)
    cmap, norm = _make_dataset_cmap(class_ids)
    class_names = _dataset_class_names(dname, class_ids)

    rgb = _make_rgb_image(cube)
    gray = cube.mean(axis=-1)

    n_panels = 3 + len(method_names)
    n_columns = 4
    n_rows = int(np.ceil(n_panels / float(n_columns)))

    fig, axes = plt.subplots(n_rows, n_columns, figsize=(4.0 * n_columns, 4.2 * n_rows), squeeze=False)
    axes = axes.ravel()

    axes[0].imshow(rgb)
    axes[0].set_title("Color image")
    axes[1].imshow(gray, cmap="gray")
    axes[1].set_title("Gray image")

    gt_index = _prediction_to_index_map(gt, class_ids)
    gt_display = np.ma.masked_where(gt_index < 0, gt_index)
    axes[2].imshow(gt_display, cmap=cmap, norm=norm)
    axes[2].set_title("Ground truth")

    for axis, method in zip(axes[3:], method_names):
        pred_index = _prediction_to_index_map(prediction_maps[method], class_ids)
        pred_display = np.ma.masked_where(pred_index < 0, pred_index)
        axis.imshow(pred_display, cmap=cmap, norm=norm)
        axis.set_title(method, fontsize=9)

    for axis in axes:
        axis.axis("off")
    for axis in axes[n_panels:]:
        axis.remove()

    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=10,
                   markerfacecolor=cmap(norm(i)), markeredgecolor="none",
                   label="%d: %s" % (int(class_id), class_names[i]))
        for i, class_id in enumerate(class_ids)
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=2, fontsize=8, frameon=True)
    fig.suptitle("%s classification panel (seed=%d)" % (dname, seed), fontsize=14)
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))

    path = os.path.join(MAP_DIR, "classification_panel_%s_seed%d.pdf" % (dname, seed))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log("Saved classification panel: %s" % path)
    return path


def run_classification_maps(datasets=ALL_DATASETS, seed=0):
    """Create one readable multi-panel classification figure per dataset."""
    for dname in datasets:
        cube, gt = load_dataset(dname)
        _, gray = build_gray_and_flat(cube)

        results, data, _ = evaluate_all_methods_one_run(dname, cube, gt, gray, seed, experiment="maps", make_shap=False)

        prediction_maps = {}
        for method, (_, prediction) in results.items():
            prediction_map = np.zeros_like(gt, dtype=np.int32)
            for (r, c), label in zip(data["test"], prediction):
                prediction_map[r, c] = int(label)
            prediction_maps[method] = prediction_map

        _save_dataset_panel(dname, cube, gt, prediction_maps, seed)


def log_split_summary(datasets=ALL_DATASETS, seeds=(0,)):
    """Print and save a CSV summary of train/val/test sizes for each dataset/seed."""
    rows = []
    for dname in datasets:
        cube, gt = load_dataset(dname)
        for seed in seeds:
            train, val, test, strict = get_pixel_disjoint_split(dname, gt, seed)
            rows.append({
                "Dataset": dname,
                "Seed": seed,
                "Strict": strict,
                "Labeled": int(np.sum(gt > 0)),
                "Train": len(train),
                "Val": len(val),
                "Test": len(test),
                "Train_pct": 100.0 * len(train) / max(1, int(np.sum(gt > 0))),
                "Val_pct": 100.0 * len(val) / max(1, int(np.sum(gt > 0))),
                "Test_pct": 100.0 * len(test) / max(1, int(np.sum(gt > 0))),
            })
            log("[Split summary] %s seed=%d strict=%s: train=%d (%.2f%%) val=%d (%.2f%%) test=%d (%.2f%%)" % (
                dname, seed, strict,
                len(train), rows[-1]["Train_pct"],
                len(val), rows[-1]["Val_pct"],
                len(test), rows[-1]["Test_pct"],
            ))
    frame = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, "dataset_split_summary.csv")
    frame.to_csv(path, index=False)
    log("Split summary saved to: %s" % path)
    return frame


# ============================================================
# Seed filtering for publication-quality runs
# ============================================================

def is_split_acceptable(dname, gt, seed, n_per_class=None, val_n_per_class=None):
    """Return (accepted, diagnostic_row) for one dataset/seed.

    A seed is accepted only when:
      1. strict=True;
      2. all measured footprint overlaps are exactly zero;
      3. test-center count reaches MIN_STRICT_TEST_CENTERS[dname].

    Note: this selection uses labels and geometry only. It never fits a model,
    reads test features for optimization, or uses test predictions/accuracy.
    """
    min_test = MIN_STRICT_TEST_CENTERS.get(dname, 100)

    try:
        train, val, test, strict = get_pixel_disjoint_split(
            dname,
            gt,
            seed,
            n_per_class=n_per_class,
            val_n_per_class=val_n_per_class,
        )
        leakage_free, overlaps = verify_split_leakage(
            dname,
            gt,
            train,
            val,
            test,
            DATASET_PATCH_SIZES[dname],
        )

        accepted = bool(
            strict
            and leakage_free
            and len(test) >= min_test
        )

        reason = "accepted" if accepted else (
            "not_strict" if not strict else
            "leakage_detected" if not leakage_free else
            f"test_below_minimum_{len(test)}<{min_test}"
        )

        row = {
            "Dataset": dname,
            "Seed": int(seed),
            "Accepted": accepted,
            "Reason": reason,
            "Strict": bool(strict),
            "LeakageFree": bool(leakage_free),
            "Train": len(train),
            "Val": len(val),
            "Test": len(test),
            "MinRequiredTest": min_test,
            "TrainValOverlapPx": overlaps["train_val"],
            "TrainTestOverlapPx": overlaps["train_test"],
            "ValTestOverlapPx": overlaps["val_test"],
        }
        return accepted, row

    except Exception as exc:
        row = {
            "Dataset": dname,
            "Seed": int(seed),
            "Accepted": False,
            "Reason": "split_exception",
            "Strict": False,
            "LeakageFree": False,
            "Train": np.nan,
            "Val": np.nan,
            "Test": np.nan,
            "MinRequiredTest": min_test,
            "TrainValOverlapPx": np.nan,
            "TrainTestOverlapPx": np.nan,
            "ValTestOverlapPx": np.nan,
            "Exception": repr(exc),
        }
        log(f"Seed check failed for {dname} seed={seed}: {exc}")
        return False, row


def select_good_seeds(datasets=ALL_DATASETS, target_runs=TARGET_RUNS,
                      max_candidate_seeds=MAX_CANDIDATE_SEEDS,
                      n_per_class=None, val_n_per_class=None):
    """Select up to target_runs acceptable seeds independently per dataset.

    The function examines candidate seeds 0..max_candidate_seeds-1 and stops
    for a dataset as soon as target_runs accepted seeds are found. A seed is
    never selected based on model accuracy. The full candidate audit is saved
    to seed_selection_audit.csv.
    """
    seeds_per_dataset = {}
    audit_rows = []

    for dname in datasets:
        _, gt = load_dataset(dname)
        accepted_seeds = []

        for candidate_seed in range(max_candidate_seeds):
            if len(accepted_seeds) >= target_runs:
                break

            accepted, row = is_split_acceptable(
                dname,
                gt,
                candidate_seed,
                n_per_class=n_per_class,
                val_n_per_class=val_n_per_class,
            )
            audit_rows.append(row)

            if accepted:
                accepted_seeds.append(candidate_seed)
                log(
                    f"Accepted {dname} seed {candidate_seed}: "
                    f"{len(accepted_seeds)}/{target_runs} "
                    f"(test={row['Test']})."
                )
            else:
                log(
                    f"Rejected {dname} seed {candidate_seed}: "
                    f"{row['Reason']}."
                )

        seeds_per_dataset[dname] = accepted_seeds

        if len(accepted_seeds) < target_runs:
            raise RuntimeError(
                f"Only {len(accepted_seeds)}/{target_runs} acceptable seeds found "
                f"for {dname} after {max_candidate_seeds} candidates. "
                f"Lower the minimum test threshold, reduce train/val fractions, "
                f"or revise the patch protocol before running experiments."
            )

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(os.path.join(OUT_DIR, "seed_selection_audit.csv"), index=False)

    seed_rows = [
        {"Dataset": dname, "RunIndex": i, "Seed": seed}
        for dname, seeds in seeds_per_dataset.items()
        for i, seed in enumerate(seeds)
    ]
    pd.DataFrame(seed_rows).to_csv(
        os.path.join(OUT_DIR, "accepted_seeds.csv"),
        index=False,
    )

    log(f"Accepted-seed list saved to {OUT_DIR}/accepted_seeds.csv")
    return seeds_per_dataset


if __name__ == "__main__":
    print("EXECUTING FILE:", os.path.abspath(__file__), flush=True)
    if not HAS_TORCH:
        log("WARNING: PyTorch not found. 1D-CNN will fall back to Raw+SVM predictions "
            "(logged per call) so experiments still run end-to-end. "
            "Install PyTorch (pip install torch) for a real 1D-CNN baseline.")

    # Smoke test: use 1 only for quick debugging. Use 10 for reported runs.
    TARGET_RUNS = 10
    MAX_CANDIDATE_SEEDS = 2000000

    # Select acceptable seeds using only split geometry and labels. No model
    # accuracy, test prediction, or test tuning is used here.
    accepted_seeds = select_good_seeds(
        datasets=ALL_DATASETS,
        target_runs=TARGET_RUNS,
        max_candidate_seeds=MAX_CANDIDATE_SEEDS,
    )

    # Audit the exact accepted splits.
    for dname in ALL_DATASETS:
        cube, gt = load_dataset(dname)
        for seed in accepted_seeds[dname]:
            train, val, test, strict = get_pixel_disjoint_split(dname, gt, seed)
            leakage_free, overlaps = verify_split_leakage(
                dname, gt, train, val, test, DATASET_PATCH_SIZES[dname]
            )
            if not (strict and leakage_free and len(test) >= MIN_STRICT_TEST_CENTERS[dname]):
                raise AssertionError(
                    f"Accepted seed failed final audit: {dname}, seed={seed}, "
                    f"strict={strict}, leakage_free={leakage_free}, test={len(test)}"
                )

    # Save a summary for accepted seeds only.
    summary_rows = []
    for dname in ALL_DATASETS:
        cube, gt = load_dataset(dname)
        for seed in accepted_seeds[dname]:
            train, val, test, strict = get_pixel_disjoint_split(dname, gt, seed)
            summary_rows.append({
                "Dataset": dname,
                "Seed": seed,
                "Strict": strict,
                "Labeled": int(np.sum(gt > 0)),
                "Train": len(train),
                "Val": len(val),
                "Test": len(test),
                "MinRequiredTest": MIN_STRICT_TEST_CENTERS[dname],
            })
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(OUT_DIR, "accepted_split_summary.csv"),
        index=False,
    )

    # Run split leakage diagnostic for all candidate seeds (optional audit).
    # You can restrict this to accepted seeds if you prefer.
    run_split_leakage_diagnostics(datasets=ALL_DATASETS, seed=accepted_seeds[ALL_DATASETS[0]][0])

    # Split summary for accepted seeds.
    log_split_summary(datasets=ALL_DATASETS, seeds=tuple(accepted_seeds[ALL_DATASETS[0]]))

    # Split maps for the first accepted seed.
    run_split_maps(datasets=ALL_DATASETS, seed=accepted_seeds[ALL_DATASETS[0]][0])

   # Main experiment: exactly TARGET_RUNS accepted seeds per dataset.
    run_main_comparison(datasets=ALL_DATASETS, n_runs=TARGET_RUNS, seeds_per_dataset=accepted_seeds, )

    # Optional: run remaining experiments with the same accepted seeds.
    run_per_class(datasets=ALL_DATASETS, n_runs=TARGET_RUNS, seeds_per_dataset=accepted_seeds)
    run_ablation(datasets=ALL_DATASETS, n_runs=TARGET_RUNS, seeds_per_dataset=accepted_seeds)
    run_fewshot(datasets=ALL_DATASETS, n_runs=TARGET_RUNS, seeds_per_dataset=accepted_seeds)
    run_efficiency_benchmark(datasets=ALL_DATASETS, n_runs=TARGET_RUNS, seeds_per_dataset=accepted_seeds)

    # Classification maps for the first accepted seed.
    run_classification_maps(datasets=ALL_DATASETS, seed=accepted_seeds[ALL_DATASETS[0]][0])

    save_band_count_log()
    log("All SRSF-only experiments completed. Output directory: %s" % OUT_DIR)

    # Aggregate explainability for each experiment type
    for exp_name in ("main", "perclass", "ablation", "fewshot_5", "fewshot_10", "fewshot_15", "fewshot_20", "fewshot_25", "efficiency", "maps"):
        try:
            aggregate_explainability(exp_name, datasets=ALL_DATASETS, runs=[0])
        except Exception as e:
            log("Aggregation failed for experiment %s: %s" % (exp_name, e))


            