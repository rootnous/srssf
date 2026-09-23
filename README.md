# srssf
SRSSF: Stability-Reliability Spectral-Spatial Fusion for Hyperspectral Image Classification

SRSSF Framework
===============

This repository contains the official implementation of the SRSSF framework and all codes used for experiments in our paper.

Repository Structure
--------------------
main.py    : Core framework and experiment code
plots.py         : code for plotting SHAP (main comparison) and per-class mean F1 score heatmap

Usage
-----
Run experiments with the SRSSF framework:
    python main.py 

Generate plots after placing .csv files (per_class and  4 shap files from main comparison) obtained as outputs from main.py in same path as code file. 
    python plots.py 

Datasets
--------
We evaluate on four benchmark hyperspectral datasets: Indian Pines, Pavia University, Salinas, and KSC.
Datasets are cited collectively via IEEE DataPort, with accessed copies hosted on Kaggle.

Acknowledgment
--------------
We thank Kaggle contributors Douglas Martins and Sreevalli Manda for hosting datasets and acknowledge Microsoft for providing cloud credits.

## Citation

If you use this code or our findings in your research, please cite our paper:

```bibtex
@article{author2026title,
  author    = {Ghosh, Chirantan},
  title     = {},
  journal   = {Journal Name or arXiv},
  year      = {2026},
  volume    = {1},
  number    = {1},
  pages     = {1--10},
  doi       = {}
}







