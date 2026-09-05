# Biohub Cell Tracking Pipeline (v7)

**End-to-end pipeline for 3D+time cell detection, linking, division inference, and lineage reconstruction**  
**Competition**: [Biohub – Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development) (Kaggle)

**Main deliverable**: `bio-cell-track-updated-v7.ipynb` (396 cells – fully documented, production-oriented notebook)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Competition Context](#2-competition-context)
3. [Dataset Description](#3-dataset-description)
4. [Notebook Architecture (Full Section Breakdown)](#4-notebook-architecture-full-section-breakdown)
5. [Core Methodologies](#5-core-methodologies)
6. [Feature Engineering Details](#6-feature-engineering-details)
7. [Tracking System (Safe Hybrid Tracker)](#7-tracking-system-safe-hybrid-tracker)
8. [3D Detection Pipeline](#8-3d-detection-pipeline)
9. [Visualization Suite](#9-visualization-suite)
10. [Installation & Environment](#10-installation--environment)
11. [How to Run](#11-how-to-run)
12. [Outputs & Artifacts](#12-outputs--artifacts)
13. [Validation & Quality Checks](#13-validation--quality-checks)
14. [Strengths & Design Philosophy](#14-strengths--design-philosophy)
15. [Limitations & Future Improvements](#15-limitations--future-improvements)
16. [Recommended Repository Structure](#16-recommended-repository-structure)
17. [License](#17-license)
18. [Acknowledgments](#18-acknowledgments)
19. [Citation](#19-citation)
20. [Appendix – Quick Reference](#20-appendix--quick-reference)

---

## 1. Project Overview

This repository contains a complete, research-grade pipeline for solving the **Biohub – Cell Tracking During Development** Kaggle competition. The goal is to detect cell nuclei centers and reconstruct accurate tracks (including cell divisions) across large 3D time-lapse light-sheet microscopy volumes of zebrafish embryos.

The main artifact is a single, heavily engineered Jupyter notebook:

```
bio-cell-track-updated-v7.ipynb
```

The notebook goes far beyond a simple baseline. It includes:

- Robust data loading and integrity validation
- Deep exploratory data analysis (EDA)
- Physics-informed feature engineering
- Candidate edge generation and labeling
- Group-aware train/validation splitting
- Classical + optional deep-learning 3D cell detection
- A carefully engineered **Safe Hybrid Tracker** designed to minimize track switches and fragmentation
- Extensive diagnostic and visualization tools
- Final test-set tracking and submission preparation

Every major step is transparent, reproducible, and easy to inspect inside the notebook.

---

## 2. Competition Context

- **Task**: Detect cell centers and link them across time (including divisions) in 3D+time volumes.
- **Organism**: Zebrafish embryos (early development).
- **Imaging modality**: Light-sheet microscopy.
- **Data volume**: Large 4D volumes (T × Z × Y × X).
- **Ground truth**: Sparse (only a subset of cells are annotated).
- **Annotation format**: GEFF (tracksdata).
- **Evaluation**: Edge and division quality (True Positive / False Positive / False Negative counts, micro-averaged Jaccard index, and a combined final score).
- **Prize pool**: $60,000.

### Why the problem is hard

- High cell density
- Very similar appearance of nuclei
- Frequent cell divisions
- Strong anisotropy in the Z-axis
- Large data size
- Sparse supervision (most cells are not annotated)

---

## 3. Dataset Description

### Official Directory Structure (Kaggle)

```
/kaggle/input/competitions/biohub-cell-tracking-during-development/
├── train/
│   ├── {sample_id}.zarr          # 4D image volume (T, Z, Y, X)
│   └── {sample_id}.geff          # Ground-truth graph
└── test/
    └── {sample_id}.zarr          # Image volume only (no ground truth)
```

### GEFF Format (Ground Truth)

| Component | Description |
|-----------|-------------|
| **Nodes** | Approximate cell centers with fields: `node_id`, `t`, `z`, `y`, `x` |
| **Edges** | Links between nodes. A continuation edge connects a cell to itself in the next frame. A division appears as one parent node at time `t` connected to two daughter nodes at time `t+1`. |

### Additional Characteristics

- Multiple independent movies (different embryos).
- Movies belong to groups identifiable by the prefix of the sample ID (e.g. `44b6`, `6bba`).
- Strong anisotropy between XY and Z resolution.
- Significant intensity variation across movies and depth.

---

## 4. Notebook Architecture (Full Section Breakdown)

The notebook contains **396 cells** organized into clearly numbered major sections:

### Section 1 – Data Collecting and Loading
- Directory walking and complete inventory of files
- Installation of required packages (`zarr`, etc.)
- Helper functions to load GEFF nodes and edges into clean pandas DataFrames
- Loading of all training graphs into a dictionary
- Zarr volume loading utilities
- Sample submission schema inspection
- Basic movie metadata extraction

### Section 2 – Data Validation
- Full inventory of train and test samples
- Missing value checks on nodes and edges
- Duplicate detection
- Train / test overlap detection and correction
- Group leakage verification (critical for proper validation)
- Node–edge consistency checks
- Comprehensive validation summary report

### Section 3 – Exploratory Data Analysis (EDA)
- Node and edge count distributions
- Cell count per frame analysis
- Spatial coordinate distributions
- Movement and displacement statistics
- Temporal consistency of edges
- Detailed cell division geometry analysis
- Identification of difficult division events
- Graph topology statistics (global and per group)
- Advanced structural graph analysis
- Intensity distribution studies
- Advanced intensity analysis
- Visualization summary of key EDA findings

### Section 4 – Feature Engineering
- Conversion from voxel coordinates to physical units (micrometers)
- Distance, speed, and kinematic features
- Direction and turning-angle features
- Crowding / local neighborhood features
- Graph topology features (in-degree, out-degree)
- Division-specific geometry features
- Ambiguity features useful for ranking candidate edges
- Unified feature extraction pipeline

### Section 5 – Candidate Edge Generation
- Distance-based candidate generation
- Rank-based candidate selection
- Computation of ambiguity features
- Batch processing across all movies
- Distance-by-rank diagnostic plots

### Section 6 – Candidate Labeling
- Labeling of candidate edges against ground-truth edges
- True-edge coverage analysis
- Rank-wise true-edge statistics
- Class imbalance analysis

### Section 7 – Train / Validation Split
- Group-aware splitting strategy
- Explicit prevention of data leakage between train and validation

### Later Sections (8 – 16+)
- Candidate ranking / scoring models
- Safe tracking configuration
- Implementation of the Safe Hybrid Tracker
- 3D cell detection (classical methods)
- Optional 3D CNN candidate filter (training + inference)
- Full integrated pipeline: Detection → Safe Tracking
- Track quality reporting
- Fragmentation analysis
- Track-switch detection
- Suspicious movement diagnostics
- Final tracking on all test movies
- Extensive visualization suite
- Metrics computation and submission file preparation

---

## 5. Core Methodologies

1. **Sparse Supervision Handling**  
   Only annotated edges are used for supervision. Background detections and unannotated cells are carefully managed so they do not harm training.

2. **Physics-Informed Design**  
   All distances and motion features are computed in real physical units (micrometers) rather than raw voxels. This makes thresholds meaningful across different movies.

3. **Candidate-Based Linking**  
   Instead of evaluating every possible pair of detections, a high-quality set of candidate edges is generated first and then ranked/scored.

4. **Safe Hybrid Tracker**  
   Combines rule-based safety gates with learned scores to strongly reduce identity switches and track fragmentation.

5. **Modular Detection**  
   Supports both classical 3D detection and an optional trainable 3D CNN filter that can improve precision.

---

## 6. Feature Engineering Details

| Category          | Features |
|-------------------|----------|
| Spatial           | Physical (x, y, z) coordinates, pairwise Euclidean distances |
| Kinematic         | Speed, displacement vectors, velocity |
| Directional       | Turning angles, direction consistency between consecutive frames |
| Crowding          | Local density, distance to nearest neighbors |
| Graph Topology    | In-degree, out-degree, local connectivity patterns |
| Division          | Parent–daughter distances, angles, size ratios, geometric consistency |
| Ambiguity         | Rank of the candidate, gap to the second-best candidate, local competition score |

All features are designed to be useful both for machine-learning ranking models and for hard safety thresholds inside the tracker.

---

## 7. Tracking System (Safe Hybrid Tracker)

The core tracking engine was designed with robustness as the primary goal. It includes multiple protective layers:

- Physical distance gates (not pure voxel distance)
- Frame-gap-aware distance thresholds
- Motion prediction (velocity extrapolation from previous frames)
- Prediction residual distance checks
- Hybrid candidate generation (rule-based + model scores)
- Global track ID reassignment after linking
- Post-tracking quality report
- Fragmentation detection
- Track-switch detection (sudden large jumps, abrupt direction changes, etc.)
- Suspicious movement analysis

This multi-layered design prioritizes biological plausibility and long-term track stability.

---

## 8. 3D Detection Pipeline

- Configurable classical 3D detection based on local maxima and filtering
- Optional 3D CNN candidate filter:
  - Extraction of 3D patches around candidate detections
  - Proper intensity normalization
  - Training and validation on labeled detections
  - Inference-time filtering of classical detections
- Visualization of detections on Maximum Intensity Projections (MIP) and individual Z-slices
- Seamless integration into the Safe Hybrid Tracker

---

## 9. Visualization Suite

The notebook produces a rich collection of diagnostic and scientific visualizations:

- Raw volume slices and Maximum Intensity Projections
- Detection overlays on intensity images
- 3D / 2D projected track trajectories
- Lineage / track timeline views
- Close-up views of division events
- Fragmentation heatmaps and statistics
- Highlighting of suspicious tracks
- Distance-by-rank diagnostic plots
- Quality and continuity dashboards per movie

These visualizations are essential for debugging, understanding failure modes, and communicating results.

---

## 10. Installation & Environment

### Recommended Python Version
Python ≥ 3.10

### Core Dependencies
```bash
pip install zarr numpy pandas matplotlib seaborn scikit-learn tqdm jupyter ipywidgets
```

### Optional Dependencies (for 3D CNN components)
```bash
pip install torch torchvision
```

### Kaggle Environment
The notebook already contains the necessary `%pip install` cells and is designed to run directly on the Kaggle platform.

---

## 11. How to Run

1. Clone this repository.
2. Ensure the competition data is available at the expected path (or update the `DATA_ROOT` variable inside the notebook).
3. Open `bio-cell-track-updated-v7.ipynb`.
4. Run the cells sequentially from the top (recommended for first-time execution).
5. After the data-loading and validation sections have completed successfully, you may jump to later sections.
6. For full test-set inference, execute the integrated 3D detection + Safe Hybrid Tracker sections at the end of the notebook.

---

## 12. Outputs & Artifacts

Typical artifacts produced by the notebook:

- Dictionary of loaded training graphs (nodes + edges)
- Feature tables for all movies
- Candidate edge tables (with labels)
- Detection results for every test movie
- Final track tables containing consistent track IDs
- Per-movie quality and fragmentation reports
- Multiple diagnostic plots and scientific visualizations
- Submission-ready CSV files (when the final export cells are executed)

---

## 13. Validation & Quality Checks

The notebook contains extensive built-in validation:

- Data integrity and missing-value checks
- Duplicate detection
- Group leakage prevention
- True-edge coverage of generated candidates
- Track fragmentation statistics
- Heuristic track-switch detectors
- Sudden movement / jump detectors
- Per-movie quality summaries and global reports

These checks help ensure that both the data and the tracking results remain reliable.

---

## 14. Strengths & Design Philosophy

- **Full transparency** – Everything lives in one well-documented notebook.
- **Robustness first** – Multiple safety layers against identity switches and fragmentation.
- **Physics-aware** – Real-world units and motion models instead of pure voxel heuristics.
- **Modular design** – Detection and linking components can be swapped or improved independently.
- **Diagnostic-rich** – Strong emphasis on understanding *why* tracks fail.
- **Competition-ready** – Clear path from raw data all the way to a submission file.

---

## 15. Limitations & Future Improvements

### Current Limitations
- Classical detection can struggle in extremely dense or low-contrast regions.
- The optional 3D CNN filter requires additional training time and careful hyper-parameter tuning.
- Some safety thresholds are still manually chosen.
- A fully end-to-end deep linking model (e.g. Transformer-based) is not yet the primary method.

### Possible Future Directions
- End-to-end learned detection + linking architecture
- Temporal attention / Transformer models operating on node features
- Specialized models for division events
- Active learning focused on hard cases
- Multi-embryo joint training
- Streaming / real-time tracking version
- Better uncertainty estimation for each link

---

## 16. Recommended Repository Structure

```
biohub-cell-tracking-pipeline/
├── bio-cell-track-updated-v7.ipynb   # Main notebook (core of the project)
├── README.md                         # This documentation file
├── requirements.txt                  # Python dependencies
├── LICENSE                           # MIT (recommended)
├── CITATION.cff                      # Optional citation metadata
├── docs/                             # Extra documentation (if needed)
├── notebooks/                        # Experimental or older notebooks
├── src/                              # Optional extracted reusable Python modules
└── results/                          # Saved plots, track tables, submission files
```

---

## 17. License

MIT License is recommended for the code and notebook.

Please also respect the official competition data license (the Biohub dataset is released under CC0).

---

## 18. Acknowledgments

- Chan Zuckerberg Biohub and the Royer Lab for creating this important public challenge and dataset.
- Developers of the tracksdata / GEFF format.
- The broader cell-tracking and bioimage analysis community.
- Authors of Ultrack and related tools that provided valuable inspiration.

---

## 19. Citation

If you use this notebook or pipeline in academic work, please cite:

```bibtex
@misc{biohub-cell-tracking-pipeline-v7,
  title        = {Biohub Cell Tracking Pipeline (v7)},
  author       = {[Your Name]},
  year         = {2026},
  howpublished = {\url{https://github.com/[your-username]/biohub-cell-tracking-pipeline}},
  note         = {End-to-end notebook for the Biohub Cell Tracking During Development Kaggle competition}
}
```

Please also cite the official competition and the original Biohub / Zebrahub / Ultrack publications.

---

## 20. Appendix – Quick Reference

| Item                        | Value |
|-----------------------------|-------|
| Main notebook               | `bio-cell-track-updated-v7.ipynb` |
| Total cells                 | 396 |
| Primary data formats        | Zarr (images), GEFF (graphs) |
| Key innovation              | Safe Hybrid Tracker with multiple robustness layers |
| Target platform             | Kaggle + local research environments |
| Python version              | ≥ 3.10 |
| Core libraries              | zarr, numpy, pandas, matplotlib, seaborn, scikit-learn |
| Optional deep learning      | PyTorch |

---

**Notebook Version**: v7 (Updated)  
**Documentation Version**: 1.0  
**Last Updated**: September 2026  
**Status**: Competition-ready / Research-grade pipeline

---

*Built for scientific rigor, robustness, full reproducibility, and clear communication of results.*
```
