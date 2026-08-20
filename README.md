# ECAC Project - Human Activity Recognition

This repository contains the coursework project for **ECAC - Engenharia de Caracteristicas para Aprendizagem Computacional**. The project studies Human Activity Recognition (HAR) from wearable sensor data, covering feature engineering, outlier analysis, dimensionality reduction, feature selection, transfer-learning embeddings, data augmentation, model evaluation, and a simple deployment-style classification experiment.

The main entry point is [`src/mainActivity.py`](src/mainActivity.py), which runs the complete pipeline for both project milestones.

## Project Scope

The project is organized around two main goals:

- **Meta 1 - Feature Engineering**
  - Load and inspect multi-participant sensor data.
  - Analyze outliers with IQR, Z-score, K-Means, and DBSCAN.
  - Test statistical significance of sensor-derived measurements.
  - Extract temporal and spectral handcrafted features from sliding windows.
  - Apply PCA for dimensionality reduction.
  - Compare feature-selection methods with Fisher Score and ReliefF.

- **Meta 2 - Transfer Learning and Data Augmentation**
  - Analyze class balance for activities 1 to 7.
  - Generate synthetic samples with a custom SMOTE implementation.
  - Extract and compare transfer-learning embeddings.
  - Build within-subject and between-subject train/validation/test splits.
  - Evaluate k-NN classifiers across feature scenarios.
  - Run repeated model evaluations, confusion-matrix visualizations, hypothesis tests, and deployment-style predictions.

## Dataset

The project uses the **FORTH-TRACE Dataset version 1.0**, included under [`dataset/`](dataset/).

The dataset contains recordings from 15 participants wearing 5 Shimmer sensor nodes:

- Left wrist
- Right wrist
- Torso
- Right thigh
- Left ankle

Each CSV file contains accelerometer, gyroscope, magnetometer, timestamp, device ID, and activity-label columns. More details are available in [`dataset/README.md`](dataset/README.md).

## Repository Structure

```text
.
|-- data/
|   |-- cache/                 # Cached model-evaluation results
|   `-- features/              # Extracted handcrafted features and embeddings
|-- dataset/                   # FORTH-TRACE CSV files
|-- logs/                      # Tuning and final evaluation logs
|-- plots/                     # Generated visualizations
|-- src/
|   |-- mainActivity.py        # Main script that runs the full pipeline
|   |-- modules/
|   |   |-- meta1/             # Feature engineering and outlier analysis
|   |   `-- meta2/             # Transfer learning, SMOTE, splitting, evaluation
|   `-- utils/                 # Shared constants, caching, logging, windows, embeddings
|-- requirements.txt
|-- TP1_A.pdf
|-- TP1_B.pdf
`-- Report_PT.pdf
```

## Requirements

- Python 3.10 or newer is recommended.
- The dependencies listed in [`requirements.txt`](requirements.txt).
- Internet access may be required the first time embeddings are recomputed, because the project loads the pretrained `harnet5` model from `OxWearables/ssl-wearables` through `torch.hub`.

## Installation

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you use Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running the Project

Run the complete pipeline with:

```bash
python3 src/mainActivity.py
```

The script executes all exercises in sequence and prints progress, metrics, timings, and summary tables to the terminal.

By default, `src/mainActivity.py` uses cached handcrafted features and embeddings from [`data/features/`](data/features/) when available:

```python
USE_CACHED_FEATURES = True
USE_SKLEARN_KNN = True
```

To force feature and embedding extraction from scratch, change `USE_CACHED_FEATURES` to `False` in [`src/mainActivity.py`](src/mainActivity.py). This can take longer and may require downloading the pretrained embedding model.

## Generated Outputs

The main script writes or reuses several output artefacts:

- [`data/features/feature_set.npz`](data/features/feature_set.npz): handcrafted feature matrix, labels, metadata, and feature names.
- [`data/features/embeddings_set.npz`](data/features/embeddings_set.npz): transfer-learning embeddings aligned with the same windows.
- [`data/features/feature_info.txt`](data/features/feature_info.txt): feature-set summary and feature names.
- [`data/cache/results.pkl`](data/cache/results.pkl): cached repeated-evaluation results.
- [`plots/meta1/`](plots/meta1/): outlier, PCA, clustering, and feature-engineering visualizations.
- [`plots/meta2/`](plots/meta2/): SMOTE, confusion matrix, and hypothesis-test visualizations.
- [`logs/`](logs/): model tuning and final evaluation logs.

## Feature Extraction

The handcrafted feature set is extracted from 5-second sliding windows with 50% overlap. It includes:

- Temporal features such as mean, median, standard deviation, variance, min/max, range, RMS, MAD, IQR, skewness, kurtosis, zero-crossing rate, and mean-crossing rate.
- Spectral features such as dominant frequency, spectral energy, spectral entropy, and spectral centroid.
- Multi-sensor features inspired by HAR literature.

The current cached feature set contains **27,845 windows x 174 handcrafted features**.

## Model Evaluation

The evaluation pipeline compares:

- Dataset types: handcrafted features and transfer-learning embeddings.
- Split strategies: within-subject and between-subject.
- Feature scenarios: all features, PCA-reduced features, and ReliefF-selected features.
- k-NN classifiers with odd `k` values from 1 to 19.

Repeated evaluations are cached automatically to avoid recomputing the full experiment on every run.

## Notes

- The code and terminal output use Portuguese exercise names because they follow the original coursework statement.
- Generated files are already present in this repository, so the project can usually be inspected without rerunning the full pipeline.
- The included PDFs contain the assignment statements and project report.
