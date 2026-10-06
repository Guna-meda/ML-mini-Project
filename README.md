# Application of Machine Learning Methods to Evaluate Collectibles

This repository reproduces the reference CS229 Funko Pop collectible-pricing experiments using the curated dataset bundled in this repository. The reproduction is self-contained under `project/` and does not require the original `CS229-master` repository.

## Dataset

The reproduction uses `project/data/data_curated.csv` (originally taken unchanged from the reference repository's `project/src/data_sourcing/`). No new data is scraped. A different file can be used via `--data-path`.

## Methodology

Price binning -> search-volume normalization -> categorical encoding -> stratified 4-fold cross-validation -> reference model evaluation.

The target `price_bin` is created from `average_sale_price` using the reference logarithmically spaced five-bin formula. The model features are `category`, `material`, `special_version`, and `search_volume_normalized`; `average_sale_price` is not used as a model feature.

## Models Reproduced

- Linear SVM / One-vs-Rest
- Linear SVM / One-vs-One
- RBF SVM / One-vs-One
- Custom-kernel SVM
- Decision Tree
- Bagged Decision Tree with 2, 5, 10, 20, 50, 100, and 200 trees

## Metrics

Each model is evaluated with weighted precision, weighted recall, weighted F1, and per-class precision for price bins 1 through 5 on both training and validation folds.

## Running

Create and activate an environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the full reproduction:

```powershell
python -m project.src.reproduction
```

Results are saved in `project/results/`:

- `reference_reproduction_results.csv`
- `bagging_results.csv`
- `price_bin_boundaries.csv`
- `price_bin_summary.csv`
- `reference_vs_reproduction_comparison.csv`
- figures in `project/results/figures/`

## Notes

See `REPRODUCTION_NOTES.md` for compatibility fixes and assumptions. No additional model has been added in this reproduction phase.
