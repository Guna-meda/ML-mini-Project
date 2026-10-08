# Application of Machine Learning Methods to Evaluate Collectibles

A machine-learning mini-project that predicts the **price bin** of a Funko Pop figure from its category, material, special-version flag and search volume.

The project has two parts:

1. **Part 1 – Reference reproduction:** reproduce the models from the reference CS229 project (and only those).
2. **Part 2 – Extension:** add one new model not used in the reference: **Gradient Boosting**.

## Dataset

`project/data/data_curated.csv` – 359 Funko Pop listings (product name, category, material, special version, search volume, average sale price, ...).

**Target.** `average_sale_price` is split into 5 logarithmically spaced bins between the minimum and maximum price (bin 1 = cheapest):

| Price bin | Count | Price range ($) |
| --- | --- | --- |
| 1 | 146 | 14 – 37 |
| 2 | 153 | 38.4 – 101 |
| 3 | 46 | 108 – 257 |
| 4 | 10 | 322 – 600 |
| 5 | 4 | 800 – 2150 |

The classes are strongly imbalanced, so the rare bins (4 and 5) are almost never predicted correctly.

**Features.** Normalized log search volume, plus one-hot encoded `category`, `material` and `special_version`. (The custom-kernel SVM uses an ordinal-encoded version of the same features.)

## Evaluation protocol

Identical for every model:

- `StratifiedKFold(n_splits=4, shuffle=True, random_state=1)`
- Fold-averaged **weighted** precision, recall and F1 on train and validation folds
- Per-class validation precision

## Models

**Part 1 (reference):** one-vs-rest linear SVM, one-vs-one linear SVM, one-vs-one RBF SVM, custom-kernel SVM, decision tree, and bagged decision trees (2–200 trees; the best, 100 trees, is reported).

**Part 2 (extension):** `sklearn.ensemble.GradientBoostingClassifier(random_state=1)`. Boosting fits trees sequentially, each correcting its predecessors, in contrast to the independent trees of bagging.

## Results (validation weighted F1)

| Model | Precision | Recall | F1 |
| --- | --- | --- | --- |
| linear_svm_ovr | 0.4234 | 0.4542 | 0.4176 |
| ovo_linear_svm | 0.4077 | 0.4819 | 0.4404 |
| ovo_rbf_svm | 0.4027 | 0.4821 | 0.4382 |
| custom_kernel_svm | 0.3900 | 0.4596 | 0.4205 |
| decision_tree | 0.4079 | 0.4151 | 0.4089 |
| **bagged_decision_tree_100** | 0.4522 | 0.4764 | **0.4557** |
| gradient_boosting (Part 2) | 0.4228 | 0.4457 | 0.4322 |

Bagged decision trees are the best model. Gradient Boosting (0.4322) beats the plain decision tree and the linear OvR SVM and the custom-kernel SVM, but falls short of the best bagging result and the OvO SVMs. See `project/results/PART2_SUMMARY.md` for the discussion.

## Project structure

```
project/
├── data/data_curated.csv        # curated dataset
├── notebooks/                   # reference reproduction notebook
├── src/
│   ├── preprocessing.py         # price bins, features, encoders
│   ├── metrics.py               # MyMetric (weighted + per-class precision)
│   ├── custom_kernel.py         # mixed-feature custom SVM kernel
│   ├── models.py                # reference models, bagging, Gradient Boosting
│   ├── reproduction.py          # Part 1 entry point
│   └── part2.py                 # Part 2 entry point
├── tests/                       # unit tests
└── results/                     # CSV/Markdown results, summary, figures
run_all.py                       # runs Part 1 then Part 2
REPRODUCTION_NOTES.md            # compatibility fixes vs. the reference code
```

## Setup and usage

```powershell
pip install -r requirements.txt
```

Run from the repository root:

```powershell
python -m project.src.reproduction   # Part 1 -> reference_reproduction_results.csv, bagging_results.csv, figures
python -m project.src.part2          # Part 2 -> part2_*.csv/.md, PART2_SUMMARY.md (run Part 1 first)
python run_all.py                    # both parts in sequence
python -m unittest discover -s project/tests -p "test_*.py"
```

All numbers above are produced by this code and saved in `project/results/`. Part 2 reads the saved Part 1 results to build `part2_model_comparison.csv`.

## Notes

- Gradient Boosting is fitted on a dense feature matrix; scikit-learn's sparse code path gives slightly different splits (F1 0.4295 vs 0.4322). The reference models use the sparse matrix unchanged.
- With only 359 rows and tiny rare classes, fold-averaged scores are noisy; differences of ~0.01 F1 should not be over-interpreted.
- Compatibility fixes relative to the reference notebook are listed in `REPRODUCTION_NOTES.md`.
