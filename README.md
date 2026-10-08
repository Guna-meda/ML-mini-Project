# Funko Pop Price-Bin ML Project

Predict a five-bin logarithmic price category for Funko Pop figures. The canonical implementation lives in `project/`.

## Part 1: Reference reproduction

Reproduces only the reference models on `project/data/data_curated.csv`:

- `linear_svm_ovr`, `ovo_linear_svm`, `ovo_rbf_svm`
- `custom_kernel_svm`
- `decision_tree`
- bagged decision trees (sweep over 2-200 trees; best, `bagged_decision_tree_100`, is reported)

```powershell
python -m project.src.reproduction
```

Outputs are written to `project/results/` (`reference_reproduction_results.csv`, `bagging_results.csv`, figures, etc.).

## Part 2: Gradient Boosting extension

One additional model not used in the reference: `sklearn.ensemble.GradientBoostingClassifier(random_state=1)`. It uses the same data, target bins, features, `StratifiedKFold(n_splits=4, shuffle=True, random_state=1)`, weighted precision/recall/F1 and per-class precision as Part 1, and is compared against the saved Part 1 results.

```powershell
python -m project.src.part2
```

Outputs: `part2_gradient_boosting_results.csv/.md`, `part2_model_comparison.csv/.md`, `PART2_SUMMARY.md` in `project/results/`. Run Part 1 first.

## Run everything

```powershell
python run_all.py
python -m unittest discover -s project/tests -p "test_*.py"
```

## Results (validation weighted F1, produced by the code above)

| Model | Test F1 |
| --- | --- |
| linear_svm_ovr | 0.4176 |
| ovo_linear_svm | 0.4404 |
| ovo_rbf_svm | 0.4382 |
| custom_kernel_svm | 0.4205 |
| decision_tree | 0.4089 |
| bagged_decision_tree_100 | 0.4557 |
| gradient_boosting (Part 2) | 0.4322 |
