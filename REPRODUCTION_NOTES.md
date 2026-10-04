# Reproduction Notes

This reproduction keeps the original reference files under `CS229-master/` untouched and places the runnable, organized implementation under `project/`.

## Compatibility fixes

- The reference notebook uses `kfold.split(X, y_true)` in several model cells, but `y_true` is not defined. The surrounding notebook defines `y = df['price_bin']`, so this reproduction uses `y` for all Stratified K-Fold splits.
- The reference bagging cell uses the older scikit-learn argument `base_estimator=DecisionTreeClassifier()`. Current scikit-learn uses `estimator=DecisionTreeClassifier()`, which is the only API change made for that model.
- Per-class precision is computed with explicit labels `[1, 2, 3, 4, 5]` so every saved row has stable class columns even when a fold has no predictions for a class.

## Preserved methodology

- Uses the provided curated dataset at `CS229-master/project/src/data_sourcing/data_curated.csv`.
- Does not scrape StockX or use any external data.
- Uses the reference logarithmic five-bin target construction.
- Uses `log(search_volume)` min-max normalization.
- Uses one-hot encoded `category`, `material`, `special_version`, and numeric `search_volume_normalized` for the standard models.
- Uses ordinal-encoded mixed features for the custom precomputed kernel.
- Uses `StratifiedKFold(n_splits=4, shuffle=True, random_state=1)` for all experiments.
- Reproduces only the reference models. No extension model was added.

## Reference reported results

The provided repository contains `notebook.ipynb` and the curated dataset, but no paper/report with explicit final metric tables was present in this workspace. The generated `reference_vs_reproduction_comparison.csv` therefore marks reference values as `Not explicitly reported` rather than inventing them.
