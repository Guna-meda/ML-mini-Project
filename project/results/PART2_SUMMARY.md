# Part 2: Gradient Boosting

## Model choice

Gradient Boosting is a sequential tree ensemble. Each tree is fitted to improve on errors from the preceding ensemble. This provides a useful comparison with the reference's bagged trees, which fit trees independently and aggregate their predictions.

## Fair comparison setup

The extension uses the same curated dataset, reference logarithmic five-bin target, normalized search-volume feature, one-hot encoded features, and `StratifiedKFold(n_splits=4, shuffle=True, random_state=1)` splits as the reproduction. It reports fold-averaged weighted precision, recall, F1, and precision for each price bin. The estimator uses scikit-learn defaults with `random_state=1`.

## Result and comparison

Gradient Boosting validation weighted precision, recall, and F1 are **0.4228**, **0.4457**, and **0.4322**. Its validation weighted F1 is **-0.0235** versus the best reference bagging result (`bagged_decision_tree_100`, 0.4557) and **-0.0082** versus the best reference SVM (`ovo_linear_svm`, 0.4404). The highest score in the saved comparison is **0.4557** (`bagged_decision_tree_100`), so Gradient Boosting performs **worse** by weighted F1.

The five price bins are imbalanced: the dataset has 146, 153, 46, 10, and 4 examples in bins 1 through 5. Gradient Boosting's validation precision is 0.0000 for bins 4 and 5, so the fold-average weighted scores are driven mainly by the two common lower-price bins. This likely contributes to its lower score and limits conclusions about the rarest price ranges.

The detailed model metrics, including training scores and per-class precision, are in `part2_model_comparison.csv` and `part2_model_comparison.md`. Since the data contains 359 rows, the fold-average is a useful comparison under the assignment's required split, but it may vary if the dataset or split changes.
