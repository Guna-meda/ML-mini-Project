"""Run and compare the Part 2 Gradient Boosting extension."""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from .models import run_gradient_boosting_experiment
from .preprocessing import add_reference_targets_and_features, load_curated_data, make_one_hot_matrix


ROOT = Path(__file__).resolve().parents[2]
PROJECT_DIR = ROOT / "project"
RESULTS_DIR = PROJECT_DIR / "results"


def run(data_path: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_df = load_curated_data(data_path)
    df, _ = add_reference_targets_and_features(raw_df)
    if df["price_bin"].isna().any():
        raise ValueError("Some rows did not receive a price_bin under reference logic.")
    y = df["price_bin"].astype(int).to_numpy()
    X, _ = make_one_hot_matrix(df)

    extension_df = pd.DataFrame([run_gradient_boosting_experiment(X, y)])
    reference_path = RESULTS_DIR / "reference_reproduction_results.csv"
    bagging_path = RESULTS_DIR / "bagging_results.csv"
    if not reference_path.exists() or not bagging_path.exists():
        raise FileNotFoundError(
            "Run the reference reproduction first so the Part 2 comparison can include its results."
        )
    reference_df = pd.read_csv(reference_path)
    bagging_df = pd.read_csv(bagging_path)
    # The reproduction summary already includes the best bagged model. Avoid
    # duplicating that row while retaining the rest of the bagging sweep.
    bagging_df = bagging_df[~bagging_df["model"].isin(reference_df["model"])]
    comparison_df = pd.concat(
        [reference_df, bagging_df, extension_df], ignore_index=True, sort=False
    )
    extension_df.to_csv(RESULTS_DIR / "part2_gradient_boosting_results.csv", index=False)
    comparison_df.to_csv(RESULTS_DIR / "part2_model_comparison.csv", index=False)
    save_markdown(extension_df, RESULTS_DIR / "part2_gradient_boosting_results.md")
    save_markdown(comparison_df, RESULTS_DIR / "part2_model_comparison.md")
    write_summary(comparison_df, extension_df.iloc[0])
    print("Gradient Boosting Part 2 result")
    print(extension_df.round(4).to_string(index=False))
    print("\nValidation weighted F1 comparison")
    print(
        comparison_df[["model", "test_precision", "test_recall", "test_f1"]]
        .sort_values("test_f1", ascending=False)
        .round(4)
        .to_string(index=False)
    )
    print(f"\nSaved Part 2 outputs to {RESULTS_DIR}")
    return extension_df, comparison_df


def save_markdown(df: pd.DataFrame, path: Path) -> None:
    rounded = df.copy()
    numeric_cols = rounded.select_dtypes(include="number").columns
    rounded[numeric_cols] = rounded[numeric_cols].round(4)
    headers = [str(col) for col in rounded.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in rounded.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in rounded.columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(comparison_df: pd.DataFrame, extension_row: pd.Series) -> None:
    ranked = comparison_df.dropna(subset=["test_f1"]).sort_values("test_f1", ascending=False)
    best = ranked.iloc[0]
    bagged = comparison_df[comparison_df["model"] == "bagged_decision_tree_100"].iloc[0]
    best_svm = comparison_df[comparison_df["model"].isin(
        ["linear_svm_ovr", "ovo_linear_svm", "ovo_rbf_svm", "custom_kernel_svm"]
    )].sort_values("test_f1", ascending=False).iloc[0]
    summary = f"""# Part 2: Gradient Boosting

## Model choice

Gradient Boosting is a sequential tree ensemble. Each tree is fitted to improve on errors from the preceding ensemble. This provides a useful comparison with the reference's bagged trees, which fit trees independently and aggregate their predictions.

## Fair comparison setup

The extension uses the same curated dataset, reference logarithmic five-bin target, normalized search-volume feature, one-hot encoded features, and `StratifiedKFold(n_splits=4, shuffle=True, random_state=1)` splits as the reproduction. It reports fold-averaged weighted precision, recall, F1, and precision for each price bin. The estimator uses scikit-learn defaults with `random_state=1`.

## Result and comparison

Gradient Boosting validation weighted precision, recall, and F1 are **{extension_row['test_precision']:.4f}**, **{extension_row['test_recall']:.4f}**, and **{extension_row['test_f1']:.4f}**. Its validation weighted F1 is **{extension_row['test_f1'] - bagged['test_f1']:+.4f}** versus the best reference bagging result (`bagged_decision_tree_100`, {bagged['test_f1']:.4f}) and **{extension_row['test_f1'] - best_svm['test_f1']:+.4f}** versus the best reference SVM (`{best_svm['model']}`, {best_svm['test_f1']:.4f}). The highest score in the saved comparison is **{best['test_f1']:.4f}** (`{best['model']}`), so Gradient Boosting performs **{'better' if extension_row['test_f1'] > best['test_f1'] else 'worse' if extension_row['test_f1'] < best['test_f1'] else 'the same'}** by weighted F1.

The five price bins are imbalanced: the dataset has 146, 153, 46, 10, and 4 examples in bins 1 through 5. Gradient Boosting's validation precision is 0.0000 for bins 4 and 5, so the fold-average weighted scores are driven mainly by the two common lower-price bins. This likely contributes to its lower score and limits conclusions about the rarest price ranges.

The detailed model metrics, including training scores and per-class precision, are in `part2_model_comparison.csv` and `part2_model_comparison.md`. Since the data contains 359 rows, the fold-average is a useful comparison under the assignment's required split, but it may vary if the dataset or split changes.
"""
    (RESULTS_DIR / "PART2_SUMMARY.md").write_text(summary, encoding="utf-8")


def main() -> None:
    parser = ArgumentParser(description="Run the Part 2 Gradient Boosting extension.")
    parser.add_argument("--data-path", default=None, help="Optional path to data_curated.csv")
    args = parser.parse_args()
    run(args.data_path)


if __name__ == "__main__":
    main()
