"""Run the full reference reproduction and save results."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .models import run_bagging_experiment, run_reference_models
from .preprocessing import (
    add_reference_targets_and_features,
    load_curated_data,
    make_one_hot_matrix,
    make_ordinal_matrix,
    price_bin_summary,
)


ROOT = Path(__file__).resolve().parents[2]
PROJECT_DIR = ROOT / "project"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def save_figures(df: pd.DataFrame, bagging_df: pd.DataFrame) -> None:
    sns.set_theme(style="ticks", color_codes=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    bins = range(0, 1000, 10)
    plt.hist(df["average_sale_price"].clip(lower=0, upper=990), bins=bins)
    plt.xlabel("Average sale price")
    plt.ylabel("Count")
    plt.title("Average sale price distribution")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "average_sale_price_histogram.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="price_bin", order=[1, 2, 3, 4, 5])
    plt.xlabel("Price bin")
    plt.ylabel("Count")
    plt.title("Price-bin distribution")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "price_bin_distribution.png", dpi=160)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    sns.stripplot(
        ax=axes[0][0],
        data=df,
        x="category",
        y="search_volume_normalized",
        hue="price_bin",
    )
    sns.stripplot(ax=axes[0][1], data=df, x="category", y="material", hue="price_bin")
    sns.stripplot(
        ax=axes[1][0],
        data=df,
        x="category",
        y="special_version",
        hue="price_bin",
    )
    sns.stripplot(ax=axes[1][1], data=df, x="category", y="material", hue="price_bin")
    for ax in axes.ravel():
        ax.tick_params(axis="x", rotation=70)
    axes[0][0].legend(title="price_bin", bbox_to_anchor=(-0.25, 1), loc=2)
    for ax in [axes[0][1], axes[1][0], axes[1][1]]:
        legend = ax.get_legend()
        if legend:
            legend.remove()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_price_bin_relationships.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    plot_df = bagging_df[
        [
            "num_trees",
            "train_precision",
            "train_recall",
            "train_f1",
            "test_precision",
            "test_recall",
            "test_f1",
        ]
    ].melt(id_vars="num_trees", var_name="metric", value_name="score")
    sns.lineplot(data=plot_df, x="num_trees", y="score", hue="metric", marker="o")
    plt.xlabel("Number of trees")
    plt.ylabel("Score")
    plt.title("Bagged decision tree performance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "bagging_performance.png", dpi=160)
    plt.close()


def save_markdown_table(df: pd.DataFrame, path: Path) -> None:
    rounded = df.copy()
    numeric_cols = rounded.select_dtypes(include="number").columns
    rounded[numeric_cols] = rounded[numeric_cols].round(4)
    headers = [str(col) for col in rounded.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in rounded.iterrows():
        values = [str(row[col]) for col in rounded.columns]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_reference_comparison(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["train_precision", "train_recall", "train_f1", "test_precision", "test_recall", "test_f1"]
    for _, result_row in results_df.iterrows():
        for metric in metrics:
            rows.append(
                {
                    "Model": result_row["model"],
                    "Metric": metric,
                    "Reference Reported Result": "Not explicitly reported",
                    "Our Reproduced Result": result_row[metric],
                    "Difference": "N/A",
                }
            )
    return pd.DataFrame(rows)


def run(data_path: str | None = None) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = load_curated_data(data_path)
    df, price_bins = add_reference_targets_and_features(df_raw)
    if df["price_bin"].isna().any():
        missing = int(df["price_bin"].isna().sum())
        raise ValueError(f"{missing} rows did not receive a price_bin under reference logic.")

    df["price_bin"] = df["price_bin"].astype(int)
    y = df["price_bin"].to_numpy()
    X, _ = make_one_hot_matrix(df)
    X_ordinal, _ = make_ordinal_matrix(df)

    bin_boundaries = pd.DataFrame({"boundary_index": range(6), "price_boundary": price_bins})
    bin_summary = price_bin_summary(df)
    reference_rows = run_reference_models(X, X_ordinal, y)
    bagging_rows = run_bagging_experiment(X, y)

    results_df = pd.DataFrame(
        reference_rows + [max(bagging_rows, key=lambda r: r["test_f1"])]
    )
    bagging_df = pd.DataFrame(bagging_rows)
    comparison_df = build_reference_comparison(results_df)

    bin_boundaries.to_csv(RESULTS_DIR / "price_bin_boundaries.csv", index=False)
    bin_summary.to_csv(RESULTS_DIR / "price_bin_summary.csv")
    results_df.to_csv(RESULTS_DIR / "reference_reproduction_results.csv", index=False)
    bagging_df.to_csv(RESULTS_DIR / "bagging_results.csv", index=False)
    comparison_df.to_csv(RESULTS_DIR / "reference_vs_reproduction_comparison.csv", index=False)

    save_markdown_table(results_df, RESULTS_DIR / "reference_reproduction_results.md")
    save_markdown_table(bagging_df, RESULTS_DIR / "bagging_results.md")
    save_markdown_table(comparison_df, RESULTS_DIR / "reference_vs_reproduction_comparison.md")
    save_figures(df, bagging_df)

    print("Price bin boundaries")
    print(bin_boundaries.to_string(index=False))
    print("\nPrice bin summary")
    print(bin_summary.to_string())
    print("\nReference model reproduction results")
    print(results_df.round(4).to_string(index=False))
    print("\nBagging results")
    print(bagging_df.round(4).to_string(index=False))
    print(f"\nSaved results to {RESULTS_DIR}")


def main() -> None:
    parser = ArgumentParser(description="Run the reference Funko Pop ML reproduction.")
    parser.add_argument("--data-path", default=None, help="Optional path to data_curated.csv")
    args = parser.parse_args()
    run(args.data_path)


if __name__ == "__main__":
    main()
