"""Reference preprocessing for the Funko Pop reproduction."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


PRICE_BIN_LABELS = [1, 2, 3, 4, 5]
NUMERIC_FEATURES = ["search_volume_normalized"]
CATEGORICAL_FEATURES = ["category", "material", "special_version"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def default_data_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "CS229-master"
        / "project"
        / "src"
        / "data_sourcing"
        / "data_curated.csv"
    )


def load_curated_data(data_path: Path | str | None = None) -> pd.DataFrame:
    path = Path(data_path) if data_path is not None else default_data_path()
    return pd.read_csv(path)


def add_reference_targets_and_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    df = df.copy()
    prices = np.array(df["average_sale_price"]).reshape(-1, 1)
    price_bins = (
        np.exp(np.log(np.max(prices) / np.min(prices)) / 5 * np.array([0, 1, 2, 3, 4, 5]))
        * np.min(prices)
    )

    def get_price_bin(avg_price: float) -> int | None:
        if avg_price >= price_bins[0] and avg_price < price_bins[1]:
            return 1
        if avg_price >= price_bins[1] and avg_price < price_bins[2]:
            return 2
        if avg_price >= price_bins[2] and avg_price < price_bins[3]:
            return 3
        if avg_price >= price_bins[3] and avg_price < price_bins[4]:
            return 4
        if avg_price >= price_bins[4] and avg_price < price_bins[5]:
            return 5
        return None

    df["price_bin"] = df["average_sale_price"].apply(get_price_bin)

    log_search_volume = np.log(df["search_volume"])
    log_vol_max = np.max(log_search_volume)
    log_vol_min = np.min(log_search_volume)
    df["search_volume_normalized"] = (log_search_volume - log_vol_min) / (log_vol_max - log_vol_min)
    return df, price_bins


def price_bin_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("price_bin", as_index=True)["average_sale_price"]
        .agg(["count", "min", "max", "mean", "std"])
        .reindex(PRICE_BIN_LABELS)
    )


def make_one_hot_matrix(df: pd.DataFrame):
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor.fit_transform(df[MODEL_FEATURES]), preprocessor


def make_ordinal_matrix(df: pd.DataFrame):
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(steps=[("ordinal", OrdinalEncoder())])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor.fit_transform(df[MODEL_FEATURES]), preprocessor
