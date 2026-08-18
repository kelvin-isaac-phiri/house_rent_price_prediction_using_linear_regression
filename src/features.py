"""
Feature selection helpers.

Feature selection means choosing which input columns the model should use.
Using every column is not always best:
- some columns are IDs or dates, not causes of price
- some text columns have too many unique values
- some numbers barely move when price moves (low correlation)

In this housing dataset we keep all columns, then rank them by correlation
so you can see which ones matter most.
"""

import pandas as pd

from src.config import TARGET_COLUMN


def correlation_with_price(df: pd.DataFrame) -> pd.Series:
    """
    Rank numeric columns by how strongly they move with price.

    Pearson correlation is between -1 and +1:
    - close to +1: when the feature goes up, price usually goes up
    - close to -1: when the feature goes up, price usually goes down
    - close to  0: little linear relationship (maybe still useful as a dummy)

    Args:
        df: DataFrame that includes price and numeric features.

    Returns:
        pd.Series: correlations sorted from strongest to weakest (absolute value).
    """
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"'{TARGET_COLUMN}' is required to compute correlations.")

    numeric = df.select_dtypes(include="number")
    if TARGET_COLUMN not in numeric.columns:
        raise ValueError("price must be numeric to compute Pearson correlation.")

    corr = numeric.corr()[TARGET_COLUMN].drop(labels=[TARGET_COLUMN])
    return corr.reindex(corr.abs().sort_values(ascending=False).index)


def print_feature_advice(df: pd.DataFrame) -> None:
    """
    Print a short, readable explanation of which numeric features look useful.

    Args:
        df: cleaned DataFrame (numeric + dummy columns + price).
    """
    ranked = correlation_with_price(df)
    print("--- Feature correlation with price ---")
    print(ranked.to_string())
    print(
        "\nRule of thumb: |correlation| above about 0.3 is a useful linear signal.\n"
        "Yes/no columns (airconditioning, prefarea, ...) and furnishing dummy\n"
        "columns can still matter even when their correlation looks modest,\n"
        "because they are 0/1 switches."
    )
