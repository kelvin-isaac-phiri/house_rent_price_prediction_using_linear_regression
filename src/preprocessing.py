"""
Clean and prepare the housing table for linear regression.

Your CSV mixes two kinds of columns:

- numbers already: price, area, bedrooms, bathrooms, stories, parking
- text that a math model cannot use yet:
    - yes/no flags: mainroad, guestroom, basement, hotwaterheating,
      airconditioning, prefarea
    - three-way label: furnishingstatus (furnished / semi-furnished / unfurnished)

This module turns the text into numbers, fills gaps, and removes extreme prices.
"""

import numpy as np
import pandas as pd

from src.config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    COLUMNS_TO_DROP,
    IQR_MULTIPLIER,
    NUMERIC_FEATURES,
    PROCESSED_CSV_PATH,
    TARGET_COLUMN,
)

# Map every common spelling of yes/no onto 1 and 0.
YES_NO_MAP = {
    "yes": 1,
    "no": 0,
    "true": 1,
    "false": 0,
    "1": 1,
    "0": 0,
    "y": 1,
    "n": 0,
}


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns listed in COLUMNS_TO_DROP (none, for this dataset).

    Args:
        df: DataFrame to clean.

    Returns:
        pd.DataFrame: a copy, possibly with fewer columns.
    """
    cleaned = df.copy()
    existing = [col for col in COLUMNS_TO_DROP if col in cleaned.columns]
    if existing:
        cleaned = cleaned.drop(columns=existing)
        print(f"Dropped unused columns: {existing}")
    return cleaned


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop exact duplicate rows.

    Duplicate houses would give those rows extra weight during training.

    Args:
        df: DataFrame to clean.

    Returns:
        pd.DataFrame: a copy with duplicates removed.
    """
    before = len(df)
    cleaned = df.drop_duplicates()
    removed = before - len(cleaned)
    print(f"Removed {removed} duplicate row(s).")
    return cleaned


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill or drop missing values.

    Strategy used here (simple and beginner-friendly):
    - numeric columns: fill with the median (middle value, robust to outliers)
    - text columns: fill with the mode (most common category)
    - if price is missing, drop the row (we cannot train without it)

    Args:
        df: DataFrame that may contain NaN values.

    Returns:
        pd.DataFrame: a copy with missing values handled.
    """
    cleaned = df.copy()

    if TARGET_COLUMN in cleaned.columns:
        missing_target = cleaned[TARGET_COLUMN].isna().sum()
        cleaned = cleaned.dropna(subset=[TARGET_COLUMN])
        print(f"Dropped {missing_target} row(s) with missing {TARGET_COLUMN}.")

    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        missing = cleaned[col].isna().sum()
        if missing:
            median_value = cleaned[col].median()
            cleaned[col] = cleaned[col].fillna(median_value)
            print(f"Filled {missing} missing value(s) in '{col}' with median {median_value}.")

    categorical_cols = cleaned.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        missing = cleaned[col].isna().sum()
        if missing:
            if cleaned[col].mode().empty:
                raise ValueError(f"Column '{col}' is empty, so a fill value cannot be chosen.")
            mode_value = cleaned[col].mode().iloc[0]
            cleaned[col] = cleaned[col].fillna(mode_value)
            print(f"Filled {missing} missing value(s) in '{col}' with mode '{mode_value}'.")

    return cleaned


def remove_price_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove extreme price values using the IQR (Interquartile Range) rule.

    Why: a few luxury houses with huge prices can tilt the regression line
    and make predictions worse for typical homes.

    IQR rule: keep rows whose price is between
    Q1 - 1.5 * IQR and Q3 + 1.5 * IQR.

    Args:
        df: DataFrame with a price column.

    Returns:
        pd.DataFrame: a copy without extreme prices.
    """
    cleaned = df.copy()
    q1 = cleaned[TARGET_COLUMN].quantile(0.25)
    q3 = cleaned[TARGET_COLUMN].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - IQR_MULTIPLIER * iqr
    upper = q3 + IQR_MULTIPLIER * iqr

    before = len(cleaned)
    cleaned = cleaned[(cleaned[TARGET_COLUMN] >= lower) & (cleaned[TARGET_COLUMN] <= upper)]
    print(
        f"Removed {before - len(cleaned)} outlier row(s) "
        f"with {TARGET_COLUMN} outside [{lower:.0f}, {upper:.0f}]."
    )
    return cleaned


def encode_yes_no_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert yes/no text into 1 and 0.

    Linear Regression can only multiply numbers. After this step,
    airconditioning=yes becomes 1, airconditioning=no becomes 0.

    Args:
        df: DataFrame that still has yes/no text.

    Returns:
        pd.DataFrame: a copy with binary columns as integers.

    Raises:
        ValueError: if a binary column contains an unexpected word.
    """
    cleaned = df.copy()
    for col in BINARY_FEATURES:
        if col not in cleaned.columns:
            continue
        as_text = cleaned[col].astype(str).str.strip().str.lower()
        mapped = as_text.map(YES_NO_MAP)
        unknown = as_text[mapped.isna() & as_text.ne("nan")]
        if not unknown.empty:
            bad_values = sorted(unknown.unique())
            raise ValueError(
                f"Column '{col}' has values the yes/no mapper does not understand: {bad_values}. "
                "Expected yes or no."
            )
        cleaned[col] = mapped.astype("Int64")
        print(f"Encoded '{col}' as 1 (yes) / 0 (no).")
    return cleaned


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert furnishingstatus into 0/1 dummy columns.

    Example:
        furnished / semi-furnished / unfurnished
        becomes two columns (drop_first=True avoids a redundant third column):
        furnishingstatus_semi-furnished, furnishingstatus_unfurnished

    drop_first=True avoids the dummy-variable trap (perfect multicollinearity),
    which would confuse the linear model.

    Args:
        df: DataFrame with categorical columns.

    Returns:
        pd.DataFrame: numeric-only table ready for sklearn.
    """
    cleaned = df.copy()
    cols = [col for col in CATEGORICAL_FEATURES if col in cleaned.columns]
    if not cols:
        print("No categorical columns to encode.")
        return cleaned

    cleaned = pd.get_dummies(cleaned, columns=cols, drop_first=True, dtype=int)
    print(f"One-hot encoded: {cols}")
    return cleaned


def select_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep the target plus numeric, binary, and dummy columns.

    Args:
        df: fully encoded DataFrame.

    Returns:
        pd.DataFrame: only the columns the model should see.
    """
    available_numeric = [
        col for col in NUMERIC_FEATURES + BINARY_FEATURES if col in df.columns
    ]
    dummy_cols = [
        col
        for col in df.columns
        if col not in available_numeric
        and col != TARGET_COLUMN
        and df[col].dtype != "object"
    ]
    keep = available_numeric + dummy_cols
    if TARGET_COLUMN in df.columns:
        keep = keep + [TARGET_COLUMN]
    return df[keep]


def preprocess_data(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Run the full cleaning pipeline in a sensible order.

    Args:
        df: raw DataFrame from data_loader.load_dataset().
        save: if True, write the cleaned table to datasets/processed/.

    Returns:
        pd.DataFrame: cleaned, encoded table ready for train/test split.
    """
    print("\n--- Starting preprocessing ---")
    cleaned = drop_unused_columns(df)
    cleaned = remove_duplicates(cleaned)
    cleaned = handle_missing_values(cleaned)
    cleaned = remove_price_outliers(cleaned)
    cleaned = encode_yes_no_columns(cleaned)
    cleaned = encode_categoricals(cleaned)
    cleaned = select_model_columns(cleaned)

    # Int64 (nullable integers) can confuse sklearn; use plain ints/floats.
    for col in cleaned.columns:
        if pd.api.types.is_integer_dtype(cleaned[col]):
            cleaned[col] = cleaned[col].astype(int)

    if save:
        PROCESSED_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(PROCESSED_CSV_PATH, index=False)
        print(f"Saved cleaned data to {PROCESSED_CSV_PATH}")

    print(f"Preprocessing complete. Shape: {cleaned.shape}")
    return cleaned
