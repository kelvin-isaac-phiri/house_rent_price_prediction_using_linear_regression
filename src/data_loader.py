"""
Load the housing CSV into a pandas DataFrame.

The loader looks for your Kaggle file first. If it is not there yet,
it falls back to the included sample dataset so you can still practise
the full workflow.
"""

from pathlib import Path

import pandas as pd

from src.config import (
    ALTERNATE_KAGGLE_FILENAMES,
    EXPECTED_COLUMNS,
    KAGGLE_CSV_PATH,
    RAW_DATA_DIR,
    SAMPLE_CSV_PATH,
    SAMPLE_FILENAME,
    TARGET_COLUMN,
)


def _csv_has_price_column(path: Path) -> bool:
    """
    Peek at a CSV header without loading the whole file.

    Args:
        path: CSV file to inspect.

    Returns:
        bool: True if a column named price exists (after stripping spaces).
    """
    try:
        header = pd.read_csv(path, nrows=0)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        return False
    columns = [str(name).strip() for name in header.columns]
    return TARGET_COLUMN in columns


def find_dataset_path() -> Path:
    """
    Decide which CSV file to use.

    Search order:
    1. datasets/raw/Housing.csv (the usual Kaggle name)
    2. other common names listed in config.py
    3. any other CSV in datasets/raw/ that has a `price` column
    4. the bundled sample file

    Returns:
        Path: location of the chosen CSV.

    Raises:
        FileNotFoundError: if no usable file is present.
    """
    if KAGGLE_CSV_PATH.exists():
        return KAGGLE_CSV_PATH

    for name in ALTERNATE_KAGGLE_FILENAMES:
        candidate = RAW_DATA_DIR / name
        if candidate.exists() and candidate.resolve() != SAMPLE_CSV_PATH.resolve():
            return candidate

    if RAW_DATA_DIR.exists():
        for candidate in sorted(RAW_DATA_DIR.glob("*.csv")):
            if candidate.name == SAMPLE_FILENAME:
                continue
            if _csv_has_price_column(candidate):
                print(f"Using dataset file: {candidate.name}")
                return candidate

    if SAMPLE_CSV_PATH.exists():
        print(
            "Kaggle dataset not found in datasets/raw/.\n"
            "Using the bundled sample dataset instead so you can still run the project.\n"
            "When you download your CSV, place it in datasets/raw/ "
            "(Housing.csv is a good filename)."
        )
        return SAMPLE_CSV_PATH

    raise FileNotFoundError(
        "No dataset found.\n"
        f"Add your Kaggle file here: {KAGGLE_CSV_PATH}\n"
        f"Or keep the sample file here: {SAMPLE_CSV_PATH}\n"
        "See datasets/README.md for download instructions."
    )


def _normalise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip accidental spaces from header names, e.g. ' area' -> 'area'.

    Args:
        df: freshly loaded table.

    Returns:
        pd.DataFrame: same table with tidy column names.
    """
    cleaned = df.copy()
    cleaned.columns = [str(name).strip() for name in cleaned.columns]
    return cleaned


def load_dataset(csv_path: Path | None = None) -> pd.DataFrame:
    """
    Read the CSV into a pandas DataFrame and run basic safety checks.

    Args:
        csv_path: Optional custom path. If omitted, find_dataset_path() is used.

    Returns:
        pd.DataFrame: the raw (not yet cleaned) housing table.

    Raises:
        FileNotFoundError: the file does not exist.
        ValueError: the file is empty or required columns are missing.
    """
    path = Path(csv_path) if csv_path is not None else find_dataset_path()

    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"The CSV file is empty: {path}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(
            f"Could not parse {path}. Check that it is a valid CSV file."
        ) from exc

    if df.empty:
        raise ValueError(f"The CSV loaded successfully but contains 0 rows: {path}")

    df = _normalise_column_names(df)

    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "This CSV does not have the columns this project expects.\n"
            f"Missing: {missing}\n"
            f"Columns found: {list(df.columns)}\n"
            "Expected: "
            "price, area, bedrooms, bathrooms, stories, mainroad, guestroom, "
            "basement, hotwaterheating, airconditioning, parking, prefarea, "
            "furnishingstatus"
        )

    print(f"Loaded {len(df)} rows and {len(df.columns)} columns from {path.name}")
    return df


def dataset_overview(df: pd.DataFrame) -> None:
    """
    Print a beginner-friendly summary of the table.

    This is not a return-a-value function. It prints so you can read the
    output in the notebook or terminal.

    Args:
        df: the DataFrame to describe.
    """
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Rows (houses):     {df.shape[0]}")
    print(f"Columns (fields):  {df.shape[1]}")
    print("\nColumn names and data types:")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isna().sum())
    print("\nFirst 5 rows:")
    print(df.head())
    print("=" * 60)
