"""
Project-wide settings.

This file stores paths, column names, and modelling choices in ONE place.
If you later change the dataset filename, you only need to edit this file.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project folders
# ---------------------------------------------------------------------------
# config.py lives in: project_root / src / config.py
# .parent is the src folder, .parent.parent is the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASETS_DIR = PROJECT_ROOT / "datasets"
RAW_DATA_DIR = DATASETS_DIR / "raw"
PROCESSED_DATA_DIR = DATASETS_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# ---------------------------------------------------------------------------
# Dataset files
# ---------------------------------------------------------------------------
# Put your Kaggle CSV in datasets/raw/. Common names are Housing.csv or housing.csv.
# Until you add the real file, the project uses the included sample CSV.
KAGGLE_FILENAME = "Housing.csv"
SAMPLE_FILENAME = "sample_housing.csv"

KAGGLE_CSV_PATH = RAW_DATA_DIR / KAGGLE_FILENAME
SAMPLE_CSV_PATH = RAW_DATA_DIR / SAMPLE_FILENAME
PROCESSED_CSV_PATH = PROCESSED_DATA_DIR / "cleaned_housing.csv"

# Other filenames we will also accept if Housing.csv is not used.
ALTERNATE_KAGGLE_FILENAMES = [
    "housing.csv",
    "Housing.csv",
    "Housing_Price_Data.csv",
    "housing_price_data.csv",
]

# ---------------------------------------------------------------------------
# Model artefacts (saved after training so you can predict later)
# ---------------------------------------------------------------------------
MODEL_PATH = MODELS_DIR / "linear_regression_model.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.joblib"

# ---------------------------------------------------------------------------
# Column names from YOUR dataset
# price,area,bedrooms,bathrooms,stories,mainroad,guestroom,basement,
# hotwaterheating,airconditioning,parking,prefarea,furnishingstatus
# ---------------------------------------------------------------------------
TARGET_COLUMN = "price"

# Already numbers in the CSV — the model can use them as-is.
NUMERIC_FEATURES = [
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "parking",
]

# Text columns that are only "yes" or "no". We turn them into 1 and 0.
BINARY_FEATURES = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
]

# Text column with more than two options. We one-hot encode it.
CATEGORICAL_FEATURES = [
    "furnishingstatus",
]

# Every column in this dataset is useful, so we drop none by default.
COLUMNS_TO_DROP: list[str] = []

# Full list used to check that the CSV has the columns we expect.
EXPECTED_COLUMNS = (
    [TARGET_COLUMN] + NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
)

# ---------------------------------------------------------------------------
# Modelling choices
# ---------------------------------------------------------------------------
RANDOM_STATE = 42  # Makes train/test split repeatable (same shuffle every run)
TEST_SIZE = 0.20  # 20% of rows are held out to test the model
IQR_MULTIPLIER = 1.5  # Standard Tukey rule for spotting outliers
