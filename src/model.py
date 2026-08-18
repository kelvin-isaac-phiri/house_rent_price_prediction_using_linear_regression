"""
Train a Linear Regression model and save it to disk.

Linear Regression finds a line (in many dimensions) of the form:

    price = intercept + (w1 * area) + (w2 * bedrooms) + ...

The training step chooses the weights w1, w2, ... that minimise the
average squared error on the training houses.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    FEATURE_COLUMNS_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    SCALER_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
)


def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate input columns (X) from the value we want to predict (y).

    Args:
        df: cleaned DataFrame that includes the price column.

    Returns:
        tuple: (X, y) where X is features and y is price.

    Raises:
        KeyError: if the target column is missing.
    """
    if TARGET_COLUMN not in df.columns:
        raise KeyError(
            f"Cannot split features: '{TARGET_COLUMN}' is not in the DataFrame. "
            f"Columns: {list(df.columns)}"
        )

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def make_train_test_split(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Hold out part of the data so we can test on houses the model has not seen.

    If we trained AND scored on the same rows, a model could "memorise"
    the answers and look better than it really is. The test set is our
    honest exam.

    Args:
        X: feature table.
        y: price values.

    Returns:
        tuple: X_train, X_test, y_train, y_test.
    """
    if len(X) < 10:
        raise ValueError("Not enough rows to split into train and test sets.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    print(
        f"Train set: {len(X_train)} rows ({100 * (1 - TEST_SIZE):.0f}%) | "
        f"Test set: {len(X_test)} rows ({100 * TEST_SIZE:.0f}%)"
    )
    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Standardise numeric columns so they have mean 0 and standard deviation 1.

    area might be 2000-8000 while bedrooms is 1-6. Without scaling, the large
    numbers can dominate. Scaling does not change the model's accuracy
    for ordinary linear regression much, but it makes coefficients easier
    to compare and is a good habit.

    Important: fit the scaler on TRAIN data only, then transform test data.
    Fitting on the test set would leak information from the exam into training.

    Args:
        X_train: training features.
        X_test: test features.

    Returns:
        tuple: scaled X_train, scaled X_test, and the fitted scaler.
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    return X_train_scaled, X_test_scaled, scaler


def train_linear_regression(X_train, y_train) -> LinearRegression:
    """
    Fit sklearn's LinearRegression on the training houses.

    Args:
        X_train: scaled training features.
        y_train: training prices.

    Returns:
        LinearRegression: the fitted model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Linear Regression model trained.")
    return model


def save_artefacts(model, scaler, feature_columns) -> None:
    """
    Write the model, scaler, and column order to the models/ folder.

    We need all three later: the model to predict, the scaler to transform
    new houses the same way, and the column list so new data has the same
    dummy columns in the same order.

    Args:
        model: trained LinearRegression.
        scaler: fitted StandardScaler.
        feature_columns: list/Index of column names used during training.
    """
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(list(feature_columns), FEATURE_COLUMNS_PATH)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved scaler to {SCALER_PATH}")
    print(f"Saved feature column order to {FEATURE_COLUMNS_PATH}")


def load_artefacts():
    """
    Load a previously saved model, scaler, and feature column list.

    Returns:
        tuple: (model, scaler, feature_columns)

    Raises:
        FileNotFoundError: if training has not been run yet.
    """
    for path in (MODEL_PATH, SCALER_PATH, FEATURE_COLUMNS_PATH):
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Missing {path}. Train the model first "
                "(run the notebook or `python -m src.pipeline`)."
            )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    return model, scaler, feature_columns
