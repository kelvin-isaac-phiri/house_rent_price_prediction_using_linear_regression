"""
Charts for exploratory data analysis (EDA) and model results.

Every function:
1. Builds a matplotlib figure
2. Saves it under outputs/figures/
3. Returns the figure so a notebook can display it
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, TARGET_COLUMN


def _prepare_save_path(filename: str) -> Path:
    """
    Make sure the figures folder exists and return the full file path.

    Args:
        filename: image name, for example "price_distribution.png".

    Returns:
        Path: where matplotlib should save the image.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR / filename


def plot_price_distribution(df: pd.DataFrame, filename: str = "price_distribution.png"):
    """
    Histogram of house price. Shows whether prices are skewed.

    Args:
        df: DataFrame that still has the original price column.
        filename: output image name.

    Returns:
        matplotlib.figure.Figure: the created figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df[TARGET_COLUMN], bins=30, color="steelblue", edgecolor="white")
    ax.set_title("Distribution of House Price")
    ax.set_xlabel("Price")
    ax.set_ylabel("Number of houses")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(_prepare_save_path(filename), dpi=120)
    return fig


def plot_price_vs_area(df: pd.DataFrame, filename: str = "price_vs_area.png"):
    """
    Scatter plot: area vs price. A rising cloud of points is a good
    sign that area is a useful predictor.

    Args:
        df: DataFrame with area and price columns.
        filename: output image name.

    Returns:
        matplotlib.figure.Figure: the created figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["area"], df[TARGET_COLUMN], alpha=0.45, color="teal")
    ax.set_title("Price vs Area")
    ax.set_xlabel("Area (square feet)")
    ax.set_ylabel("Price")
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(_prepare_save_path(filename), dpi=120)
    return fig


def plot_price_by_furnishing(
    df: pd.DataFrame, filename: str = "price_by_furnishing.png"
):
    """
    Bar chart of average price for each furnishing status.

    Args:
        df: DataFrame with furnishingstatus and price
            (use the data BEFORE one-hot encoding).
        filename: output image name.

    Returns:
        matplotlib.figure.Figure | None: the created figure, or None if the
        column is missing.
    """
    if "furnishingstatus" not in df.columns:
        print("No furnishingstatus column available for this plot.")
        return None

    averages = (
        df.groupby("furnishingstatus")[TARGET_COLUMN].mean().sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(averages.index.astype(str), averages.values, color="coral")
    ax.set_title("Average Price by Furnishing Status")
    ax.set_xlabel("Furnishing status")
    ax.set_ylabel("Average price")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(_prepare_save_path(filename), dpi=120)
    return fig


def plot_price_by_airconditioning(
    df: pd.DataFrame, filename: str = "price_by_airconditioning.png"
):
    """
    Bar chart of average price for houses with and without air conditioning.

    Args:
        df: DataFrame with airconditioning and price (original yes/no text is fine).
        filename: output image name.

    Returns:
        matplotlib.figure.Figure | None: the created figure, or None if missing.
    """
    if "airconditioning" not in df.columns:
        print("No airconditioning column available for this plot.")
        return None

    averages = df.groupby("airconditioning")[TARGET_COLUMN].mean().sort_values()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(averages.index.astype(str), averages.values, color="mediumseagreen")
    ax.set_title("Average Price by Air Conditioning")
    ax.set_xlabel("Air conditioning")
    ax.set_ylabel("Average price")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(_prepare_save_path(filename), dpi=120)
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame, filename: str = "correlation_heatmap.png"
):
    """
    Heatmap of Pearson correlations between numeric columns.

    Values near +1 mean "as X goes up, price goes up".
    Values near -1 mean "as X goes up, price goes down".
    Values near 0 mean little linear relationship.

    Args:
        df: DataFrame with numeric columns.
        filename: output image name.

    Returns:
        matplotlib.figure.Figure: the created figure.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Correlation Heatmap")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)

    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(_prepare_save_path(filename), dpi=120)
    return fig


def plot_actual_vs_predicted(
    y_true, y_pred, filename: str = "actual_vs_predicted.png"
):
    """
    Scatter plot of real price vs model prediction.

    A perfect model would put every point on the diagonal line.

    Args:
        y_true: actual prices from the test set.
        y_pred: prices predicted by the model.
        filename: output image name.

    Returns:
        matplotlib.figure.Figure: the created figure.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_true, y_pred, alpha=0.5, color="mediumpurple")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="black",
        linestyle="--",
        label="Perfect prediction",
    )
    ax.set_title("Actual Price vs Predicted Price")
    ax.set_xlabel("Actual price")
    ax.set_ylabel("Predicted price")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(_prepare_save_path(filename), dpi=120)
    return fig


def plot_residuals(y_true, y_pred, filename: str = "residuals.png"):
    """
    Residual plot: prediction error vs predicted price.

    Residual = actual - predicted.
    For a healthy linear model the cloud of points should look random
    around the zero line, not a curve.

    Args:
        y_true: actual prices.
        y_pred: predicted prices.
        filename: output image name.

    Returns:
        matplotlib.figure.Figure: the created figure.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred, residuals, alpha=0.5, color="darkorange")
    ax.axhline(0, color="black", linestyle="--")
    ax.set_title("Residual Plot")
    ax.set_xlabel("Predicted price")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(_prepare_save_path(filename), dpi=120)
    return fig
