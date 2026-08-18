"""
Score how well the trained model predicts price on unseen houses.

Four metrics are reported (all are standard for regression):

- MAE  (Mean Absolute Error): average |actual - predicted| in price units.
       Easy to explain: "we are typically off by this many currency units."

- MSE  (Mean Squared Error): average of squared errors.
       Punishes large mistakes more than MAE.

- RMSE (Root Mean Squared Error): square root of MSE, back in price units.
       The most common "typical error size" metric.

- R²   (R-squared): fraction of price variation the model explains.
       1.0 is a perfect fit. 0.0 means "no better than predicting the mean."
       Negative means worse than predicting the mean.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(y_true, y_pred) -> dict[str, float]:
    """
    Compute MAE, MSE, RMSE, and R².

    Args:
        y_true: actual prices (test set).
        y_pred: predicted prices.

    Returns:
        dict[str, float]: metric name -> value.

    Raises:
        ValueError: if the two arrays have different lengths or are empty.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("Cannot evaluate: actual or predicted array is empty.")
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Length mismatch: {y_true.shape} actual values vs {y_pred.shape} predictions."
        )

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)

    metrics = {"MAE": float(mae), "MSE": float(mse), "RMSE": rmse, "R2": float(r2)}
    print_metrics(metrics)
    return metrics


def print_metrics(metrics: dict[str, float]) -> None:
    """
    Pretty-print the four scores with a short reminder of what they mean.

    Args:
        metrics: dictionary returned by evaluate_model().
    """
    print("\n--- Model evaluation (test set) ---")
    print(f"MAE  : {metrics['MAE']:,.2f}   (average absolute error in price)")
    print(f"MSE  : {metrics['MSE']:,.2f}   (squared error; larger mistakes count more)")
    print(f"RMSE : {metrics['RMSE']:,.2f}   (typical error size, same unit as price)")
    print(f"R²   : {metrics['R2']:.4f}   (1.0 = perfect, 0.0 = no better than the mean)")
