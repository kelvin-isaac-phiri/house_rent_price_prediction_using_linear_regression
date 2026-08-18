"""
Run the full training pipeline from the command line:

    python -m src.pipeline

This is the same sequence as the notebook, packed into one script:
load -> clean -> split -> scale -> train -> evaluate -> save -> plot.
"""

# Use a non-interactive backend so charts save even without a window
# (this must happen before matplotlib.pyplot is imported via visualization).
import matplotlib

matplotlib.use("Agg")

from src.data_loader import dataset_overview, load_dataset
from src.evaluation import evaluate_model
from src.model import (
    load_artefacts,
    make_train_test_split,
    save_artefacts,
    scale_features,
    split_features_and_target,
    train_linear_regression,
)
from src.preprocessing import preprocess_data
from src.visualization import plot_actual_vs_predicted, plot_residuals


def run_pipeline() -> dict[str, float]:
    """
    Train Linear Regression on the available dataset and report test metrics.

    Returns:
        dict[str, float]: MAE, MSE, RMSE, and R² on the test set.
    """
    raw_df = load_dataset()
    dataset_overview(raw_df)

    clean_df = preprocess_data(raw_df, save=True)
    X, y = split_features_and_target(clean_df)
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    model = train_linear_regression(X_train_scaled, y_train)
    save_artefacts(model, scaler, X_train.columns)

    predictions = model.predict(X_test_scaled)
    metrics = evaluate_model(y_test, predictions)

    plot_actual_vs_predicted(y_test, predictions)
    plot_residuals(y_test, predictions)
    print("Saved evaluation plots in outputs/figures/")
    return metrics


if __name__ == "__main__":
    run_pipeline()
    # Quick sanity check that saved files can be loaded again.
    load_artefacts()
    print("Pipeline finished successfully.")
