"""
Predict house price for one home or for a table of homes.

New houses must go through the SAME steps as training data:
encode yes/no and furnishingstatus, put columns in the same order, then scale.
Otherwise the model would see a different layout of numbers.
"""

import pandas as pd

from src.model import load_artefacts


def prepare_features_for_prediction(
    features: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    """
    Align a new table with the columns the model was trained on.

    After one-hot encoding, a new house might be missing a dummy column
    (for example furnishingstatus_unfurnished). We add missing columns as 0,
    then keep only the training order.

    Args:
        features: numeric/dummy feature table WITHOUT the price column.
        feature_columns: column order saved during training.

    Returns:
        pd.DataFrame: columns matching the trained model.
    """
    aligned = features.copy()

    for col in feature_columns:
        if col not in aligned.columns:
            aligned[col] = 0

    extra = [col for col in aligned.columns if col not in feature_columns]
    if extra:
        aligned = aligned.drop(columns=extra)

    return aligned[feature_columns]


def predict_price(features: pd.DataFrame) -> pd.Series:
    """
    Predict price for each row in `features`.

    Args:
        features: DataFrame already cleaned/encoded the same way as training
                  (see preprocessing.preprocess_data), without price.

    Returns:
        pd.Series: predicted price for each row.
    """
    if features.empty:
        raise ValueError("Cannot predict: the feature table has 0 rows.")

    model, scaler, feature_columns = load_artefacts()
    aligned = prepare_features_for_prediction(features, feature_columns)
    scaled = scaler.transform(aligned)
    predictions = model.predict(scaled)
    return pd.Series(predictions, index=features.index, name="predicted_price")


def predict_from_raw_row(raw_row: dict) -> float:
    """
    Predict price from the original CSV column names.

    This is the beginner-friendly option. You describe one house the same
    way the CSV does, and this function cleans, encodes, scales, then predicts.

    Example:
        predict_from_raw_row({
            "area": 7420,
            "bedrooms": 4,
            "bathrooms": 2,
            "stories": 3,
            "mainroad": "yes",
            "guestroom": "no",
            "basement": "no",
            "hotwaterheating": "no",
            "airconditioning": "yes",
            "parking": 2,
            "prefarea": "yes",
            "furnishingstatus": "furnished",
        })

    Args:
        raw_row: dictionary using the original dataset column names.

    Returns:
        float: predicted house price.
    """
    from src.config import TARGET_COLUMN
    from src.preprocessing import (
        drop_unused_columns,
        encode_categoricals,
        encode_yes_no_columns,
        select_model_columns,
    )

    frame = pd.DataFrame([raw_row])
    frame = drop_unused_columns(frame)
    frame = encode_yes_no_columns(frame)
    frame = encode_categoricals(frame)
    frame = select_model_columns(frame)
    if TARGET_COLUMN in frame.columns:
        frame = frame.drop(columns=[TARGET_COLUMN])
    for col in frame.columns:
        if pd.api.types.is_integer_dtype(frame[col]):
            frame[col] = frame[col].astype(int)
    return float(predict_price(frame).iloc[0])


def predict_one_house(house: dict) -> float:
    """
    Convenience wrapper: pass a dictionary of already-encoded features.

    Beginners usually find it easier to use predict_from_raw_row() with
    the original yes/no text from the CSV.

    Args:
        house: mapping of feature name -> value (already numeric / dummy).

    Returns:
        float: predicted price.
    """
    frame = pd.DataFrame([house])
    return float(predict_price(frame).iloc[0])
