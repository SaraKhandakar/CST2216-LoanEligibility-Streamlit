# =========================
# Prediction Module
# =========================
# This file handles preparing user input and generating predictions
# using the trained loan eligibility model.

import pandas as pd


def build_input_row(raw_df, user_inputs: dict) -> pd.DataFrame:
    """
    Convert user input dictionary into a DataFrame.

    Parameters:
    raw_df (DataFrame): Original dataset (used for reference if needed)
    user_inputs (dict): User-provided feature values

    Returns:
    pd.DataFrame: Single-row DataFrame ready for prediction

    Purpose:
    Converts user input into the tabular format required by the model.
    """
    # Convert dictionary into a single-row DataFrame
    return pd.DataFrame([user_inputs])


def predict_with_model(model, X, threshold=None):
    """
    Generate prediction using trained model.

    Parameters:
    model: Trained classification model
    X: Input features for prediction
    threshold (float, optional): Custom probability threshold

    Returns:
    tuple:
        int: Predicted class (0 or 1)
        float or None: Probability of positive class (if available)

    Purpose:
    - Supports both default prediction and threshold-based prediction
    - Returns both predicted class and probability (if available)
    """

    # =========================
    # Threshold-Based Prediction
    # =========================
    # If threshold is provided and model supports probabilities,
    # classify based on custom threshold
    if threshold is not None and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= threshold).astype(int)
        return int(pred[0]), float(proba[0])

    else:
        # =========================
        # Default Prediction
        # =========================
        pred = model.predict(X)

        # Try to extract probability if model supports it
        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X)[0][1])

        return int(pred[0]), proba