# =========================
# Model Evaluation Module
# =========================
# This file contains functions used to evaluate the performance
# of the loan eligibility classification model.

from sklearn.metrics import accuracy_score, confusion_matrix


def eval_model(model, X, y, threshold=None):
    """
    Evaluate a classification model using accuracy and confusion matrix.

    Parameters:
    model: Trained classification model
    X: Feature set used for evaluation
    y: True target labels
    threshold (float, optional): Probability threshold for converting
        predicted probabilities into class labels

    Returns:
    tuple:
        float: Accuracy score
        cm: Confusion matrix
        pred: Predicted class labels

    Purpose:
    This function evaluates how well the trained model performs.
    It supports both default class prediction and threshold-based
    prediction using probabilities.
    """

    # =========================
    # Generate Predictions
    # =========================
    # If a threshold is provided and the model supports probability output,
    # use predict_proba to classify observations based on the custom threshold
    if threshold is not None and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= threshold).astype(int)
    else:
        # Otherwise, use the model's default prediction method
        pred = model.predict(X)

    # =========================
    # Compute Evaluation Metrics
    # =========================
    # Accuracy measures the proportion of correct predictions
    acc = accuracy_score(y, pred)

    # Confusion matrix shows correct and incorrect classifications
    cm = confusion_matrix(y, pred)

    return float(acc), cm, pred