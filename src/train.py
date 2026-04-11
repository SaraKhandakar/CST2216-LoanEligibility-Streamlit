# =========================
# Model Training and Artifact Management Module
# =========================
# This file contains functions for splitting data, scaling features,
# training multiple classification models, and saving/loading trained artifacts.

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def split_scale_train_all(df, target_col, test_size, random_state, rf_tuned_params, logger):
    """
    Split dataset, scale features where needed, and train multiple models.

    Parameters:
    df (DataFrame): Final processed dataset
    target_col (str): Name of the target column
    test_size (float): Proportion of data used for testing
    random_state (int): Random seed for reproducibility
    rf_tuned_params (dict): Tuned hyperparameters for Random Forest
    logger: Logger object for tracking execution

    Returns:
    dict: Dictionary containing train/test splits, scaled data,
          scaler object, and trained models

    Purpose:
    - Separate features and target
    - Split dataset into training and testing sets
    - Scale data for models that require normalized input
    - Train multiple models for comparison
    - Keep all training artifacts together for reuse
    """

    # =========================
    # Separate Features and Target
    # =========================
    x = df.drop(target_col, axis=1)
    y = df[target_col]

    # Log target information before splitting
    logger.info(f"Target dtype before split: {y.dtype}")
    logger.info(f"Target unique values before split: {y.unique()}")

    # Log dataset shape and missing-value status
    logger.info(f"Training dataframe shape: {df.shape}")
    logger.info(f"NaNs in full X before split: {x.isna().sum().sum()}")
    logger.info(f"NaNs in full y before split: {y.isna().sum()}")

    # =========================
    # Train-Test Split
    # =========================
    # stratify=y preserves class balance in both training and test sets
    xtrain, xtest, ytrain, ytest = train_test_split(
        x,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # Log missing values after splitting
    logger.info(f"NaNs in xtrain before scaling: {xtrain.isna().sum().sum()}")
    logger.info(f"NaNs in xtest before scaling: {xtest.isna().sum().sum()}")
    logger.info(f"NaNs in ytrain: {ytrain.isna().sum()}")
    logger.info(f"NaNs in ytest: {ytest.isna().sum()}")

    # =========================
    # Feature Scaling
    # =========================
    # MinMaxScaler normalizes feature values into a similar range
    # This is helpful for models such as Logistic Regression
    scaler = MinMaxScaler()
    xtrain_scaled = scaler.fit_transform(xtrain)
    xtest_scaled = scaler.transform(xtest)

    # Log any invalid values after scaling
    logger.info(f"NaNs in xtrain_scaled: {np.isnan(xtrain_scaled).sum()}")
    logger.info(f"NaNs in xtest_scaled: {np.isnan(xtest_scaled).sum()}")
    logger.info(f"Infs in xtrain_scaled: {np.isinf(xtrain_scaled).sum()}")
    logger.info(f"Infs in xtest_scaled: {np.isinf(xtest_scaled).sum()}")

    # Final validation to ensure scaled data is clean
    if np.isnan(xtrain_scaled).sum() > 0 or np.isnan(xtest_scaled).sum() > 0:
        raise ValueError("NaN values still exist after preprocessing/scaling.")

    # =========================
    # Train Logistic Regression
    # =========================
    # Logistic Regression is trained on scaled data
    logger.info("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000).fit(xtrain_scaled, ytrain)

    # =========================
    # Train Decision Tree
    # =========================
    # Decision Tree is also trained here on scaled data
    logger.info("Training Decision Tree...")
    dt = DecisionTreeClassifier().fit(xtrain_scaled, ytrain)

    # =========================
    # Train Random Forest (Default)
    # =========================
    # Default Random Forest model for baseline comparison
    logger.info("Training Random Forest (default)...")
    rf_default = RandomForestClassifier().fit(xtrain_scaled, ytrain)

    # =========================
    # Train Random Forest (Tuned)
    # =========================
    # Tuned Random Forest is trained on unscaled data to match notebook logic
    logger.info(f"Training Random Forest (tuned params={rf_tuned_params}) on unscaled data...")
    rf_tuned = RandomForestClassifier(**rf_tuned_params).fit(xtrain, ytrain)

    # =========================
    # Return Training Artifacts
    # =========================
    return {
        "xtrain": xtrain,
        "xtest": xtest,
        "ytrain": ytrain,
        "ytest": ytest,
        "xtrain_scaled": xtrain_scaled,
        "xtest_scaled": xtest_scaled,
        "scaler": scaler,
        "models": {
            "Logistic Regression (scaled)": lr,
            "Decision Tree (scaled)": dt,
            "Random Forest Default (scaled)": rf_default,
            "Random Forest Tuned (unscaled)": rf_tuned,
        }
    }


def save_artifacts(artifacts, path, logger):
    """
    Save trained artifacts to disk.

    Parameters:
    artifacts (dict): Dictionary containing trained models and preprocessing objects
    path (Path): File path to save artifacts
    logger: Logger object for tracking execution

    Purpose:
    Saves models, scaler, and other training outputs so they can
    be reused later without retraining.
    """
    try:
        # Create folder if it does not exist
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save artifacts using joblib
        joblib.dump(artifacts, path)

        logger.info(f"Saved artifacts to: {path}")

    except Exception:
        logger.exception("Failed to save artifacts.")
        raise


def load_artifacts(path, logger):
    """
    Load saved training artifacts from disk.

    Parameters:
    path (Path): File path of saved artifacts
    logger: Logger object for tracking execution

    Returns:
    Loaded artifacts object

    Purpose:
    Retrieves previously saved models and preprocessing objects
    for prediction or evaluation.
    """
    try:
        logger.info(f"Loading artifacts from: {path}")
        return joblib.load(path)

    except Exception:
        logger.exception("Failed to load artifacts.")
        raise