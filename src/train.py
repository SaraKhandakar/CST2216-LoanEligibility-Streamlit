import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def split_scale_train_all(df, target_col, test_size, random_state, rf_tuned_params, logger):
    x = df.drop(target_col, axis=1)
    y = df[target_col]
    
    logger.info(f"Target dtype before split: {y.dtype}")
    logger.info(f"Target unique values before split: {y.unique()}")

    logger.info(f"Training dataframe shape: {df.shape}")
    logger.info(f"NaNs in full X before split: {x.isna().sum().sum()}")
    logger.info(f"NaNs in full y before split: {y.isna().sum()}")

    xtrain, xtest, ytrain, ytest = train_test_split(
        x, y, test_size=test_size, stratify=y, random_state=random_state
    )

    logger.info(f"NaNs in xtrain before scaling: {xtrain.isna().sum().sum()}")
    logger.info(f"NaNs in xtest before scaling: {xtest.isna().sum().sum()}")
    logger.info(f"NaNs in ytrain: {ytrain.isna().sum()}")
    logger.info(f"NaNs in ytest: {ytest.isna().sum()}")

    # MinMaxScaler exactly like notebook
    scaler = MinMaxScaler()
    xtrain_scaled = scaler.fit_transform(xtrain)
    xtest_scaled = scaler.transform(xtest)

    logger.info(f"NaNs in xtrain_scaled: {np.isnan(xtrain_scaled).sum()}")
    logger.info(f"NaNs in xtest_scaled: {np.isnan(xtest_scaled).sum()}")
    logger.info(f"Infs in xtrain_scaled: {np.isinf(xtrain_scaled).sum()}")
    logger.info(f"Infs in xtest_scaled: {np.isinf(xtest_scaled).sum()}")

    # Final safety check
    if np.isnan(xtrain_scaled).sum() > 0 or np.isnan(xtest_scaled).sum() > 0:
        raise ValueError("NaN values still exist after preprocessing/scaling.")

    logger.info("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000).fit(xtrain_scaled, ytrain)

    logger.info("Training Decision Tree...")
    dt = DecisionTreeClassifier().fit(xtrain_scaled, ytrain)

    logger.info("Training Random Forest (default)...")
    rf_default = RandomForestClassifier().fit(xtrain_scaled, ytrain)

    logger.info(f"Training Random Forest (tuned params={rf_tuned_params}) on unscaled data...")
    rf_tuned = RandomForestClassifier(**rf_tuned_params).fit(xtrain, ytrain)

    return {
        "xtrain": xtrain, "xtest": xtest,
        "ytrain": ytrain, "ytest": ytest,
        "xtrain_scaled": xtrain_scaled, "xtest_scaled": xtest_scaled,
        "scaler": scaler,
        "models": {
            "Logistic Regression (scaled)": lr,
            "Decision Tree (scaled)": dt,
            "Random Forest Default (scaled)": rf_default,
            "Random Forest Tuned (unscaled)": rf_tuned,
        }
    }

def save_artifacts(artifacts, path, logger):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifacts, path)
        logger.info(f"Saved artifacts to: {path}")
    except Exception:
        logger.exception("Failed to save artifacts.")
        raise

def load_artifacts(path, logger):
    try:
        logger.info(f"Loading artifacts from: {path}")
        return joblib.load(path)
    except Exception:
        logger.exception("Failed to load artifacts.")
        raise