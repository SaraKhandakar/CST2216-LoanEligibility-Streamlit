import pandas as pd

def build_input_row(raw_df, user_inputs: dict) -> pd.DataFrame:
    # Convert dict to a single-row dataframe
    return pd.DataFrame([user_inputs])

def predict_with_model(model, X, threshold=None):
    if threshold is not None and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= threshold).astype(int)
        return int(pred[0]), float(proba[0])
    else:
        pred = model.predict(X)
        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X)[0][1])
        return int(pred[0]), proba