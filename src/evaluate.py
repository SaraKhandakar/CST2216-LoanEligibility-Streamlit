from sklearn.metrics import accuracy_score, confusion_matrix

def eval_model(model, X, y, threshold=None):
    # If threshold provided, use predict_proba
    if threshold is not None and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= threshold).astype(int)
    else:
        pred = model.predict(X)

    acc = accuracy_score(y, pred)
    cm = confusion_matrix(y, pred)
    return float(acc), cm, pred