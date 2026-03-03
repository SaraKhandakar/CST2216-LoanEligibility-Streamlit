import streamlit as st
import pandas as pd

from config import (
    DATA_PATH, ARTIFACT_PATH, LOG_PATH,
    TARGET_COL, CATEGORICAL_DUMMY_COLS,
    TEST_SIZE, RANDOM_STATE, RF_TUNED_PARAMS
)
from src.utils import setup_logger
from src.data_loader import load_data
from src.preprocessing import preprocess_like_notebook, make_dummies_and_target
from src.train import split_scale_train_all, save_artifacts, load_artifacts
from src.evaluate import eval_model
from src.predict import build_input_row, predict_with_model

logger = setup_logger(LOG_PATH)

st.set_page_config(page_title="Loan Eligibility - Multi Model", layout="wide")

# ---------- Header ----------
st.title("🏦 Loan Eligibility Predictor (Multi-Model)")
st.caption(
    "Multiple machine learning models trained and evaluated on loan data."
)

# ---------- Load / Train ----------
@st.cache_resource
def get_artifacts():
    if ARTIFACT_PATH.exists():
        return load_artifacts(ARTIFACT_PATH, logger)

    raw = load_data(DATA_PATH, logger)
    cleaned = preprocess_like_notebook(raw, logger)
    processed = make_dummies_and_target(cleaned, CATEGORICAL_DUMMY_COLS, TARGET_COL, logger)

    artifacts = split_scale_train_all(
        processed, TARGET_COL, TEST_SIZE, RANDOM_STATE, RF_TUNED_PARAMS, logger
    )
    artifacts["processed_columns"] = processed.drop(TARGET_COL, axis=1).columns.tolist()

    save_artifacts(artifacts, ARTIFACT_PATH, logger)
    return artifacts

art = get_artifacts()

models = art["models"]
xtest = art["xtest"]
ytest = art["ytest"]
xtest_scaled = art["xtest_scaled"]
scaler = art["scaler"]
processed_cols = art["processed_columns"]

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Settings")

model_name = st.sidebar.selectbox("Choose a model", list(models.keys()))

threshold = None
if "Logistic Regression" in model_name:
    st.sidebar.caption("Optional: apply a probability cutoff (like notebook)")
    threshold = st.sidebar.slider("Approval threshold", 0.10, 0.90, 0.50, 0.05)

st.sidebar.divider()
st.sidebar.subheader("🧾 Applicant Inputs")

# Build inputs from ORIGINAL cleaned data (human-readable)
raw = load_data(DATA_PATH, logger)
cleaned = preprocess_like_notebook(raw, logger)
feature_cols = [c for c in cleaned.columns if c not in [TARGET_COL, "Loan_ID"]]

user_inputs = {}
for col in feature_cols:
    if cleaned[col].dtype == object:
        user_inputs[col] = st.sidebar.selectbox(
            col,
            sorted(cleaned[col].dropna().unique().tolist())
        )
    else:
        user_inputs[col] = st.sidebar.number_input(
            col,
            value=float(cleaned[col].median())
        )

predict_clicked = st.sidebar.button("🔮 Predict")

# ---------- Main Layout ----------
left, right = st.columns([1.2, 1])

with left:
    st.subheader("📊 Test Set Results (All Models)")

    rows = []
    cms = {}
    for name, model in models.items():
        if "unscaled" in name:
            acc, cm, _ = eval_model(model, xtest, ytest)
        else:
            acc, cm, _ = eval_model(model, xtest_scaled, ytest)
        rows.append({"Model": name, "Accuracy": acc})
        cms[name] = cm

    results_df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False)
    st.dataframe(results_df, use_container_width=True)

    st.caption("Confusion matrix shown for the selected model (below).")

    st.subheader("🧩 Confusion Matrix (Selected Model)")
    cm = cms[model_name]
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
    st.table(cm_df)

with right:
    st.subheader("✅ Prediction Output")

    st.write(f"**Selected model:** {model_name}")
    if threshold is not None:
        st.write(f"**Threshold:** {threshold:.2f}")

    if predict_clicked:
        # Convert single row into same dummy-columns as training
        single = build_input_row(cleaned, user_inputs)

        # Add a dummy target column just so we can reuse the same function,
        # then immediately drop it.
        single = make_dummies_and_target(
            single.assign(**{TARGET_COL: "N"}),
            CATEGORICAL_DUMMY_COLS,
            TARGET_COL,
            logger
        )
        single_X = single.drop(TARGET_COL, axis=1)

        # Align columns exactly to training feature set
        single_X = single_X.reindex(columns=processed_cols, fill_value=0)

        model = models[model_name]

        # Scaled models use scaler; tuned RF uses unscaled
        if "unscaled" in model_name:
            X_for_model = single_X
        else:
            X_for_model = scaler.transform(single_X)

        pred, proba = predict_with_model(model, X_for_model, threshold=threshold)

        if pred == 1:
            st.success("Loan Approved ✅")
        else:
            st.error("Loan Not Approved ❌")

        if proba is not None:
            st.metric("Approval probability", f"{proba:.1%}")

        st.divider()
        st.subheader("📝 What you entered")
        st.json(user_inputs)

    else:
        st.info("Set inputs in the sidebar and click **Predict**.")

st.divider()
with st.expander("ℹ️ Notes / Limitations"):
    st.markdown(
        """
- This app reproduces the notebook workflow (including dummy encoding + scaling).
- Some models are trained on scaled features; tuned Random Forest follows the notebook’s unscaled training.
- Predictions depend on the dataset distribution and may not generalize to different populations.
        """
    )