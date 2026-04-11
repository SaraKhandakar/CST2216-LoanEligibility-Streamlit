# =========================
# Loan Eligibility Streamlit App
# =========================
# This file builds the Streamlit interface for the loan eligibility project.
# It loads data, trains or loads saved models, evaluates model performance,
# and allows the user to enter applicant details for prediction.

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

# =========================
# Logger Setup
# =========================
# Logger is used to track application activity and errors
logger = setup_logger(LOG_PATH)

# =========================
# Streamlit Page Configuration
# =========================
st.set_page_config(page_title="Loan Eligibility - Multi Model", layout="wide")

# =========================
# App Header
# =========================
# Display app title and short description
st.title("🏦 Loan Eligibility Predictor (Multi-Model)")
st.caption(
    "Multiple machine learning models trained and evaluated on loan data."
)

# =========================
# Load or Train Artifacts
# =========================
# This function either loads previously saved models and preprocessing
# artifacts or trains them from scratch if they do not exist.
@st.cache_resource
def get_artifacts():
    """
    Load saved training artifacts if available.
    Otherwise, preprocess data, train models, and save artifacts.

    Returns:
    dict: Dictionary containing trained models, scaler,
          train/test splits, and processed feature columns
    """
    if ARTIFACT_PATH.exists():
        return load_artifacts(ARTIFACT_PATH, logger)

    # Load raw dataset
    raw = load_data(DATA_PATH, logger)

    # Apply notebook-style preprocessing
    cleaned = preprocess_like_notebook(raw, logger)

    # Convert categorical variables and target column
    processed = make_dummies_and_target(cleaned, CATEGORICAL_DUMMY_COLS, TARGET_COL, logger)

    # Train all models and collect artifacts
    artifacts = split_scale_train_all(
        processed, TARGET_COL, TEST_SIZE, RANDOM_STATE, RF_TUNED_PARAMS, logger
    )

    # Save processed feature column names for later input alignment
    artifacts["processed_columns"] = processed.drop(TARGET_COL, axis=1).columns.tolist()

    # Save artifacts to disk for reuse
    save_artifacts(artifacts, ARTIFACT_PATH, logger)
    return artifacts


# Retrieve saved or newly trained artifacts
art = get_artifacts()

# Extract artifacts for app use
models = art["models"]
xtest = art["xtest"]
ytest = art["ytest"]
xtest_scaled = art["xtest_scaled"]
scaler = art["scaler"]
processed_cols = art["processed_columns"]

# =========================
# Sidebar Settings
# =========================
# Sidebar allows model selection and user input
st.sidebar.header("⚙️ Settings")

# Let user choose which trained model to use
model_name = st.sidebar.selectbox("Choose a model", list(models.keys()))

# Optional threshold only for models with probability output
threshold = None
if "Logistic Regression" in model_name:
    st.sidebar.caption("Optional: apply a probability cutoff (like notebook)")
    threshold = st.sidebar.slider("Approval threshold", 0.10, 0.90, 0.50, 0.05)

st.sidebar.divider()
st.sidebar.subheader("🧾 Applicant Inputs")

# =========================
# Build User Input Controls
# =========================
# Inputs are created from cleaned original data so they remain human-readable
raw = load_data(DATA_PATH, logger)
cleaned = preprocess_like_notebook(raw, logger)
feature_cols = [c for c in cleaned.columns if c not in [TARGET_COL, "Loan_ID"]]
user_inputs = {}

for col in feature_cols:
    series = cleaned[col]

    # Use dropdowns for categorical/text-like columns
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series) or pd.api.types.is_categorical_dtype(series):
        options = sorted(series.dropna().astype(str).unique().tolist())
        user_inputs[col] = st.sidebar.selectbox(col, options)

    # Use numeric input boxes for numeric columns
    elif pd.api.types.is_numeric_dtype(series):
        default_value = float(series.dropna().median()) if not series.dropna().empty else 0.0
        user_inputs[col] = st.sidebar.number_input(col, value=default_value)

    # Fallback in case column type is unusual
    else:
        options = sorted(series.dropna().astype(str).unique().tolist())
        user_inputs[col] = st.sidebar.selectbox(col, options)

# Prediction button in sidebar
predict_clicked = st.sidebar.button("🔮 Predict")

# =========================
# Main Layout
# =========================
# Split page into result section and prediction section
left, right = st.columns([1.2, 1])

with left:
    st.subheader("📊 Test Set Results (All Models)")

    # Evaluate all models on the test set for comparison
    rows = []
    cms = {}
    for name, model in models.items():
        # Use unscaled or scaled data depending on how model was trained
        if "unscaled" in name:
            acc, cm, _ = eval_model(model, xtest, ytest)
        else:
            acc, cm, _ = eval_model(model, xtest_scaled, ytest)

        rows.append({"Model": name, "Accuracy": acc})
        cms[name] = cm

    # Show model comparison table sorted by accuracy
    results_df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False)
    st.dataframe(results_df, use_container_width=True)

    st.caption("Confusion matrix shown for the selected model (below).")

    # Display confusion matrix for chosen model
    st.subheader("🧩 Confusion Matrix (Selected Model)")
    cm = cms[model_name]
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
    st.table(cm_df)

with right:
    st.subheader("✅ Prediction Output")

    # Show selected settings
    st.write(f"**Selected model:** {model_name}")
    if threshold is not None:
        st.write(f"**Threshold:** {threshold:.2f}")

    if predict_clicked:
        # =========================
        # Prepare Single User Input
        # =========================
        # Convert user inputs into a single-row DataFrame
        single = build_input_row(cleaned, user_inputs)

        # Add a temporary target column so the same preprocessing
        # function can be reused, then remove it immediately
        single = make_dummies_and_target(
            single.assign(**{TARGET_COL: "N"}),
            CATEGORICAL_DUMMY_COLS,
            TARGET_COL,
            logger
        )
        single_X = single.drop(TARGET_COL, axis=1)

        # Align input columns to exactly match training features
        single_X = single_X.reindex(columns=processed_cols, fill_value=0)

        model = models[model_name]

        # Apply scaling only for models trained on scaled data
        if "unscaled" in model_name:
            X_for_model = single_X
        else:
            X_for_model = scaler.transform(single_X)

        # Generate prediction and probability
        pred, proba = predict_with_model(model, X_for_model, threshold=threshold)

        # Display prediction outcome
        if pred == 1:
            st.success("Loan Approved ✅")
        else:
            st.error("Loan Not Approved ❌")

        # Display probability if available
        if proba is not None:
            st.metric("Approval probability", f"{proba:.1%}")

        st.divider()
        st.subheader("📝 What you entered")
        st.json(user_inputs)

    else:
        st.info("Set inputs in the sidebar and click **Predict**.")

# =========================
# Notes and Limitations
# =========================
# Explain important assumptions of the application
st.divider()
with st.expander("ℹ️ Notes / Limitations"):
    st.markdown(
        """
- This app reproduces the notebook workflow (including dummy encoding + scaling).
- Some models are trained on scaled features; tuned Random Forest follows the notebook’s unscaled training.
- Predictions depend on the dataset distribution and may not generalize to different populations.
        """
    )