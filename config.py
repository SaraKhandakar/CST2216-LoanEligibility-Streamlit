# =========================
# Configuration Module
# =========================
# This file stores all project-level constants such as file paths,
# model parameters, and preprocessing settings.

from pathlib import Path

# =========================
# Project Paths
# =========================
# Base directory of the project
PROJECT_ROOT = Path(__file__).resolve().parent

# Paths to data, saved models, and logs
DATA_PATH = PROJECT_ROOT / "data" / "credit.csv"
ARTIFACT_PATH = PROJECT_ROOT / "models" / "artifacts.joblib"
LOG_PATH = PROJECT_ROOT / "logs" / "app.log"

# =========================
# Training Settings
# =========================
# Random seed ensures reproducibility of results
RANDOM_STATE = 42

# Proportion of dataset used for testing
TEST_SIZE = 0.2

# =========================
# Dataset Configuration
# =========================
# Target variable (what the model predicts)
TARGET_COL = "Loan_Approved"

# Columns to drop (not useful for prediction)
DROP_COLS = ["Loan_ID"]

# Categorical columns that will be converted into dummy variables
CATEGORICAL_DUMMY_COLS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area"
]

# =========================
# Model Parameters
# =========================
# Tuned Random Forest parameters (based on notebook experimentation)
RF_TUNED_PARAMS = dict(
    n_estimators=2,
    max_depth=2,
    max_features=10
)