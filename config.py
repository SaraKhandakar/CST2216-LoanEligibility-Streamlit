from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = PROJECT_ROOT / "data" / "credit.csv"
ARTIFACT_PATH = PROJECT_ROOT / "models" / "artifacts.joblib"
LOG_PATH = PROJECT_ROOT / "logs" / "app.log"

RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET_COL = "Loan_Approved"
DROP_COLS = ["Loan_ID"]

CATEGORICAL_DUMMY_COLS = [
    "Gender", "Married", "Dependents",
    "Education", "Self_Employed", "Property_Area"
]

# Notebook tuned RF params (the second RF)
RF_TUNED_PARAMS = dict(n_estimators=2, max_depth=2, max_features=10)