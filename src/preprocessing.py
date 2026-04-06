import pandas as pd

def preprocess_like_notebook(df: pd.DataFrame, logger) -> pd.DataFrame:
    df = df.copy()

    logger.info(f"Initial shape: {df.shape}")
    logger.info(f"Initial missing values:\n{df.isna().sum()}")

    # Convert to object like notebook
    df["Credit_History"] = df["Credit_History"].astype("object")
    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].astype("object")

    # Fill missing categorical values
    df["Gender"] = df["Gender"].fillna("Male")
    df["Married"] = df["Married"].fillna(df["Married"].mode()[0])
    df["Dependents"] = df["Dependents"].fillna(df["Dependents"].mode()[0])
    df["Self_Employed"] = df["Self_Employed"].fillna(df["Self_Employed"].mode()[0])
    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0])
    df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mode()[0])

    # Fill missing numeric values
    df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
    df["ApplicantIncome"] = df["ApplicantIncome"].fillna(df["ApplicantIncome"].median())
    df["CoapplicantIncome"] = df["CoapplicantIncome"].fillna(df["CoapplicantIncome"].median())

    logger.info(f"Missing values after fillna:\n{df.isna().sum()}")

    # Drop any leftover missing rows
    df = df.dropna()

    logger.info(f"Shape after dropna: {df.shape}")
    logger.info(f"Missing values after dropna:\n{df.isna().sum()}")

    # Drop Loan_ID
    if "Loan_ID" in df.columns:
        df = df.drop("Loan_ID", axis=1)

    return df


def make_dummies_and_target(df: pd.DataFrame, dummy_cols: list[str], target_col: str, logger) -> pd.DataFrame:
    df = df.copy()

    # Dummies exactly like notebook
    df = pd.get_dummies(df, columns=dummy_cols, dtype=int)

    # Clean target column first
    df[target_col] = df[target_col].astype(str).str.strip()

    # Convert Y/N -> 1/0
    df[target_col] = df[target_col].replace({"Y": 1, "N": 0})

    # Force numeric
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    # Drop any rows where target is invalid
    df = df.dropna(subset=[target_col])

    # Force integer class labels
    df[target_col] = df[target_col].astype(int)

    logger.info(f"Target unique values after conversion: {df[target_col].unique()}")
    logger.info(f"Target dtype after conversion: {df[target_col].dtype}")
    logger.info(f"Final columns after dummies:\n{df.columns.tolist()}")
    logger.info(f"Final missing values before training:\n{df.isna().sum()}")

    return df