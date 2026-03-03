import pandas as pd

def preprocess_like_notebook(df: pd.DataFrame, logger) -> pd.DataFrame:
    df = df.copy()

    # Convert to object like notebook
    df["Credit_History"] = df["Credit_History"].astype("object")
    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].astype("object")

    # Impute missing values exactly like notebook
    df["Gender"].fillna("Male", inplace=True)
    df["Married"].fillna(df["Married"].mode()[0], inplace=True)
    df["Dependents"].fillna(df["Dependents"].mode()[0], inplace=True)
    df["Self_Employed"].fillna(df["Self_Employed"].mode()[0], inplace=True)
    df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0], inplace=True)
    df["Credit_History"].fillna(df["Credit_History"].mode()[0], inplace=True)

    df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace=True)

    # Drop Loan_ID
    if "Loan_ID" in df.columns:
        df = df.drop("Loan_ID", axis=1)

    return df

def make_dummies_and_target(df: pd.DataFrame, dummy_cols: list[str], target_col: str, logger) -> pd.DataFrame:
    df = df.copy()

    # Dummies exactly like notebook
    df = pd.get_dummies(df, columns=dummy_cols, dtype=int)

    # Target Y/N -> 1/0 exactly like notebook
    df[target_col] = df[target_col].replace({"Y": 1, "N": 0})

    return df