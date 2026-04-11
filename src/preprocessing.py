# =========================
# Data Preprocessing Module
# =========================
# This file contains functions used to clean, transform,
# and prepare the loan eligibility dataset for model training.

import pandas as pd


def preprocess_like_notebook(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Clean the dataset using the same preprocessing steps
    applied in the original notebook.

    Parameters:
    df (pd.DataFrame): Raw dataset
    logger: Logger object for tracking preprocessing steps

    Returns:
    pd.DataFrame: Cleaned dataset ready for encoding

    Purpose:
    - Handle missing values
    - Convert selected columns to categorical type
    - Remove unnecessary columns
    - Ensure dataset is ready for feature engineering
    """
    # Create a copy to avoid modifying original DataFrame
    df = df.copy()

    # Log initial dataset state
    logger.info(f"Initial shape: {df.shape}")
    logger.info(f"Initial missing values:\n{df.isna().sum()}")

    # =========================
    # Convert Columns to Categorical Type
    # =========================
    # These columns are treated as categorical variables
    # to match the notebook preprocessing approach
    df["Credit_History"] = df["Credit_History"].astype("object")
    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].astype("object")

    # =========================
    # Fill Missing Categorical Values
    # =========================
    # Use mode for most categorical columns
    # Gender is filled with "Male" to match notebook logic
    df["Gender"] = df["Gender"].fillna("Male")
    df["Married"] = df["Married"].fillna(df["Married"].mode()[0])
    df["Dependents"] = df["Dependents"].fillna(df["Dependents"].mode()[0])
    df["Self_Employed"] = df["Self_Employed"].fillna(df["Self_Employed"].mode()[0])
    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0])
    df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mode()[0])

    # =========================
    # Fill Missing Numeric Values
    # =========================
    # Median is used because it is less affected by extreme values
    df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
    df["ApplicantIncome"] = df["ApplicantIncome"].fillna(df["ApplicantIncome"].median())
    df["CoapplicantIncome"] = df["CoapplicantIncome"].fillna(df["CoapplicantIncome"].median())

    # Log missing values after filling
    logger.info(f"Missing values after fillna:\n{df.isna().sum()}")

    # =========================
    # Drop Remaining Missing Rows
    # =========================
    # Remove any leftover incomplete rows to avoid training issues
    df = df.dropna()

    logger.info(f"Shape after dropna: {df.shape}")
    logger.info(f"Missing values after dropna:\n{df.isna().sum()}")

    # =========================
    # Drop Unnecessary Identifier Column
    # =========================
    # Loan_ID is not useful for prediction
    if "Loan_ID" in df.columns:
        df = df.drop("Loan_ID", axis=1)

    return df


def make_dummies_and_target(df: pd.DataFrame, dummy_cols: list[str], target_col: str, logger) -> pd.DataFrame:
    """
    Apply one-hot encoding and convert the target variable into numeric form.

    Parameters:
    df (pd.DataFrame): Preprocessed dataset
    dummy_cols (list[str]): Columns to encode as dummy variables
    target_col (str): Name of the target column
    logger: Logger object for tracking steps

    Returns:
    pd.DataFrame: Final dataset ready for model training

    Purpose:
    - Convert categorical features into numeric dummy variables
    - Convert target labels from Y/N to 1/0
    - Ensure target column is valid numeric data
    """
    # Create copy to avoid modifying original DataFrame
    df = df.copy()

    # =========================
    # One-Hot Encoding
    # =========================
    # Convert selected categorical columns into dummy variables
    df = pd.get_dummies(df, columns=dummy_cols, dtype=int)

    # =========================
    # Clean and Convert Target Column
    # =========================
    # Remove extra spaces and standardize target values
    df[target_col] = df[target_col].astype(str).str.strip()

    # Convert target labels from Y/N to 1/0
    df[target_col] = df[target_col].replace({"Y": 1, "N": 0})

    # Force numeric conversion
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    # Drop rows with invalid target values
    df = df.dropna(subset=[target_col])

    # Convert target to integer class labels
    df[target_col] = df[target_col].astype(int)

    # Log final dataset information
    logger.info(f"Target unique values after conversion: {df[target_col].unique()}")
    logger.info(f"Target dtype after conversion: {df[target_col].dtype}")
    logger.info(f"Final columns after dummies:\n{df.columns.tolist()}")
    logger.info(f"Final missing values before training:\n{df.isna().sum()}")

    return df