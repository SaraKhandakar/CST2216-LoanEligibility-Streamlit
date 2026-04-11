# =========================
# Data Loading Module
# =========================
# This file handles loading the dataset used for the loan eligibility model.

import pandas as pd
from pathlib import Path


def load_data(csv_path: Path, logger) -> pd.DataFrame:
    """
    Load dataset from a CSV file.

    Parameters:
    csv_path (Path): Path to the dataset file
    logger: Logger object for tracking execution

    Returns:
    pd.DataFrame: Loaded dataset

    Purpose:
    - Load dataset into memory
    - Log dataset loading process
    - Help debug issues if file loading fails
    """
    try:
        # =========================
        # Load Dataset
        # =========================
        logger.info(f"Loading dataset: {csv_path}")
        df = pd.read_csv(csv_path)

        # Log dataset shape (rows, columns)
        logger.info(f"Loaded shape: {df.shape}")

        return df

    except Exception:
        # Log any error during loading
        logger.exception("Failed to load dataset.")
        raise