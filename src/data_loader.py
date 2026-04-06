import pandas as pd
from pathlib import Path

def load_data(csv_path: Path, logger) -> pd.DataFrame:
    try:
        logger.info(f"Loading dataset: {csv_path}")
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded shape: {df.shape}")
        return df
    except Exception:
        logger.exception("Failed to load dataset.")
        raise