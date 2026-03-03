import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("loan_app")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = RotatingFileHandler(log_file, maxBytes=300_000, backupCount=3)
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger