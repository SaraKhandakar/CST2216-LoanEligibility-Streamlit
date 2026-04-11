# =========================
# Logging Utility Module
# =========================
# This file configures a centralized logger for the loan eligibility application.
# It supports both file-based logging and console output.

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logger(log_file: Path) -> logging.Logger:
    """
    Configure and return a logger for the application.

    Parameters:
    log_file (Path): Path to the log file

    Returns:
    logging.Logger: Configured logger instance

    Purpose:
    - Store logs in a file with rotation to prevent large file sizes
    - Display logs in the console for real-time debugging
    - Ensure consistent logging across the project
    """

    # =========================
    # Ensure Log Directory Exists
    # =========================
    # Create logs folder if it does not exist
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Create or retrieve logger instance
    logger = logging.getLogger("loan_app")
    logger.setLevel(logging.INFO)

    # =========================
    # Prevent Duplicate Handlers
    # =========================
    # Avoid adding multiple handlers if logger is already configured
    if logger.handlers:
        return logger

    # Define log message format
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # =========================
    # File Handler (with Rotation)
    # =========================
    # Rotates log file when it reaches a certain size
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=300_000,   # Max size (~300 KB)
        backupCount=3       # Keep last 3 log files
    )
    file_handler.setFormatter(fmt)

    # =========================
    # Console Handler
    # =========================
    # Outputs logs to terminal for real-time monitoring
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger