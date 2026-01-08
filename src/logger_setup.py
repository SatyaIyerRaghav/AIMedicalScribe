from __future__ import annotations
import logging
import os
from datetime import datetime
import sys
import warnings


# --------------------------------------------------------------------
# Helper: Ensure directory exists
# --------------------------------------------------------------------
def ensure_logs_directory(base_dir: str = "Logs") -> str:
    path = os.path.abspath(base_dir)
    os.makedirs(path, exist_ok=True)
    return path


# --------------------------------------------------------------------
# Initialize global logging (root logger)
# --------------------------------------------------------------------
def init_logging(base_dir: str = "Logs") -> None:
    """
    Initialize root logger with flow and error file handlers.
    Should be called once at program startup (e.g., in run.py).
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_aws")
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

    logs_dir = ensure_logs_directory(base_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    flow_path = os.path.join(logs_dir, f"flow_{stamp}.log")
    error_path = os.path.join(logs_dir, f"error_{stamp}.log")

    logger = logging.getLogger()
    if getattr(logger, "_custom_logging_initialized", False):
        return

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # File handlers
    flow_handler = logging.FileHandler(flow_path, encoding="utf-8")
    flow_handler.setLevel(logging.INFO)
    flow_handler.setFormatter(fmt)

    error_handler = logging.FileHandler(error_path, encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(fmt)

    # Console (only warnings and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(fmt)

    logger.addHandler(flow_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)

    logger._custom_logging_initialized = True  # type: ignore[attr-defined]


# --------------------------------------------------------------------
# Create or get a per-module logger (e.g., pdf_saver, rcm_coder, etc.)
# --------------------------------------------------------------------
def get_logger(name: str, filename: str | None = None, base_dir: str = "Logs/Modules") -> logging.Logger:
    """
    Returns a logger for a specific module with its own log file.
    Example:
        logger = get_logger("pdf_saver", "pdf_saver.log")
    """
    logs_dir = ensure_logs_directory(base_dir)
    logger = logging.getLogger(name)

    # Avoid re-adding handlers
    if getattr(logger, "_custom_module_logging_initialized", False):
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    if filename:
        # ✅ FIX: Ensure directory exists before file creation
        file_path = os.path.join(logs_dir, filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        fh = logging.FileHandler(file_path, encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger._custom_module_logging_initialized = True  # type: ignore[attr-defined]
    return logger
