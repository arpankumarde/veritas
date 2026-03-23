"""Logging configuration for the Veritas engine."""

import logging
import os
from pathlib import Path

_logging_initialized = False


def setup_logging(log_dir: str | None = None) -> None:
    """Initialize centralized file logging for the Veritas engine.

    Sets up a root-level file handler so all loggers (engine.*, api.*) write
    to a shared log file.  Safe to call multiple times -- only the first
    invocation has any effect.
    """
    global _logging_initialized
    if _logging_initialized:
        return
    _logging_initialized = True

    level_name = os.environ.get("VERITAS_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    # File handler
    log_path = Path(log_dir) if log_dir else Path("logs")
    log_path.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path / "veritas.log")
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name, configured for Veritas."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        level = os.environ.get("VERITAS_LOG_LEVEL", "INFO").upper()
        handler.setLevel(getattr(logging, level, logging.INFO))
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, level, logging.INFO))
    return logger
