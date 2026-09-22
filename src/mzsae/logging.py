"""Telemetry and logging utilities for MZSAE."""

import logging
import sys

_LOGGER_NAME = "mzsae"


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    """Returns a configured logger instance for MZSAE."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
