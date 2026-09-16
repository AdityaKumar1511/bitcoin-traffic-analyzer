"""
Logging configuration for Bitcoin Traffic Analyzer.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get or create a configured logger instance.

    Args:
        name: Name of the logger (defaults to root/caller module name).

    Returns:
        Configured logging.Logger.
    """
    logger = logging.getLogger(name or "bitcoin_traffic_analyzer")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger
