"""Compatibility shim for legacy imports.

The project stores the real logger configuration in src/logger_config.py,
but some modules still import setup_logger from the repository root.
"""

from src.logger_config import get_logger


def setup_logger(name: str, level=None):
    """Backward-compatible wrapper around src.logger_config.get_logger."""
    if level is None:
        return get_logger(name)
    return get_logger(name, level)
