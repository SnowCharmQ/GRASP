"""Project logging independent of the training framework's root handlers."""

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a stderr logger with one handler and no duplicate propagation."""
    logger = logging.getLogger(f"grasp.{name}")
    logger.disabled = False
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    return logger
