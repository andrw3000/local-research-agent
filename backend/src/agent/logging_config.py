import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO", name: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration for the agent.

    Args:
        level: The logging level to use. Defaults to "INFO".
        name: The name of the logger. If None, returns the root logger.

    Returns:
        A configured logger instance.
    """
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create console handler and set level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Get logger
    logger = logging.getLogger(name)
    logger.setLevel(level.upper())

    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    logger.addHandler(console_handler)

    # Prevent propagation to avoid duplicate logs
    logger.propagate = False

    return logger
