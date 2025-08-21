import logging
import os
from typing import Optional

from src.config import Config, get_log_level


def setup_logging(cfg: Optional[Config] = None) -> None:
    level = get_log_level(cfg)
    # Respect existing handlers to avoid duplicate logs when called multiple times
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    fmt = os.environ.get(
        "LOG_FORMAT",
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    datefmt = os.environ.get("LOG_DATEFMT", "%Y-%m-%d %H:%M:%S")
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)
