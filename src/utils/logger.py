import json
import logging
import platform
from pathlib import Path
from typing import Any, Dict, Optional

from .io import ensure_dir


def setup_logger(name: str, log_path: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_path:
        path = Path(log_path)
        ensure_dir(path.parent)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_config(logger: logging.Logger, cfg: Dict[str, Any]) -> None:
    logger.info("Config: %s", json.dumps(cfg, indent=2, sort_keys=True))


def log_versions(logger: logging.Logger) -> None:
    logger.info("Python: %s", platform.python_version())
    try:
        import torch  # type: ignore

        logger.info("torch: %s", torch.__version__)
    except Exception:
        pass
    try:
        import transformers  # type: ignore

        logger.info("transformers: %s", transformers.__version__)
    except Exception:
        pass
    try:
        import datasets  # type: ignore

        logger.info("datasets: %s", datasets.__version__)
    except Exception:
        pass
