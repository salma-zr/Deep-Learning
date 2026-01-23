"""Logging configuration for Medical QA project."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import platform
from importlib import metadata

from rich.console import Console
from rich.logging import RichHandler

# Global console for rich output
console = Console()

# Global logger registry
_loggers: dict[str, logging.Logger] = {}


def setup_logger(
    name: str = "medqa",
    level: str = "INFO",
    log_file: Optional[str | Path] = None,
    verbose: bool = False,
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
        verbose: If True, set level to DEBUG
        
    Returns:
        Configured logger instance
    """
    if name in _loggers:
        return _loggers[name]
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if verbose else getattr(logging, level.upper()))
    logger.handlers = []  # Clear any existing handlers
    
    # Console handler with Rich
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
    )
    console_handler.setLevel(logging.DEBUG if verbose else getattr(logging, level.upper()))
    console_format = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    _loggers[name] = logger
    return logger


def get_logger(name: str = "medqa") -> logging.Logger:
    """Get an existing logger or create a new one with default settings."""
    if name not in _loggers:
        return setup_logger(name)
    return _loggers[name]


def log_experiment_start(
    logger: logging.Logger,
    experiment_name: str,
    config: dict,
) -> None:
    """Log the start of an experiment with configuration details."""
    logger.info("=" * 60)
    logger.info(f"EXPERIMENT: {experiment_name}")
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info("-" * 60)
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("-" * 60)
    logger.info("Environment:")
    logger.info(f"  python: {sys.version.split()[0]}")
    logger.info(f"  platform: {platform.platform()}")
    for pkg in [
        "datasets",
        "transformers",
        "openai",
        "ollama",
        "sacrebleu",
        "rouge-score",
        "sentence-transformers",
    ]:
        try:
            version = metadata.version(pkg)
            logger.info(f"  {pkg}: {version}")
        except metadata.PackageNotFoundError:
            logger.info(f"  {pkg}: not installed")
    logger.info("=" * 60)


def log_experiment_end(
    logger: logging.Logger,
    experiment_name: str,
    metrics: Optional[dict] = None,
    duration_seconds: Optional[float] = None,
) -> None:
    """Log the end of an experiment with optional metrics."""
    logger.info("=" * 60)
    logger.info(f"COMPLETED: {experiment_name}")
    logger.info(f"Finished at: {datetime.now().isoformat()}")
    
    if duration_seconds is not None:
        minutes = int(duration_seconds // 60)
        seconds = duration_seconds % 60
        logger.info(f"Duration: {minutes}m {seconds:.1f}s")
    
    if metrics:
        logger.info("-" * 60)
        logger.info("Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
    
    logger.info("=" * 60)
