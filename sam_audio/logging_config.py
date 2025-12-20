# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Centralized logging configuration for SAM-Audio.

Provides rotating file handler support and consistent log formatting
across all modules.
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# Default configuration
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_BACKUP_COUNT = 5


def setup_logging(
    log_dir: Optional[str | Path] = None,
    log_level: int | str = logging.INFO,
    enable_console: bool = True,
    enable_file: bool = True,
    log_filename: str = "sam_audio.log",
    suppress_hf_progress: bool = True,
) -> logging.Logger:
    """
    Configure logging for SAM-Audio with rotating file handlers.

    Args:
        log_dir: Directory for log files. If None, file logging is disabled.
                 Created if it doesn't exist.
        log_level: Logging level (e.g., logging.DEBUG, logging.INFO, "DEBUG", "INFO").
        enable_console: Whether to output logs to console.
        enable_file: Whether to output logs to file.
        log_filename: Name of the log file.
        suppress_hf_progress: Suppress HuggingFace Hub tqdm progress bars.

    Returns:
        The root logger for SAM-Audio.
    """
    # Convert string level to int if needed
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), logging.INFO)

    # Get root logger for sam_audio
    root_logger = logging.getLogger("sam_audio")
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(DEFAULT_LOG_FORMAT, DEFAULT_DATE_FORMAT)

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        # Force line buffering for real-time output
        if hasattr(console_handler.stream, "reconfigure"):
            console_handler.stream.reconfigure(line_buffering=True)
        root_logger.addHandler(console_handler)

    # File handler with rotation
    if enable_file and log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        log_file = log_path / log_filename

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=DEFAULT_MAX_BYTES,
            backupCount=DEFAULT_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Suppress HuggingFace Hub progress bars and excessive logging
    if suppress_hf_progress:
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        # Reduce HF Hub logging verbosity
        logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
        logging.getLogger("huggingface_hub.file_download").setLevel(logging.WARNING)

    # Suppress tqdm if not in verbose mode
    if log_level > logging.DEBUG:
        logging.getLogger("tqdm").setLevel(logging.WARNING)

    # Reduce transformers logging
    logging.getLogger("transformers").setLevel(logging.WARNING)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Args:
        name: Module name (typically __name__).

    Returns:
        A logger instance.
    """
    return logging.getLogger(name)


def flush_output():
    """Flush stdout and stderr to ensure all output is visible."""
    sys.stdout.flush()
    sys.stderr.flush()


class ProgressLogger:
    """
    Custom progress logger that writes to both console and log file.

    Replaces tqdm for HuggingFace downloads with proper flushing.
    """

    def __init__(
        self,
        total: int,
        desc: str = "",
        logger: Optional[logging.Logger] = None,
        log_interval: int = 20,
    ):
        """
        Initialize progress logger.

        Args:
            total: Total number of items.
            desc: Description prefix.
            logger: Logger to use (defaults to sam_audio logger).
            log_interval: Percentage interval for logging (e.g., 20 = log at 20%, 40%, etc.).
        """
        self.total = total
        self.desc = desc
        self.logger = logger or get_logger("sam_audio")
        self.log_interval = log_interval
        self.current = 0
        self._last_logged_pct = -1

    def update(self, n: int = 1):
        """Update progress by n items."""
        self.current += n
        pct = int(100.0 * self.current / self.total) if self.total > 0 else 100

        # Log at intervals
        if pct >= self._last_logged_pct + self.log_interval or pct == 100:
            self._last_logged_pct = pct
            self.logger.info("%s: %d%% (%d/%d)", self.desc, pct, self.current, self.total)
            flush_output()

    def close(self):
        """Complete the progress."""
        if self.current < self.total:
            self.current = self.total
            self.logger.info("%s: complete (%d/%d)", self.desc, self.total, self.total)
            flush_output()


class LogContext:
    """Context manager for timed operations with logging."""

    def __init__(self, message: str, logger: Optional[logging.Logger] = None):
        """
        Initialize timed log context.

        Args:
            message: Description of the operation.
            logger: Logger to use.
        """
        self.message = message
        self.logger = logger or get_logger("sam_audio")
        self._start_time: float = 0

    def __enter__(self):
        import time

        self._start_time = time.perf_counter()
        self.logger.info("%s...", self.message)
        flush_output()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time

        elapsed = time.perf_counter() - self._start_time
        if exc_type is None:
            self.logger.info("%s completed (%.2fs)", self.message, elapsed)
        else:
            self.logger.error("%s failed after %.2fs: %s", self.message, elapsed, exc_val)
        flush_output()
        return False


__all__ = [
    "setup_logging",
    "get_logger",
    "flush_output",
    "ProgressLogger",
    "LogContext",
]
