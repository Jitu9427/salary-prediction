"""
utils/logger.py
───────────────
Centralised logging factory for the Salary-Prediction MLOps project.

Features
--------
* Colour-coded console output (DEBUG → CRITICAL)
* Daily-rotating file handler  → logs/<name>/YYYY-MM-DD.log
* Safe for multi-module import  (same logger returned on repeated calls)
* Formats include timestamp, level, module, line-number
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional

# ── ANSI colour palette ────────────────────────────────────────────────────
_RESET = "\033[0m"
_BOLD  = "\033[1m"
_COLOURS = {
    logging.DEBUG:    "\033[36m",   # cyan
    logging.INFO:     "\033[32m",   # green
    logging.WARNING:  "\033[33m",   # yellow
    logging.ERROR:    "\033[31m",   # red
    logging.CRITICAL: "\033[35m",   # magenta
}

_LOG_FMT  = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

# ── Colour formatter ───────────────────────────────────────────────────────
class _ColourFormatter(logging.Formatter):
    """Adds ANSI colour codes to console log records."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        colour = _COLOURS.get(record.levelno, _RESET)
        record.levelname = f"{_BOLD}{colour}{record.levelname}{_RESET}"
        return super().format(record)


# ── Public factory ─────────────────────────────────────────────────────────
def get_logger(
    name: str,
    *,
    log_dir: str | Path = "logs",
    level: int = logging.DEBUG,
    console: bool = True,
    file_log: bool = True,
) -> logging.Logger:
    """
    Return (or create) a named logger.

    Parameters
    ----------
    name     : Logger name – typically ``__name__`` of the calling module.
    log_dir  : Root directory for log files.  Sub-folder ``name`` is created
               automatically.
    level    : Minimum log level (default: DEBUG).
    console  : Attach a colour-coded StreamHandler to stdout.
    file_log : Attach a TimedRotatingFileHandler (rotates daily, keeps 14 days).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on re-import
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    plain_fmt = logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT)

    # ── Console handler ────────────────────────────────────────────────────
    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(_ColourFormatter(_LOG_FMT, datefmt=_DATE_FMT))
        logger.addHandler(ch)

    # ── File handler ───────────────────────────────────────────────────────
    if file_log:
        log_path = Path(log_dir) / name
        log_path.mkdir(parents=True, exist_ok=True)
        fh = TimedRotatingFileHandler(
            filename=log_path / "run.log",
            when="midnight",
            interval=1,
            backupCount=14,
            encoding="utf-8",
            utc=False,
        )
        fh.suffix = "%Y-%m-%d"
        fh.setLevel(level)
        fh.setFormatter(plain_fmt)
        logger.addHandler(fh)

    return logger
