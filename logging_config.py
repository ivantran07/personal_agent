"""Configure metadata-only JSON logging for the application."""

import logging
import os
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pythonjsonlogger.json import JsonFormatter

LOGGER_NAMESPACE = "personal_agent"

_ALLOWED_FIELDS = {
    "attempt",
    "delay_seconds",
    "error_type",
    "event",
    "level",
    "logger",
    "max_attempts",
    "max_iterations",
    "model",
    "next_model",
    "profile",
    "reason",
    "status_code",
    "timestamp",
    "tool_name",
    "traceback",
}


class MetadataJsonFormatter(JsonFormatter):
    """Render a stable, deliberately restricted set of log fields."""

    def add_fields(
        self,
        log_data: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any],
    ) -> None:
        super().add_fields(log_data, record, message_dict)
        log_data.update(
            {
                "timestamp": datetime.fromtimestamp(record.created, UTC)
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z"),
                "level": record.levelname,
                "logger": record.name,
                "event": record.getMessage(),
            }
        )

        for field in set(log_data) - _ALLOWED_FIELDS:
            del log_data[field]


def configure_logging() -> None:
    """Configure the application logger from ``LOG_LEVEL``."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(MetadataJsonFormatter())

    app_logger = logging.getLogger(LOGGER_NAMESPACE)
    for existing_handler in app_logger.handlers[:]:
        app_logger.removeHandler(existing_handler)
        existing_handler.close()
    app_logger.addHandler(handler)
    app_logger.setLevel(logging.INFO)
    app_logger.propagate = False

    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    try:
        level = logging.getLevelNamesMapping()[level_name]
    except KeyError:
        raise ValueError(f"Invalid LOG_LEVEL: {level_name}") from None
    app_logger.setLevel(level)


def exception_metadata(error: BaseException) -> dict[str, object]:
    """Return traceback locations without exception messages or local values."""
    frames = [
        {
            "file": Path(frame.filename).name,
            "function": frame.name,
            "line": frame.lineno,
        }
        for frame in traceback.extract_tb(error.__traceback__)
    ]
    return {"error_type": type(error).__name__, "traceback": frames}
