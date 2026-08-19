"""Tests for metadata-only JSON logging configuration."""

import json
import logging
import sys
from io import StringIO

import pytest

from logging_config import LOGGER_NAMESPACE, configure_logging, exception_metadata


@pytest.fixture
def configured_logger(monkeypatch):
    """Configure an application child logger and clean up its handler afterward."""
    stream = StringIO()
    monkeypatch.setattr(sys, "stderr", stream)
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    configure_logging()
    yield logging.getLogger(f"{LOGGER_NAMESPACE}.test"), stream

    app_logger = logging.getLogger(LOGGER_NAMESPACE)
    for handler in app_logger.handlers[:]:
        app_logger.removeHandler(handler)
        handler.close()
    app_logger.setLevel(logging.NOTSET)
    app_logger.propagate = True


def test_log_line_is_structured_json(configured_logger):
    logger, stream = configured_logger
    logger.debug("tool.started")
    logger.info("tool.completed", extra={"tool_name": "add"})

    record = json.loads(stream.getvalue())

    assert record["timestamp"].endswith("Z")
    assert record["level"] == "INFO"
    assert record["logger"] == f"{LOGGER_NAMESPACE}.test"
    assert record["event"] == "tool.completed"
    assert record["tool_name"] == "add"


def test_exception_and_unapproved_fields_are_redacted(configured_logger):
    logger, stream = configured_logger
    secret = "never-log-this-value"

    try:
        raise ValueError(secret)
    except ValueError as error:
        logger.error(
            "tool.failed",
            extra={"unapproved": secret, **exception_metadata(error)},
        )

    output = stream.getvalue()
    record = json.loads(output)

    assert secret not in output
    assert "unapproved" not in record
    assert record["error_type"] == "ValueError"
    assert record["traceback"][-1]["file"] == "test_logging_config.py"
    assert record["traceback"][-1]["function"] == (
        "test_exception_and_unapproved_fields_are_redacted"
    )
    assert isinstance(record["traceback"][-1]["line"], int)


def test_usage_fields_are_allowed_but_content_is_redacted(configured_logger):
    logger, stream = configured_logger

    logger.info(
        "llm.usage",
        extra={
            "request_kind": "agent",
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "content": "do not log this prompt",
        },
    )

    record = json.loads(stream.getvalue())
    assert record["prompt_tokens"] == 100
    assert record["completion_tokens"] == 20
    assert record["total_tokens"] == 120
    assert "content" not in record
