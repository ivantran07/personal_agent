"""Tests for model-limit discovery, preflight counting, and compaction."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from litellm import ModelResponse

import compaction


def context_config(**overrides):
    return {
        "model": "test-model",
        "max_completion_tokens": 100,
        "context_trigger_ratio": 0.85,
        "context_target_ratio": 0.60,
        "context_safety_tokens": 10,
        "compaction_max_completion_tokens": 50,
        "context_window_overrides": {},
        **overrides,
    }


@pytest.fixture(autouse=True)
def clear_limit_caches():
    compaction._MODEL_LIMITS_CACHE.clear()
    compaction._OPENROUTER_CATALOG_CACHE.clear()


async def test_explicit_context_override_avoids_provider_discovery(monkeypatch):
    get_model_info = Mock(side_effect=AssertionError("unexpected discovery"))
    monkeypatch.setattr(compaction.litellm, "get_model_info", get_model_info)

    limits = await compaction.discover_model_limits(
        "test-model",
        context_config(context_window_overrides={"test-model": 32768}),
    )

    assert limits == compaction.ModelLimits(context_window=32768)
    get_model_info.assert_not_called()


async def test_openrouter_catalog_is_shared_between_models(monkeypatch):
    get_json = Mock(
        return_value={
            "data": [
                {"id": "vendor/primary", "context_length": 1000},
                {
                    "id": "vendor/fallback",
                    "context_length": 2000,
                    "top_provider": {"max_completion_tokens": 300},
                },
            ]
        }
    )
    monkeypatch.setattr(compaction, "_get_json", get_json)
    config = context_config(api_base="https://openrouter.test/v1")

    primary = await compaction.discover_model_limits(
        "openrouter/vendor/primary", config
    )
    fallback = await compaction.discover_model_limits(
        "openrouter/vendor/fallback", config
    )

    assert primary.context_window == 1000
    assert fallback == compaction.ModelLimits(
        context_window=2000, max_output_tokens=300
    )
    get_json.assert_called_once()


async def test_llamacpp_context_is_read_from_props(monkeypatch):
    get_json = Mock(return_value={"default_generation_settings": {"n_ctx": 16384}})
    monkeypatch.setattr(compaction, "_get_json", get_json)

    limits = await compaction.discover_model_limits(
        "openai/local-model",
        context_config(
            metadata_provider="llamacpp",
            api_base="http://127.0.0.1:8080/v1",
        ),
    )

    assert limits.context_window == 16384
    assert get_json.call_args.args[0] == "http://127.0.0.1:8080/props"


async def test_litellm_model_info_supplies_context_and_output_limits(monkeypatch):
    monkeypatch.setattr(
        compaction.litellm,
        "get_model_info",
        Mock(return_value={"max_input_tokens": 8192, "max_output_tokens": 1024}),
    )

    limits = await compaction.discover_model_limits(
        "gemini/test-model", context_config()
    )

    assert limits == compaction.ModelLimits(context_window=8192, max_output_tokens=1024)


def test_count_request_tokens_passes_messages_and_tools(monkeypatch):
    counter = Mock(return_value=123)
    monkeypatch.setattr(compaction.litellm, "token_counter", counter)
    messages = [{"role": "user", "content": "hello"}]
    tools = [{"type": "function", "function": {"name": "example"}}]

    assert compaction.count_request_tokens("test-model", messages, tools) == 123
    counter.assert_called_once_with(model="test-model", messages=messages, tools=tools)


async def test_compaction_preserves_system_active_turn_and_tool_group(monkeypatch):
    def fake_count(_model, messages, _tools=None):
        return sum(len(str(message.get("content") or "")) for message in messages)

    summarize = AsyncMock(return_value="## Current State\nOlder work summarized.")
    monkeypatch.setattr(compaction, "count_request_tokens", fake_count)
    monkeypatch.setattr(compaction, "_summarize_history", summarize)
    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "lookup", "arguments": "{}"},
    }
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "a" * 300},
        {"role": "assistant", "content": None, "tool_calls": [tool_call]},
        {"role": "tool", "tool_call_id": "call_1", "content": "b" * 300},
        {"role": "user", "content": "recent question"},
        {"role": "assistant", "content": "recent answer"},
        {"role": "user", "content": "current question"},
    ]

    await compaction.compact_messages(
        messages, "test-model", context_config(), compaction.ModelLimits(1200)
    )

    assert messages[0] == {"role": "system", "content": "system"}
    assert messages[1]["content"].startswith(compaction.COMPACTED_HISTORY_PREFIX)
    assert messages[-3:] == [
        {"role": "user", "content": "recent question"},
        {"role": "assistant", "content": "recent answer"},
        {"role": "user", "content": "current question"},
    ]
    selected = summarize.call_args.args[0]
    assert selected[1]["tool_calls"] == [tool_call]
    assert selected[2]["tool_call_id"] == "call_1"


async def test_summary_request_is_non_streaming_and_has_no_tools(
    monkeypatch, fake_message_factory
):
    response = Mock(spec=ModelResponse)
    response.model = "test-model"
    response.usage = None
    response.choices = [
        SimpleNamespace(message=fake_message_factory(content="summary"))
    ]
    completion = AsyncMock(return_value=response)
    monkeypatch.setattr(compaction, "acompletion", completion)
    monkeypatch.setattr(
        compaction, "count_request_tokens", lambda *_args, **_kwargs: 10
    )

    summary = await compaction._summarize_history(
        [{"role": "user", "content": "old"}],
        "test-model",
        context_config(api_base="https://example.test", api_key="secret"),
        compaction.ModelLimits(1000),
    )

    assert summary == "summary"
    assert completion.call_args.kwargs["stream"] is False
    assert "tools" not in completion.call_args.kwargs
    assert completion.call_args.kwargs["max_completion_tokens"] == 50


async def test_oversized_summary_is_bounded_and_still_reduces_history(monkeypatch):
    def fake_count(_model, messages, _tools=None):
        return sum(len(str(message.get("content") or "")) for message in messages)

    summarize = AsyncMock(return_value="summary " * 300)
    monkeypatch.setattr(compaction, "count_request_tokens", fake_count)
    monkeypatch.setattr(compaction, "_summarize_history", summarize)
    config = context_config(compaction_max_completion_tokens=1000)
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "a" * 150},
        {"role": "assistant", "content": "b" * 150},
        {"role": "user", "content": "current"},
    ]
    previous_tokens = fake_count("test-model", messages)

    await compaction.compact_messages(
        messages, "test-model", config, compaction.ModelLimits(1200)
    )

    assert fake_count("test-model", messages) < previous_tokens
    assert messages[1]["content"].startswith(compaction.COMPACTED_HISTORY_PREFIX)
    summary_budget = summarize.call_args.kwargs["max_completion_tokens"]
    assert 0 < summary_budget < config["compaction_max_completion_tokens"]


def test_provider_usage_is_logged_as_metadata(monkeypatch):
    logger = Mock()
    monkeypatch.setattr(compaction, "logger", logger)
    response = Mock(spec=ModelResponse)
    response.model = "provider-model"
    response.usage = SimpleNamespace(
        prompt_tokens=100, completion_tokens=20, total_tokens=120
    )

    compaction.log_usage(response, "requested-model", "agent")

    logger.info.assert_called_once_with(
        "llm.usage",
        extra={
            "request_kind": "agent",
            "model": "requested-model",
            "response_model": "provider-model",
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        },
    )
