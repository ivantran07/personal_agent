"""Tests for streaming and retry behavior in ``main.run_agent_loop``."""

from unittest.mock import call

import litellm
import pytest

import main


def config(**overrides):
    return {
        "model": "test-model",
        "max_completion_tokens": 100,
        "max_iterations": 5,
        **overrides,
    }


def make_transient_error(error_type: type[Exception]) -> Exception:
    """Construct a transient LiteLLM exception with required request context."""
    return error_type(
        message="temporary API failure",
        llm_provider="test-provider",
        model="test-model",
    )


def make_api_error(status_code: int) -> litellm.APIError:
    """Construct a generic LiteLLM API error for status classification tests."""
    return litellm.APIError(
        status_code=status_code,
        message=f"API failure with status {status_code}",
        llm_provider="test-provider",
        model="test-model",
    )


async def test_prompt_is_appended_and_loop_stops_without_tool_calls(
    mock_input,
    mock_acompletion,
    fake_message_factory,
    fake_response_factory,
    fake_stream_factory,
):
    mock_input.side_effect = None
    mock_input.return_value = "hello"
    message = fake_message_factory(content="done")
    mock_acompletion.return_value = fake_stream_factory(
        fake_response_factory(message=message)
    )

    messages = []
    await main.run_agent_loop(messages, config())

    mock_input.assert_called_once()
    assert mock_acompletion.await_count == 1
    assert messages == [{"role": "user", "content": "hello"}, message.model_dump()]


async def test_model_name_is_printed_once_across_tool_iterations(
    mock_input,
    mock_acompletion,
    fake_message_factory,
    fake_response_factory,
    fake_stream_factory,
    patched_tools,
    tool_call_factory,
    capsys,
):
    mock_input.side_effect = None
    mock_input.return_value = "hello"
    patched_tools["noop"] = {"function": lambda: None, "schema": {}}
    tool_call = tool_call_factory(name="noop")
    mock_acompletion.side_effect = [
        fake_stream_factory(
            fake_response_factory(message=fake_message_factory(tool_calls=[tool_call]))
        ),
        fake_stream_factory(
            fake_response_factory(message=fake_message_factory(content="done"))
        ),
    ]

    await main.run_agent_loop([], config())

    assert (
        capsys.readouterr().out.count(f"{main.YELLOW}MODEL: test-model{main.RESET}")
        == 1
    )


async def test_tool_iteration_appends_rebuilt_assistant_and_tool_messages(
    mock_input,
    mock_acompletion,
    fake_message_factory,
    fake_response_factory,
    fake_stream_factory,
    patched_tools,
    tool_call_factory,
):
    mock_input.side_effect = None
    mock_input.return_value = "hello"
    patched_tools["add"] = {"function": lambda x, y: x + y, "schema": {}}
    tool_call = tool_call_factory(name="add", arguments='{"x": 1, "y": 2}')
    first_message = fake_message_factory(tool_calls=[tool_call])
    final_message = fake_message_factory(content="done")
    mock_acompletion.side_effect = [
        fake_stream_factory(fake_response_factory(message=first_message)),
        fake_stream_factory(fake_response_factory(message=final_message)),
    ]

    messages = []
    await main.run_agent_loop(messages, config())

    assert messages == [
        {"role": "user", "content": "hello"},
        first_message.model_dump(),
        {"tool_call_id": "call_1", "role": "tool", "content": "3"},
        final_message.model_dump(),
    ]


async def test_length_finish_reason_retries_without_appending_assistant_message(
    mock_input,
    mock_acompletion,
    fake_message_factory,
    fake_response_factory,
    fake_stream_factory,
):
    mock_input.side_effect = None
    mock_input.return_value = "hello"
    final_message = fake_message_factory(content="done")
    mock_acompletion.side_effect = [
        fake_stream_factory(fake_response_factory(finish_reason="length")),
        fake_stream_factory(fake_response_factory(message=final_message)),
    ]

    messages = []
    await main.run_agent_loop(messages, config())

    assert mock_acompletion.await_count == 2
    assert messages[1]["role"] == "user"
    assert "cut off" in messages[1]["content"]
    assert messages[-1] == final_message.model_dump()


async def test_empty_rebuilt_choices_logs_error_and_returns(
    mock_input,
    mock_acompletion,
    fake_response_factory,
    fake_stream_factory,
    mock_logger,
):
    mock_input.side_effect = None
    mock_input.return_value = "hello"
    mock_acompletion.return_value = fake_stream_factory(
        fake_response_factory(empty_choices=True)
    )

    await main.run_agent_loop([], config())

    mock_logger.error.assert_called_once_with("llm.response.empty")


async def test_streamed_content_is_printed_chunk_by_chunk(
    mock_input,
    mock_acompletion,
    fake_chunk_factory,
    fake_message_factory,
    fake_response_factory,
    fake_stream_factory,
    monkeypatch,
):
    """Only text deltas are printed, preserving each chunk and flushing stdout."""
    mock_input.side_effect = None
    mock_input.return_value = "hello"
    chunks = [
        fake_chunk_factory(has_choices=False),
        fake_chunk_factory(content=None),
        fake_chunk_factory(content="hel"),
        fake_chunk_factory(content="lo"),
    ]
    mock_acompletion.return_value = fake_stream_factory(
        fake_response_factory(message=fake_message_factory(content="hello")),
        chunks=chunks,
    )
    printed = []
    monkeypatch.setattr(
        "builtins.print", lambda *args, **kwargs: printed.append((args, kwargs))
    )

    await main.run_agent_loop([], config())

    assert (("hel",), {"end": "", "flush": True}) in printed
    assert (("lo",), {"end": "", "flush": True}) in printed


async def test_max_iterations_logs_warning_for_repeated_tool_calls(
    mock_input,
    mock_acompletion,
    fake_message_factory,
    fake_response_factory,
    fake_stream_factory,
    patched_tools,
    tool_call_factory,
    mock_logger,
):
    mock_input.side_effect = None
    mock_input.return_value = "hello"
    patched_tools["noop"] = {"function": lambda: None, "schema": {}}
    message = fake_message_factory(tool_calls=[tool_call_factory(name="noop")])
    mock_acompletion.return_value = fake_stream_factory(
        fake_response_factory(message=message)
    )

    await main.run_agent_loop([], config(max_iterations=3))

    assert mock_acompletion.await_count == 3
    mock_logger.warning.assert_called_once_with(
        "agent.max_iterations", extra={"max_iterations": 3}
    )


async def test_acompletion_uses_streaming_and_expected_configuration(
    mock_input,
    mock_acompletion,
    fake_message_factory,
    fake_response_factory,
    fake_stream_factory,
) -> None:
    """Disable LiteLLM retries while passing the expected request settings."""
    mock_input.side_effect = None
    mock_input.return_value = "hello"
    message = fake_message_factory(content="done")
    mock_acompletion.return_value = fake_stream_factory(
        fake_response_factory(message=message)
    )
    expected_config = config(
        fallbacks=["backup-model"], api_base="https://example.test", api_key="secret"
    )

    await main.run_agent_loop([], expected_config)

    assert mock_acompletion.call_args.kwargs == {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}, message.model_dump()],
        "max_completion_tokens": 100,
        "tools": main.TOOL_SCHEMAS,
        "api_base": "https://example.test",
        "api_key": "secret",
        "stream": True,
        "max_retries": 0,
        "num_retries": 0,
    }


@pytest.mark.parametrize(
    "error_type",
    [
        pytest.param(litellm.RateLimitError, id="rate-limit"),
        pytest.param(litellm.Timeout, id="timeout"),
        pytest.param(litellm.APIConnectionError, id="connection"),
        pytest.param(litellm.InternalServerError, id="internal-server"),
        pytest.param(litellm.BadGatewayError, id="bad-gateway"),
        pytest.param(litellm.ServiceUnavailableError, id="service-unavailable"),
    ],
)
async def test_transient_litellm_errors_retry_after_backoff(
    error_type,
    mock_input,
    mock_acompletion,
    mock_retry_sleep,
    fake_message_factory,
    fake_response_factory,
    fake_stream_factory,
) -> None:
    """Retry each explicitly supported transient LiteLLM exception."""
    mock_input.side_effect = None
    mock_input.return_value = "hello"
    message = fake_message_factory(content="recovered")
    mock_acompletion.side_effect = [
        make_transient_error(error_type),
        fake_stream_factory(fake_response_factory(message=message)),
    ]

    messages = []
    await main.run_agent_loop(messages, config())

    assert mock_acompletion.await_count == 2
    mock_retry_sleep.assert_awaited_once_with(1)
    assert messages[-1] == message.model_dump()


@pytest.mark.parametrize("status_code", [408, 409, 429, 500, 599])
async def test_retryable_api_statuses_retry_after_backoff(
    status_code,
    mock_input,
    mock_acompletion,
    mock_retry_sleep,
    fake_message_factory,
    fake_response_factory,
    fake_stream_factory,
) -> None:
    """Retry transient API status codes and server errors."""
    mock_input.side_effect = None
    mock_input.return_value = "hello"
    message = fake_message_factory(content="recovered")
    mock_acompletion.side_effect = [
        make_api_error(status_code),
        fake_stream_factory(fake_response_factory(message=message)),
    ]

    await main.run_agent_loop([], config())

    assert mock_acompletion.await_count == 2
    mock_retry_sleep.assert_awaited_once_with(1)


@pytest.mark.parametrize("status_code", [400, 401, 499])
async def test_non_retryable_api_status_fails_immediately(
    status_code,
    mock_input,
    mock_acompletion,
    mock_retry_sleep,
) -> None:
    """Propagate permanent API errors without sleeping or trying fallbacks."""
    mock_input.side_effect = None
    mock_input.return_value = "hello"
    error = make_api_error(status_code)
    mock_acompletion.side_effect = error

    with pytest.raises(litellm.APIError) as exc_info:
        await main.run_agent_loop([], config(fallbacks=["backup-model"]))

    assert exc_info.value is error
    assert mock_acompletion.await_count == 1
    mock_retry_sleep.assert_not_awaited()
    assert [item.kwargs["model"] for item in mock_acompletion.await_args_list] == [
        "test-model"
    ]


async def test_stream_failure_retries_with_fresh_completion(
    mock_input,
    mock_acompletion,
    mock_retry_sleep,
    mock_stream_chunk_builder,
    fake_chunk_factory,
    fake_message_factory,
    fake_response_factory,
    fake_stream_factory,
) -> None:
    """Discard partial chunks and retry API failures raised during streaming."""
    mock_input.side_effect = None
    mock_input.return_value = "hello"
    partial_chunk = fake_chunk_factory(content="partial")
    successful_chunk = fake_chunk_factory(content="complete")
    final_message = fake_message_factory(content="complete")
    mock_acompletion.side_effect = [
        fake_stream_factory(
            fake_response_factory(message=fake_message_factory(content="partial")),
            chunks=[partial_chunk],
            iteration_error=make_transient_error(litellm.APIConnectionError),
        ),
        fake_stream_factory(
            fake_response_factory(message=final_message), chunks=[successful_chunk]
        ),
    ]

    messages = []
    await main.run_agent_loop(messages, config())

    assert mock_acompletion.await_count == 2
    mock_retry_sleep.assert_awaited_once_with(1)
    mock_stream_chunk_builder.assert_called_once()
    assert mock_stream_chunk_builder.call_args.args[0] == [successful_chunk]
    assert messages[-1] == final_message.model_dump()


async def test_fallback_models_are_tried_before_sleeping(
    mock_input,
    mock_acompletion,
    mock_retry_sleep,
    mock_logger,
    fake_message_factory,
    fake_response_factory,
    fake_stream_factory,
) -> None:
    """Try every configured model before backing off and restarting the sequence."""
    mock_input.side_effect = None
    mock_input.return_value = "hello"
    final_message = fake_message_factory(content="recovered")
    mock_acompletion.side_effect = [
        make_transient_error(litellm.Timeout),
        make_transient_error(litellm.Timeout),
        fake_stream_factory(fake_response_factory(message=final_message)),
    ]

    await main.run_agent_loop([], config(fallbacks=["backup-model"]))

    assert [item.kwargs["model"] for item in mock_acompletion.await_args_list] == [
        "test-model",
        "backup-model",
        "test-model",
    ]
    mock_retry_sleep.assert_awaited_once_with(1)
    warning_events = [item.args[0] for item in mock_logger.warning.call_args_list]
    assert warning_events == [
        "llm.request.failed",
        "llm.fallback.selected",
        "llm.request.failed",
        "llm.retry.scheduled",
    ]
    fallback_metadata = mock_logger.warning.call_args_list[1].kwargs["extra"]
    assert fallback_metadata == {
        "model": "test-model",
        "next_model": "backup-model",
    }
    retry_metadata = mock_logger.warning.call_args_list[-1].kwargs["extra"]
    assert retry_metadata == {"attempt": 2, "delay_seconds": 1}
    assert "temporary API failure" not in repr(mock_logger.method_calls)


async def test_retry_exhaustion_uses_full_backoff_and_reraises_last_error(
    mock_input,
    mock_acompletion,
    mock_retry_sleep,
    mock_logger,
) -> None:
    """Use the 1/2/4/8/16/32 schedule and re-raise the final API failure."""
    mock_input.side_effect = None
    mock_input.return_value = "hello"
    errors = [
        litellm.Timeout(
            message=f"temporary API failure {attempt}",
            llm_provider="test-provider",
            model="test-model",
        )
        for attempt in range(7)
    ]
    mock_acompletion.side_effect = errors

    with pytest.raises(litellm.Timeout) as exc_info:
        await main.run_agent_loop([], config())

    assert exc_info.value is errors[-1]
    assert mock_acompletion.await_count == 7
    assert mock_retry_sleep.await_args_list == [
        call(1),
        call(2),
        call(4),
        call(8),
        call(16),
        call(32),
    ]
    mock_logger.error.assert_called_once_with(
        "llm.retries_exhausted", extra={"max_attempts": 7}
    )
    assert "temporary API failure" not in repr(mock_logger.method_calls)
