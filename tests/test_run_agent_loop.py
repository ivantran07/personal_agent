"""Tests for the async streaming behavior of ``main.run_agent_loop``."""

import main


def config(**overrides):
    return {
        "model": "test-model",
        "max_completion_tokens": 100,
        "max_iterations": 5,
        **overrides,
    }


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


async def test_empty_rebuilt_choices_prints_message_and_returns(
    mock_input, mock_acompletion, fake_response_factory, fake_stream_factory, capsys
):
    mock_input.side_effect = None
    mock_input.return_value = "hello"
    mock_acompletion.return_value = fake_stream_factory(
        fake_response_factory(empty_choices=True)
    )

    await main.run_agent_loop([], config())

    assert "No choices" in capsys.readouterr().out


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


async def test_max_iterations_prints_message_for_repeated_tool_calls(
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
    message = fake_message_factory(tool_calls=[tool_call_factory(name="noop")])
    mock_acompletion.return_value = fake_stream_factory(
        fake_response_factory(message=message)
    )

    await main.run_agent_loop([], config(max_iterations=3))

    assert mock_acompletion.await_count == 3
    assert "Max iterations reached without a final answer" in capsys.readouterr().out


async def test_acompletion_uses_streaming_and_expected_configuration(
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
    expected_config = config(
        fallbacks=["backup-model"], api_base="https://example.test", api_key="secret"
    )

    await main.run_agent_loop([], expected_config)

    assert mock_acompletion.call_args.kwargs == {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}, message.model_dump()],
        "max_completion_tokens": 100,
        "tools": main.TOOL_SCHEMAS,
        "fallbacks": ["backup-model"],
        "api_base": "https://example.test",
        "api_key": "secret",
        "stream": True,
    }
