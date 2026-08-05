"""Tests for main.run_agent_loop — see test/TEST_PLAN.md, section `test_run_agent_loop.py`.

Every test here needs a `config` dict with at least `max_iterations` and
`max_completion_tokens` set, and drives behavior entirely through
`mock_acompletion.side_effect` / `.return_value` plus `fake_response_factory` /
`fake_message_factory`.
"""

import pytest

import main


async def test_input_called_once_for_user_prompt(
    mock_input, mock_acompletion, fake_response_factory, fake_message_factory
):
    """Case 1: input() is called exactly once (before the iteration loop), and
    its return value is appended to messages as {"role": "user", "content": ...}.
    """
    mock_input.side_effect = None
    mock_input.return_value = "hello"

    fake_message = fake_message_factory(content="ok", tool_calls=None)
    mock_acompletion.return_value = fake_response_factory(message=fake_message)

    messages = []
    config = {"model": "test-model", "max_completion_tokens": 100, "max_iterations": 5}

    await main.run_agent_loop(messages, config)

    mock_input.assert_called_once()
    assert messages[0] == {"role": "user", "content": "hello"}


async def test_breaks_when_no_tool_calls(
    mock_input, mock_acompletion, fake_response_factory, fake_message_factory
):
    """Case 2: a response whose message.tool_calls is falsy ends the loop after
    exactly one acompletion call.
    """
    mock_input.side_effect = None
    mock_input.return_value = "hello"

    fake_message = fake_message_factory(content="done", tool_calls=None)
    mock_acompletion.return_value = fake_response_factory(message=fake_message)

    messages = []
    config = {"model": "test-model", "max_completion_tokens": 100, "max_iterations": 5}

    await main.run_agent_loop(messages, config)

    assert mock_acompletion.call_count == 1


async def test_model_name_printed_once_across_iterations(
    mock_input,
    mock_acompletion,
    fake_response_factory,
    fake_message_factory,
    patched_tools,
    tool_call_factory,
    capsys,
):
    """Case 3: across two acompletion calls (first has tool_calls, second doesn't),
    the yellow "MODEL: ..." line is printed exactly once.
    """
    mock_input.side_effect = None
    mock_input.return_value = "hello"

    patched_tools["noop"] = {"function": lambda: None, "schema": {}}
    tool_call = tool_call_factory(id="call_1", name="noop", arguments="{}")

    message_1 = fake_message_factory(content="", tool_calls=[tool_call])
    message_2 = fake_message_factory(content="done", tool_calls=None)
    mock_acompletion.side_effect = [
        fake_response_factory(model="test-model", message=message_1),
        fake_response_factory(model="test-model", message=message_2),
    ]

    messages = []
    config = {"model": "test-model", "max_completion_tokens": 100, "max_iterations": 5}

    await main.run_agent_loop(messages, config)

    assert mock_acompletion.call_count == 2
    captured = capsys.readouterr()
    assert captured.out.count(f"{main.YELLOW}MODEL: test-model{main.RESET}") == 1


async def test_tool_call_iteration_appends_messages_in_order(
    mock_input,
    mock_acompletion,
    fake_response_factory,
    fake_message_factory,
    patched_tools,
    tool_call_factory,
):
    """Case 4: response 1 has tool_calls mapped to a patched_tools entry,
    response 2 doesn't -> messages end up
    [user, assistant_1, tool_result(s), assistant_2] in that order, and the
    second acompletion call's `messages` kwarg is longer than the first's.
    """
    mock_input.side_effect = None
    mock_input.return_value = "hello"

    patched_tools["add"] = {"function": lambda x, y: x + y, "schema": {}}
    tool_call = tool_call_factory(id="call_1", name="add", arguments='{"x": 1, "y": 2}')

    message_1 = fake_message_factory(content="", tool_calls=[tool_call])
    message_2 = fake_message_factory(content="done", tool_calls=None)
    responses = [
        fake_response_factory(message=message_1),
        fake_response_factory(message=message_2),
    ]

    # call_args_list stores a REFERENCE to the same `messages` list each time,
    # so comparing lengths after the loop finishes would compare the same,
    # fully-mutated list to itself. Snapshot the length live instead.
    seen_message_lengths = []

    def fake_acompletion(*args, **kwargs):
        seen_message_lengths.append(len(kwargs["messages"]))
        return responses.pop(0)

    mock_acompletion.side_effect = fake_acompletion

    messages = []
    config = {"model": "test-model", "max_completion_tokens": 100, "max_iterations": 5}

    await main.run_agent_loop(messages, config)

    assert messages == [
        {"role": "user", "content": "hello"},
        message_1.model_dump(),
        {"tool_call_id": "call_1", "role": "tool", "content": "3"},
        message_2.model_dump(),
    ]
    assert seen_message_lengths[1] > seen_message_lengths[0]


async def test_length_finish_reason_retries_without_appending_assistant_message(
    mock_input, mock_acompletion, fake_response_factory, fake_message_factory
):
    """Case 5: finish_reason == "length" appends the "cut off, retry briefly"
    user message and continues; no assistant message is appended for that
    iteration (total message count reflects: user + retry-user + final assistant).
    """
    mock_input.side_effect = None
    mock_input.return_value = "hello"

    final_message = fake_message_factory(content="done", tool_calls=None)
    mock_acompletion.side_effect = [
        fake_response_factory(finish_reason="length", message=None),
        fake_response_factory(finish_reason="stop", message=final_message),
    ]

    messages = []
    config = {"model": "test-model", "max_completion_tokens": 100, "max_iterations": 5}

    await main.run_agent_loop(messages, config)

    assert mock_acompletion.call_count == 2
    assert messages == [
        {"role": "user", "content": "hello"},
        {
            "role": "user",
            "content": "Your previous response was cut off for being too long. Answer again, briefly, without re-deriving your reasoning.",
        },
        final_message.model_dump(),
    ]


async def test_empty_choices_raises_index_error(
    mock_input, mock_acompletion, fake_response_factory
):
    """Case 6: KNOWN BUG (see TEST_PLAN.md Context) — response.choices == []
    causes an IndexError on line 93 (finish_reason check) before the intended
    "No choices" guard on line 102 ever runs. This test documents the current,
    buggy behavior; it is not something to fix here.
    """
    mock_input.side_effect = None
    mock_input.return_value = "hello"

    mock_acompletion.return_value = fake_response_factory(empty_choices=True)

    messages = []
    config = {"model": "test-model", "max_completion_tokens": 100, "max_iterations": 5}

    with pytest.raises(IndexError):
        await main.run_agent_loop(messages, config)


async def test_max_iterations_exhausted_prints_message(
    mock_input,
    mock_acompletion,
    fake_response_factory,
    fake_message_factory,
    patched_tools,
    tool_call_factory,
    capsys,
):
    """Case 7: every response has non-empty tool_calls -> the for/else on the
    iteration loop fires, "Max iterations reached without a final answer" is
    printed, and acompletion is called exactly config["max_iterations"] times.
    """
    mock_input.side_effect = None
    mock_input.return_value = "hello"

    patched_tools["noop"] = {"function": lambda: None, "schema": {}}
    tool_call = tool_call_factory(id="call_1", name="noop", arguments="{}")
    fake_message = fake_message_factory(content="", tool_calls=[tool_call])
    mock_acompletion.return_value = fake_response_factory(message=fake_message)

    messages = []
    config = {"model": "test-model", "max_completion_tokens": 100, "max_iterations": 3}

    await main.run_agent_loop(messages, config)

    assert mock_acompletion.call_count == 3
    captured = capsys.readouterr()
    assert (
        f"{main.YELLOW}Max iterations reached without a final answer{main.RESET}"
        in captured.out
    )


@pytest.mark.parametrize("content", ["hello", None, ""])
async def test_content_print_presence_and_absence(
    content,
    mock_input,
    mock_acompletion,
    fake_response_factory,
    fake_message_factory,
    capsys,
):
    """Case 8: message.content truthy prints the green MODEL content line;
    falsy (None or "") means that line is absent.
    """
    mock_input.side_effect = None
    mock_input.return_value = "hi"

    fake_message = fake_message_factory(content=content, tool_calls=None)
    mock_acompletion.return_value = fake_response_factory(message=fake_message)

    messages = []
    config = {"model": "test-model", "max_completion_tokens": 100, "max_iterations": 5}

    await main.run_agent_loop(messages, config)

    captured = capsys.readouterr()
    expected_line = f"{main.GREEN}MODEL{main.RESET}: {content}"
    if content:
        assert expected_line in captured.out
    else:
        assert expected_line not in captured.out


async def test_acompletion_called_with_expected_kwargs_and_defaults(
    mock_input, mock_acompletion, fake_response_factory, fake_message_factory
):
    """Case 9: acompletion is called with model, max_completion_tokens,
    tools=main.TOOL_SCHEMAS, and fallbacks/api_base/api_key derived via
    config.get(...) — including the case where config omits those keys
    entirely, verifying the defaults ([]/None/None) rather than a KeyError.
    """
    mock_input.side_effect = None
    mock_input.return_value = "hi"

    fake_message = fake_message_factory(content="done", tool_calls=None)
    mock_acompletion.return_value = fake_response_factory(message=fake_message)

    config = {
        "model": "test-model",
        "max_completion_tokens": 100,
        "max_iterations": 5,
        "fallbacks": ["backup-model"],
        "api_base": "https://example.test",
        "api_key": "secret",
    }
    await main.run_agent_loop([], config)

    call_kwargs = mock_acompletion.call_args.kwargs
    assert call_kwargs["model"] == "test-model"
    assert call_kwargs["max_completion_tokens"] == 100
    assert call_kwargs["tools"] == main.TOOL_SCHEMAS
    assert call_kwargs["fallbacks"] == ["backup-model"]
    assert call_kwargs["api_base"] == "https://example.test"
    assert call_kwargs["api_key"] == "secret"

    mock_acompletion.reset_mock()
    minimal_config = {
        "model": "test-model",
        "max_completion_tokens": 100,
        "max_iterations": 5,
    }
    await main.run_agent_loop([], minimal_config)

    call_kwargs = mock_acompletion.call_args.kwargs
    assert call_kwargs["fallbacks"] == []
    assert call_kwargs["api_base"] is None
    assert call_kwargs["api_key"] is None


async def test_model_dump_return_value_lands_in_messages_unchanged(
    mock_input, mock_acompletion, fake_response_factory, fake_message_factory
):
    """Case 10: whatever FakeMessage.model_dump() returns is exactly what gets
    appended into `messages` — no re-serialization or transformation happens.
    """
    mock_input.side_effect = None
    mock_input.return_value = "hi"

    fake_message = fake_message_factory(content="done", tool_calls=None)
    expected_dump = fake_message.model_dump()
    mock_acompletion.return_value = fake_response_factory(message=fake_message)

    messages = []
    config = {"model": "test-model", "max_completion_tokens": 100, "max_iterations": 5}

    await main.run_agent_loop(messages, config)

    assert messages[-1] == expected_dump
