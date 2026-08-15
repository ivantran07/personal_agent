"""Tests for main.run_tool_call — see test/TEST_PLAN.md, section `test_run_tool_call.py`."""

import main


async def test_unknown_tool_returns_error_message(patched_tools, tool_call_factory):
    """Case 1: a tool name absent from main.TOOLS returns a "does not exist"
    content string that lists TOOLS.keys(), without raising.
    """
    patched_tools["known_tool"] = {"function": lambda: None, "schema": {}}
    tool = tool_call_factory(id="call_1", name="does_not_exist", arguments="{}")

    result = await main.run_tool_call(tool)

    assert result == {
        "tool_call_id": "call_1",
        "role": "tool",
        "content": "Tool does_not_exist does not exist. The list of available tools are ['known_tool']",
    }


async def test_known_tool_happy_path(patched_tools, tool_call_factory):
    """Case 2: a registered tool is looked up, its JSON arguments parsed and
    passed as kwargs, and the result returned as
    {"tool_call_id": ..., "role": "tool", "content": str(result)}.
    """
    patched_tools["known_tool"] = {"function": lambda: None, "schema": {}}
    tool = tool_call_factory(id="call_1", name="known_tool", arguments="{}")

    result = await main.run_tool_call(tool)

    assert result == {
        "tool_call_id": "call_1",
        "role": "tool",
        "content": "None",
    }


async def test_non_string_return_value_is_stringified(patched_tools, tool_call_factory):
    """Case 3: a tool returning a non-string (e.g. an int or dict) has its
    result wrapped in str(...) before being placed in `content`.
    """
    patched_tools["add"] = {"function": lambda x, y: x + y, "schema": {}}
    tool = tool_call_factory(id="call_1", name="add", arguments='{"x": 1, "y": 2}')

    result = await main.run_tool_call(tool)

    assert result == {
        "tool_call_id": "call_1",
        "role": "tool",
        "content": "3",
    }


async def test_malformed_json_arguments_are_caught(patched_tools, tool_call_factory):
    """Case 4: invalid JSON in tool.function.arguments is caught; content
    startswith "An exception occured: " and the tool function is never called.
    """
    function_called = False

    def add(x, y):
        nonlocal function_called
        function_called = True
        return x + y

    # Must be registered under the SAME name the tool call uses ("add"),
    # otherwise "name not in TOOLS" fires first and json.loads is never
    # reached at all.
    patched_tools["add"] = {"function": add, "schema": {}}
    tool = tool_call_factory(id="call_1", name="add", arguments='{"x": 1 "y": 2}')

    result = await main.run_tool_call(tool)

    assert result["content"].startswith("An exception occured: ")
    assert not function_called


async def test_tool_function_raises_value_error(patched_tools, tool_call_factory):
    """Case 5: a tool raising ValueError("boom") produces the exact content
    "An exception occured: boom. Try differently."
    """

    def func():
        raise ValueError("boom")

    patched_tools["add"] = {"function": func, "schema": {}}
    tool = tool_call_factory(id="call_1", name="add", arguments="{}")

    result = await main.run_tool_call(tool)

    assert result == {
        "tool_call_id": "call_1",
        "role": "tool",
        "content": "An exception occured: boom. Try differently.",
    }


async def test_tool_function_raises_type_error_on_bad_kwargs(
    patched_tools, tool_call_factory
):
    """Case 6: JSON arguments missing a required kwarg (or with an extra one)
    causes the underlying call to raise TypeError, wrapped the same way as case 5.
    """

    def add(x: int, y: int) -> int:
        return x + y

    patched_tools["add"] = {"function": add, "schema": {}}
    tool = tool_call_factory(id="call_1", name="add", arguments='{"x": 1}')

    result = await main.run_tool_call(tool)

    # Exact TypeError wording can vary by Python version, so check the
    # wrapping/shape precisely and the distinctive substring rather than the
    # full message.
    assert result["tool_call_id"] == "call_1"
    assert result["role"] == "tool"
    assert result["content"].startswith("An exception occured: ")
    assert result["content"].endswith("Try differently.")
    assert "missing" in result["content"]


async def test_success_path_logs_tool_name_without_arguments(
    patched_tools, tool_call_factory, mock_logger
):
    """Case 7: successful execution emits metadata without argument values."""
    patched_tools["add"] = {"function": lambda x, y: x + y, "schema": {}}
    arguments = '{"x": 1, "y": 2}'
    tool = tool_call_factory(id="call_1", name="add", arguments=arguments)

    await main.run_tool_call(tool)

    mock_logger.info.assert_called_once_with(
        "tool.completed", extra={"tool_name": "add"}
    )
    assert arguments not in repr(mock_logger.info.call_args)


async def test_failure_paths_log_sanitized_metadata(
    patched_tools, tool_call_factory, mock_logger
):
    """Case 8: unknown and failed tools emit metadata-only events."""
    unknown_name = "prompt-derived-secret"
    unknown_tool = tool_call_factory(id="call_1", name=unknown_name, arguments="{}")
    await main.run_tool_call(unknown_tool)

    mock_logger.warning.assert_called_once_with("tool.unknown")
    assert unknown_name not in repr(mock_logger.warning.call_args)

    def raises():
        raise ValueError("secret failure details")

    patched_tools["raises"] = {"function": raises, "schema": {}}
    arguments = "{}"
    raising_tool = tool_call_factory(id="call_2", name="raises", arguments=arguments)
    await main.run_tool_call(raising_tool)

    (event,) = mock_logger.error.call_args.args
    metadata = mock_logger.error.call_args.kwargs["extra"]
    assert event == "tool.failed"
    assert metadata["tool_name"] == "raises"
    assert metadata["error_type"] == "ValueError"
    assert metadata["traceback"]
    assert "secret failure details" not in repr(metadata)


async def test_result_dict_has_exactly_expected_keys(patched_tools, tool_call_factory):
    """Case 9: regardless of which branch runs (unknown/success/exception), the
    returned dict has exactly the keys {"tool_call_id", "role", "content"} and
    role == "tool".
    """
    patched_tools["add"] = {"function": lambda x, y: x + y, "schema": {}}

    def raises():
        raise ValueError("boom")

    patched_tools["raises"] = {"function": raises, "schema": {}}

    scenarios = [
        tool_call_factory(id="call_1", name="add", arguments='{"x": 1, "y": 2}'),
        tool_call_factory(id="call_2", name="does_not_exist", arguments="{}"),
        tool_call_factory(id="call_3", name="raises", arguments="{}"),
    ]

    for tool in scenarios:
        result = await main.run_tool_call(tool)
        assert set(result.keys()) == {"tool_call_id", "role", "content"}
        assert result["role"] == "tool"
