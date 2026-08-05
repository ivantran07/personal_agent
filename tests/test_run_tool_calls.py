"""Tests for main.run_tool_calls — see test/TEST_PLAN.md, section `test_run_tool_calls.py`.

Read the "Ordering subtlety" note in TEST_PLAN.md before writing case 8 — the
final result order is NOT the original interleaved tool_calls order.
"""

import threading

import pytest

import main


@pytest.fixture
def confirm_tools(monkeypatch):
    """Monkeypatch main.CONFIRM_TOOLS to a known, test-local set (e.g.
    {"needs_confirm"}) so tests don't depend on the real tool-name set.
    """
    tools = {"needs_confirm"}
    monkeypatch.setattr("main.CONFIRM_TOOLS", tools)
    return tools


async def test_empty_tool_calls_returns_empty_list(fake_message_factory, mock_input):
    """Case 1: message.tool_calls=[] -> returns [], input() never called."""
    fake_message = fake_message_factory(content="", tool_calls=[])

    results = await main.run_tool_calls(fake_message)

    assert results == []
    mock_input.assert_not_called()


async def test_all_normal_tools_run_without_prompt(
    confirm_tools, patched_tools, tool_call_factory, fake_message_factory, mock_input
):
    """Case 2: tools not in CONFIRM_TOOLS run with no input() prompt; results
    come back in original order.
    """
    patched_tools["first"] = {"function": lambda: "a", "schema": {}}
    patched_tools["second"] = {"function": lambda: "b", "schema": {}}
    patched_tools["third"] = {"function": lambda: "c", "schema": {}}
    tool_calls = [
        tool_call_factory(id="call_1", name="first", arguments="{}"),
        tool_call_factory(id="call_2", name="second", arguments="{}"),
        tool_call_factory(id="call_3", name="third", arguments="{}"),
    ]
    fake_message = fake_message_factory(content="", tool_calls=tool_calls)

    results = await main.run_tool_calls(fake_message)

    mock_input.assert_not_called()
    assert [r["tool_call_id"] for r in results] == ["call_1", "call_2", "call_3"]


async def test_confirmed_tool_runs(
    confirm_tools, patched_tools, tool_call_factory, fake_message_factory, mock_input
):
    """Case 3: a CONFIRM_TOOLS tool + input() returning "yes" -> the tool
    actually runs and its real output surfaces in content.
    """
    tool_runs = False

    def add(x: int, y: int) -> int:
        nonlocal tool_runs
        tool_runs = True
        return x + y

    patched_tools["add"] = {"function": add, "schema": {}}
    confirm_tools.add("add")
    mock_input.side_effect = None
    mock_input.return_value = "yes"

    tool_call = tool_call_factory(id="call_1", name="add", arguments='{"x": 1, "y": 2}')
    fake_message = fake_message_factory(content="", tool_calls=[tool_call])

    results = await main.run_tool_calls(fake_message)

    assert tool_runs
    assert results[0]["content"] == "3"


async def test_declined_tool_does_not_run(
    confirm_tools, patched_tools, tool_call_factory, fake_message_factory, mock_input
):
    """Case 4: a CONFIRM_TOOLS tool + input() returning "no" -> tool function
    never called; result is exactly the cancellation message dict.
    """
    tool_runs = False

    def add(x: int, y: int) -> int:
        nonlocal tool_runs
        tool_runs = True
        return x + y

    patched_tools["add"] = {"function": add, "schema": {}}
    confirm_tools.add("add")
    mock_input.side_effect = None
    mock_input.return_value = "no"

    tool_call = tool_call_factory(id="call_1", name="add", arguments='{"x": 1, "y": 2}')
    fake_message = fake_message_factory(content="", tool_calls=[tool_call])

    results = await main.run_tool_calls(fake_message)

    assert not tool_runs
    assert results == [
        {
            "tool_call_id": "call_1",
            "role": "tool",
            "content": "Action cancelled: user did not confirm",
        }
    ]


@pytest.mark.parametrize("answer", ["yes", "Yes", " yes \n", "YES"])
async def test_accept_variants(
    answer,
    confirm_tools,
    patched_tools,
    tool_call_factory,
    fake_message_factory,
    mock_input,
):
    """Case 5: these variants are all treated as acceptance (strip().lower() == "yes")."""
    tool_runs = False

    def confirm_me():
        nonlocal tool_runs
        tool_runs = True
        return "ok"

    patched_tools["confirm_me"] = {"function": confirm_me, "schema": {}}
    confirm_tools.add("confirm_me")
    mock_input.side_effect = None
    mock_input.return_value = answer

    tool_call = tool_call_factory(id="call_1", name="confirm_me", arguments="{}")
    fake_message = fake_message_factory(content="", tool_calls=[tool_call])

    results = await main.run_tool_calls(fake_message)

    assert tool_runs
    assert results[0]["content"] == "ok"


@pytest.mark.parametrize("answer", ["", "y", "no", "yess"])
async def test_decline_variants(
    answer,
    confirm_tools,
    patched_tools,
    tool_call_factory,
    fake_message_factory,
    mock_input,
):
    """Case 6: these variants are all treated as decline."""
    tool_runs = False

    def confirm_me():
        nonlocal tool_runs
        tool_runs = True
        return "ok"

    patched_tools["confirm_me"] = {"function": confirm_me, "schema": {}}
    confirm_tools.add("confirm_me")
    mock_input.side_effect = None
    mock_input.return_value = answer

    tool_call = tool_call_factory(id="call_1", name="confirm_me", arguments="{}")
    fake_message = fake_message_factory(content="", tool_calls=[tool_call])

    results = await main.run_tool_calls(fake_message)

    assert not tool_runs
    assert results == [
        {
            "tool_call_id": "call_1",
            "role": "tool",
            "content": "Action cancelled: user did not confirm",
        }
    ]


async def test_confirmation_prompt_text_exact(
    confirm_tools, patched_tools, tool_call_factory, fake_message_factory, mock_input
):
    """Case 7: the input() prompt matches exactly
    f"{main.RED}VALIDATION{main.RESET}: Run {name} with arguments {arguments}? Type 'yes' to confirm: ".
    """
    patched_tools["confirm_me"] = {"function": lambda: None, "schema": {}}
    confirm_tools.add("confirm_me")
    mock_input.side_effect = None
    mock_input.return_value = "yes"

    arguments = '{"x": 1}'
    tool_call = tool_call_factory(id="call_1", name="confirm_me", arguments=arguments)
    fake_message = fake_message_factory(content="", tool_calls=[tool_call])

    await main.run_tool_calls(fake_message)

    expected_prompt = (
        f"{main.RED}VALIDATION{main.RESET}: Run confirm_me with arguments "
        f"{arguments}? Type 'yes' to confirm: "
    )
    mock_input.assert_called_once_with(expected_prompt)


async def test_mixed_decline_and_normal_ordering(
    confirm_tools, patched_tools, tool_call_factory, fake_message_factory, mock_input
):
    """Case 8: decline, normal, decline, normal -> final tool_call_id order is
    [declined_1, declined_2, normal_1, normal_2] (all cancellations first, then
    all approved results) — NOT the original interleaved order. This pins down
    current behavior, not a spec requirement.
    """
    confirm_tools.add("confirm_me")
    patched_tools["confirm_me"] = {"function": lambda: "confirmed", "schema": {}}
    patched_tools["normal"] = {"function": lambda: "normal", "schema": {}}
    mock_input.side_effect = None
    mock_input.return_value = "no"

    tool_calls = [
        tool_call_factory(id="declined_1", name="confirm_me", arguments="{}"),
        tool_call_factory(id="normal_1", name="normal", arguments="{}"),
        tool_call_factory(id="declined_2", name="confirm_me", arguments="{}"),
        tool_call_factory(id="normal_2", name="normal", arguments="{}"),
    ]
    fake_message = fake_message_factory(content="", tool_calls=tool_calls)

    results = await main.run_tool_calls(fake_message)

    assert [r["tool_call_id"] for r in results] == [
        "declined_1",
        "declined_2",
        "normal_1",
        "normal_2",
    ]


async def test_confirmed_and_normal_preserve_relative_order(
    confirm_tools, patched_tools, tool_call_factory, fake_message_factory, mock_input
):
    """Case 9: when everything is approved (mix of confirmed-accepted and
    normal tools), result order matches the original relative order (gather
    preserves input order).
    """
    confirm_tools.add("confirm_me")
    patched_tools["confirm_me"] = {"function": lambda: "confirmed", "schema": {}}
    patched_tools["normal"] = {"function": lambda: "normal", "schema": {}}
    mock_input.side_effect = None
    mock_input.return_value = "yes"

    tool_calls = [
        tool_call_factory(id="call_1", name="confirm_me", arguments="{}"),
        tool_call_factory(id="call_2", name="normal", arguments="{}"),
        tool_call_factory(id="call_3", name="confirm_me", arguments="{}"),
        tool_call_factory(id="call_4", name="normal", arguments="{}"),
    ]
    fake_message = fake_message_factory(content="", tool_calls=tool_calls)

    results = await main.run_tool_calls(fake_message)

    assert [r["tool_call_id"] for r in results] == [
        "call_1",
        "call_2",
        "call_3",
        "call_4",
    ]


async def test_concurrent_execution_via_event_rendezvous(
    patched_tools, tool_call_factory, fake_message_factory
):
    """Case 10: two approved tools each set their own threading.Event and then
    .wait() on the other's event with a bounded timeout. If execution were
    actually sequential (not concurrent via asyncio.to_thread), the first tool
    would block until timeout — deterministic, non-flaky proof of concurrency.
    """
    event_a = threading.Event()
    event_b = threading.Event()

    def tool_a():
        event_a.set()
        return event_b.wait(timeout=2)

    def tool_b():
        event_b.set()
        return event_a.wait(timeout=2)

    patched_tools["tool_a"] = {"function": tool_a, "schema": {}}
    patched_tools["tool_b"] = {"function": tool_b, "schema": {}}

    tool_calls = [
        tool_call_factory(id="call_a", name="tool_a", arguments="{}"),
        tool_call_factory(id="call_b", name="tool_b", arguments="{}"),
    ]
    fake_message = fake_message_factory(content="", tool_calls=tool_calls)

    results = await main.run_tool_calls(fake_message)

    contents = {r["tool_call_id"]: r["content"] for r in results}
    assert contents["call_a"] == "True"
    assert contents["call_b"] == "True"


async def test_approved_tool_raises_still_surfaces(
    patched_tools, tool_call_factory, fake_message_factory
):
    """Case 11: an approved tool that raises still produces a correctly
    exception-wrapped content with the right tool_call_id in the aggregate result.
    """

    def raises():
        raise ValueError("boom")

    patched_tools["raises"] = {"function": raises, "schema": {}}
    tool_call = tool_call_factory(id="call_1", name="raises", arguments="{}")
    fake_message = fake_message_factory(content="", tool_calls=[tool_call])

    results = await main.run_tool_calls(fake_message)

    assert results == [
        {
            "tool_call_id": "call_1",
            "role": "tool",
            "content": "An exception occured: boom. Try differently.",
        }
    ]


async def test_all_declined_returns_only_cancellations(
    confirm_tools, patched_tools, tool_call_factory, fake_message_factory, mock_input
):
    """Case 12: every tool call is a declined CONFIRM_TOOLS entry -> approved_calls
    stays empty, asyncio.gather() with no args returns [] without hanging, and
    the final result is exactly the cancellation entries.
    """
    confirm_tools.add("confirm_me")
    patched_tools["confirm_me"] = {"function": lambda: None, "schema": {}}
    mock_input.side_effect = None
    mock_input.return_value = "no"

    tool_calls = [
        tool_call_factory(id="call_1", name="confirm_me", arguments="{}"),
        tool_call_factory(id="call_2", name="confirm_me", arguments="{}"),
    ]
    fake_message = fake_message_factory(content="", tool_calls=tool_calls)

    results = await main.run_tool_calls(fake_message)

    assert results == [
        {
            "tool_call_id": "call_1",
            "role": "tool",
            "content": "Action cancelled: user did not confirm",
        },
        {
            "tool_call_id": "call_2",
            "role": "tool",
            "content": "Action cancelled: user did not confirm",
        },
    ]
