"""Shared fixtures for the main.py test suite.

See test/TEST_PLAN.md for the full design rationale for each fixture below,
and test/TESTING_TUTORIAL.md if you're implementing these for the first time.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
import yaml


class StopLoop(Exception):
    """Sentinel exception used to break main()'s `while True` loop in tests."""


class FakeMessage:
    """Stand-in for litellm's Message object.

    Needs `.content`, `.tool_calls`, and a real `.model_dump()` method (main.py
    calls `message.model_dump()` and appends the result to `messages`).
    """

    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self):
        return {
            "role": "assistant",
            "content": self.content,
            "tool_calls": self.tool_calls,
        }


@pytest.fixture(autouse=True)
def mock_input(monkeypatch):
    """Patch builtins.input so no test can hang on real stdin.

    Default behavior: raise AssertionError if called unexpectedly. Tests that
    need a specific prompt response should override `.return_value` /
    `.side_effect` on the returned mock.

    Patch target: "builtins.input" (main.py calls the builtin directly, it's
    never imported by name).
    """
    mock = MagicMock(side_effect=AssertionError("input() called unexpectedly in test"))
    monkeypatch.setattr("builtins.input", mock)
    return mock


@pytest.fixture(autouse=True)
def stub_load_dotenv(monkeypatch):
    """Patch main.load_dotenv with a no-op Mock so main() tests never touch a
    real .env file regardless of the dev machine's state.
    """
    mock = Mock()
    monkeypatch.setattr("main.load_dotenv", mock)
    return mock


@pytest.fixture
def tool_call_factory():
    """Factory fixture: _make(id="call_1", name="tool", arguments="{}") -> object
    shaped like a litellm tool call, i.e. exposes `.id`, `.function.name`,
    `.function.arguments`. A SimpleNamespace-of-SimpleNamespace is sufficient.
    """

    def _make(id="call_1", name="tool", arguments="{}"):
        return SimpleNamespace(
            id=id, function=SimpleNamespace(name=name, arguments=arguments)
        )

    return _make


@pytest.fixture
def fake_message_factory():
    """Factory fixture: _make(content=None, tool_calls=None) -> FakeMessage instance.

    tool_calls, when provided, should be a list of objects shaped like
    tool_call_factory's output.
    """

    def _make(content=None, tool_calls=None):
        return FakeMessage(content=content, tool_calls=tool_calls)

    return _make


@pytest.fixture
def fake_response_factory():
    """Factory fixture: _make(model="test-model", finish_reason="stop", message=None,
    empty_choices=False) -> object shaped like a litellm ModelResponse, i.e.
    exposes `.model` and `.choices` (a list of objects with `.finish_reason`
    and `.message`, or an empty list when empty_choices=True).
    """

    def _make(
        model="test-model", finish_reason="stop", message=None, empty_choices=False
    ):
        return SimpleNamespace(
            model=model,
            choices=[]
            if empty_choices
            else [SimpleNamespace(finish_reason=finish_reason, message=message)],
        )

    return _make


@pytest.fixture
def patched_tools(monkeypatch):
    """Replace main.TOOLS with a fresh, empty dict for the duration of the test.

    Tests populate it directly, e.g.:
        patched_tools["double"] = {"function": lambda n: n * 2, "schema": {}}

    Fake tool functions MUST be plain sync callables (not `async def`) since
    run_tool_call dispatches them via asyncio.to_thread.
    """
    tools = {}
    monkeypatch.setattr("main.TOOLS", tools)
    return tools


@pytest.fixture
def mock_acompletion(monkeypatch):
    """Patch main.acompletion with an AsyncMock and return it.

    Patch target: "main.acompletion" (main.py does `from litellm import
    acompletion`, binding the name into its own module namespace — patching
    litellm.acompletion directly would not affect main's already-imported copy).
    """
    mock = AsyncMock()
    monkeypatch.setattr("main.acompletion", mock)
    return mock


@pytest.fixture
def write_config(tmp_path, monkeypatch):
    """Factory fixture: _write(config_dict) -> None.

    chdirs into tmp_path (via monkeypatch, auto-restored after the test) and
    writes config_dict as config.yaml there, so main()'s hardcoded relative
    `open("config.yaml")` read finds it instead of the real project config.
    """
    monkeypatch.chdir(tmp_path)

    def _write(config_dict):
        with open("config.yaml", "w", encoding="utf8") as f:
            yaml.safe_dump(config_dict, f)

    return _write
