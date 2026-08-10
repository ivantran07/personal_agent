"""Shared fixtures for the main.py test suite.

See test/TEST_PLAN.md for the full design rationale for each fixture below,
and test/TESTING_TUTORIAL.md if you're implementing these for the first time.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
import yaml
from litellm import ModelResponse


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


class FakeStream:
    """Async iterable stand-in for LiteLLM's streaming response wrapper."""

    def __init__(self, model, chunks):
        self.model = model
        self._chunks = chunks

    def __aiter__(self):
        async def iterate():
            for chunk in self._chunks:
                yield chunk

        return iterate()


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
    empty_choices=False) -> ModelResponse-spec mock with `.model` and `.choices`.

    Using a spec makes the fake satisfy main.py's ``isinstance(...,
    ModelResponse)`` safety guard while retaining lightweight test messages.
    """

    def _make(
        model="test-model", finish_reason="stop", message=None, empty_choices=False
    ):
        response = Mock(spec=ModelResponse)
        response.model = model
        response.choices = (
            []
            if empty_choices
            else [SimpleNamespace(finish_reason=finish_reason, message=message)]
        )
        return response

    return _make


@pytest.fixture
def fake_chunk_factory():
    """Create a streaming chunk with optional choices, delta, and text content."""

    def _make(content=None, *, has_choices=True, has_delta=True):
        if not has_choices:
            return SimpleNamespace(choices=[])

        delta = SimpleNamespace(content=content) if has_delta else None
        return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])

    return _make


@pytest.fixture
def fake_stream_factory(fake_chunk_factory):
    """Create an async stream whose chunks rebuild to ``response`` in tests."""

    def _make(response, model="test-model", chunks=None):
        chunks = chunks or [fake_chunk_factory()]
        for chunk in chunks:
            chunk.final_response = response
        return FakeStream(model, chunks)

    return _make


@pytest.fixture(autouse=True)
def mock_stream_chunk_builder(monkeypatch):
    """Rebuild fake streams to their preconfigured final response."""

    mock = Mock(side_effect=lambda chunks, **_: chunks[0].final_response)
    monkeypatch.setattr("main.litellm.stream_chunk_builder", mock)
    return mock


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
