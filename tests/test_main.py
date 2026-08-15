"""Tests for main.main — see test/TEST_PLAN.md, section `test_main.py`.

Every test here drives main() via a mocked main.run_agent_loop (AsyncMock),
using `.side_effect` to unwind the infinite `while True` loop through the
StopLoop sentinel exception, and inspects the (messages, config) args it was
awaited with via `mock_run_agent_loop.call_args` / `.await_args_list`.
"""

import json
import logging
import sys
from unittest.mock import AsyncMock, Mock

import pytest
from conftest import StopLoop

import main
from logging_config import LOGGER_NAMESPACE


@pytest.fixture
def mock_run_agent_loop(monkeypatch):
    """Patch main.run_agent_loop with an AsyncMock and return it. Tests set
    `.side_effect = [None, ..., StopLoop()]` and wrap the `await main.main()`
    call in `pytest.raises(StopLoop)`.
    """
    mock = AsyncMock()
    monkeypatch.setattr("main.run_agent_loop", mock)
    return mock


def base_config():
    """Helper: returns a config dict mirroring config.yaml's real shape
    (active_profile, profiles, system_prompt, max_iterations, ...) for tests
    to customize per-case. Not a fixture — plain helper so tests can freely
    mutate their own copy.
    """
    return {
        "active_profile": "profile_a",
        "profiles": {
            "profile_a": {"model": "model-a"},
            "profile_b": {"model": "model-b"},
        },
        "max_completion_tokens": 4096,
        "system_prompt": None,
        "max_iterations": 5,
    }


def test_cli_runs_async_main(monkeypatch):
    """The synchronous console entry point delegates to asyncio.run(main())."""

    async def fake_main() -> None:
        pass

    mock_run = Mock(side_effect=lambda coroutine: coroutine.close())
    monkeypatch.setattr(main, "main", fake_main)
    monkeypatch.setattr(main.asyncio, "run", mock_run)

    main.cli()

    mock_run.assert_called_once()


def test_cli_logs_graceful_eof_shutdown(monkeypatch, mock_logger):
    """EOF exits cleanly and records a metadata-only shutdown reason."""

    async def fake_main() -> None:
        pass

    def raise_eof(coroutine):
        coroutine.close()
        raise EOFError

    monkeypatch.setattr(main, "main", fake_main)
    monkeypatch.setattr(main.asyncio, "run", raise_eof)

    main.cli()

    mock_logger.info.assert_called_once_with(
        "app.stopped", extra={"reason": "EOFError"}
    )


def test_invalid_log_level_fails_with_structured_json(monkeypatch, capsys):
    """Logging configuration failures still reach the JSON process boundary."""
    invalid_level = "secret-invalid-level"
    monkeypatch.setenv("LOG_LEVEL", invalid_level)

    try:
        with pytest.raises(SystemExit) as exc_info:
            main.cli()

        assert exc_info.value.code == 1
        output = capsys.readouterr().err
        record = json.loads(output)
        assert record["level"] == "CRITICAL"
        assert record["event"] == "app.failed"
        assert record["error_type"] == "ValueError"
        assert invalid_level not in output
    finally:
        app_logger = logging.getLogger(LOGGER_NAMESPACE)
        for handler in app_logger.handlers[:]:
            app_logger.removeHandler(handler)
            handler.close()
        app_logger.setLevel(logging.NOTSET)
        app_logger.propagate = True


async def test_default_profile_used_when_no_argv(
    monkeypatch, write_config, mock_run_agent_loop
):
    """Case 1: with no argv[1], the merged config passed to run_agent_loop
    equals {**base, **base["profiles"][active_profile]}.
    """
    config = base_config()
    write_config(config)
    monkeypatch.setattr(sys, "argv", ["main.py"])
    mock_run_agent_loop.side_effect = StopLoop()

    with pytest.raises(StopLoop):
        await main.main()

    expected_config = {**config, **config["profiles"]["profile_a"]}
    _, actual_config = mock_run_agent_loop.call_args.args
    assert actual_config == expected_config


async def test_argv_overrides_active_profile(
    monkeypatch, write_config, mock_run_agent_loop
):
    """Case 2: sys.argv[1] present -> that profile's overrides are used
    instead of config["active_profile"].
    """
    config = base_config()
    write_config(config)
    monkeypatch.setattr(sys, "argv", ["main.py", "profile_b"])
    mock_run_agent_loop.side_effect = StopLoop()

    with pytest.raises(StopLoop):
        await main.main()

    expected_config = {**config, **config["profiles"]["profile_b"]}
    _, actual_config = mock_run_agent_loop.call_args.args
    assert actual_config == expected_config


async def test_unknown_profile_name_raises_key_error(
    monkeypatch, write_config, mock_run_agent_loop
):
    """Case 3: argv[1] names a profile absent from config["profiles"] ->
    pytest.raises(KeyError).
    """
    config = base_config()
    write_config(config)
    monkeypatch.setattr(sys, "argv", ["main.py", "does_not_exist"])

    with pytest.raises(KeyError):
        await main.main()


async def test_system_prompt_truthy_adds_system_message(
    monkeypatch, write_config, mock_run_agent_loop
):
    """Case 4: a truthy system_prompt -> messages passed to run_agent_loop is
    [{"role": "system", "content": system_prompt}].
    """
    config = base_config()
    config["system_prompt"] = "You are a helpful assistant."
    write_config(config)
    monkeypatch.setattr(sys, "argv", ["main.py"])
    mock_run_agent_loop.side_effect = StopLoop()

    with pytest.raises(StopLoop):
        await main.main()

    actual_messages, _ = mock_run_agent_loop.call_args.args
    assert actual_messages == [
        {"role": "system", "content": "You are a helpful assistant."}
    ]


@pytest.mark.parametrize("system_prompt", [None, ""])
async def test_system_prompt_falsy_no_system_message(
    system_prompt, monkeypatch, write_config, mock_run_agent_loop
):
    """Case 5: a falsy (None/"") system_prompt -> messages is []."""
    config = base_config()
    config["system_prompt"] = system_prompt
    write_config(config)
    monkeypatch.setattr(sys, "argv", ["main.py"])
    mock_run_agent_loop.side_effect = StopLoop()

    with pytest.raises(StopLoop):
        await main.main()

    actual_messages, _ = mock_run_agent_loop.call_args.args
    assert actual_messages == []


async def test_while_true_loop_calls_run_agent_loop_repeatedly(
    monkeypatch, write_config, mock_run_agent_loop
):
    """Case 6: run_agent_loop.side_effect = [None, None, StopLoop()] ->
    pytest.raises(StopLoop) around `await main.main()`; awaited exactly 3
    times, each call's config argument identical.
    """
    config = base_config()
    write_config(config)
    monkeypatch.setattr(sys, "argv", ["main.py"])
    mock_run_agent_loop.side_effect = [None, None, StopLoop()]

    with pytest.raises(StopLoop):
        await main.main()

    assert mock_run_agent_loop.await_count == 3
    configs = [call.args[1] for call in mock_run_agent_loop.await_args_list]
    assert configs[0] == configs[1] == configs[2]


async def test_load_dotenv_called_once(
    monkeypatch, write_config, mock_run_agent_loop, stub_load_dotenv
):
    """Case 7: main.load_dotenv (patched) is called exactly once."""
    config = base_config()
    write_config(config)
    monkeypatch.setattr(sys, "argv", ["main.py"])
    mock_run_agent_loop.side_effect = StopLoop()

    with pytest.raises(StopLoop):
        await main.main()

    stub_load_dotenv.assert_called_once()


async def test_missing_config_file_raises_file_not_found(
    monkeypatch, tmp_path, mock_run_agent_loop
):
    """Case 8: cwd has no config.yaml at all -> pytest.raises(FileNotFoundError).

    Note: unlike other cases, do NOT use write_config here — chdir into an
    empty tmp_path directly instead.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["main.py"])

    with pytest.raises(FileNotFoundError):
        await main.main()
