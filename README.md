# Personal Agent

![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![uv](https://img.shields.io/badge/managed%20with-uv-orange.svg)

A from-scratch agent loop built to learn how tool-calling agents actually work under the hood — model-agnostic via [litellm](https://github.com/BerriAI/litellm), with async tool dispatch, a small set of file/web/math tools, and a confirmation flow for destructive actions.

This is a personal learning project, not a production framework. It's intentionally minimal and hand-rolled rather than built on something like LangChain — the goal is to fully understand every moving part, not to cover every use case. It's still under active development; expect rough edges.

## Table of Contents

- [Motivation](#motivation)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Available Tools](#available-tools)
- [Architecture](#architecture)
- [Roadmap / Known Limitations](#roadmap--known-limitations)
- [License](#license)

## Motivation

I wanted to learn how agent loops and tool-calling work under the hood — the request/response cycle, how tool schemas get passed to a model, how tool results feed back in, and how to layer things like async execution and human-in-the-loop confirmation on top without a framework doing it for me. LangChain and similar frameworks are powerful but heavy and opinionated; this project trades coverage for the ability to understand (and change) every line.

## Features

- **Model-agnostic** — powered by [litellm](https://github.com/BerriAI/litellm), so the same agent loop runs against OpenRouter's free-tier models, Gemini, or a local `llama.cpp` server, just by switching a profile in `config.yaml`.
- **Async, concurrent tool dispatch** — when the model requests multiple tools in one turn, they run concurrently via `asyncio.gather` + `asyncio.to_thread`, instead of one at a time.
- **Confirm-before-destructive** — tools that mutate data irreversibly (`delete_file`, `remove_directory`) are resolved synchronously with an interactive yes/no prompt *before* any concurrent tool dispatch begins, so confirmation prompts can never interleave with other tool output.
- **Small, composable tool modules** — file, web, and math tools each live in their own module under `tools/` and get merged into one registry automatically.

## Installation

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/your-username/personal-agent.git
cd personal-agent
uv sync
cp .env.example .env
```

Fill in `.env` with whichever API key(s) your chosen profile needs (see [Configuration](#configuration)).

## Configuration

### Environment variables (`.env`)

| Variable | Purpose | Default |
|---|---|---|
| `OPENROUTER_API_KEY` | API key for the `openrouter` profile | — |
| `GEMINI_API_KEY` | API key for the `gemini` profile | — |
| `FILES_ROOT` | Root directory the file tools are restricted to | `./files` |
| `MAX_READ_BYTES` | Max size for `read_file`/`grep` | `1000000` |
| `MAX_FETCH_BYTES` | Max response size for `fetch_url` | `2000000` |
| `REQUEST_TIMEOUT` | Timeout (seconds) for `fetch_url` | `15` |

### Model profiles (`config.yaml`)

`config.yaml` defines named profiles, each pointing at a different model/provider through litellm:

- `openrouter` — free-tier OpenRouter models, with several free fallbacks configured.
- `gemini` — Google's Gemini models.
- `llama` — a local OpenAI-compatible server (e.g. `llama.cpp`), no API key required.

Set `active_profile` in `config.yaml` to pick the default, or pass a profile name as a CLI argument (see below).

## Usage

```bash
uv run python main.py            # uses active_profile from config.yaml
uv run python main.py gemini     # overrides the profile for this run
```

Example session:

```
USER: what's in the current directory, and what's 47 * 89?
MODEL: gpt-oss-120b
TOOL: list_files returned with arguments {}
TOOL: multiply returned with arguments {"a": 47, "b": 89}
MODEL: The current directory contains notes.txt and draft.md. 47 * 89 = 4183.
```

Destructive actions pause for confirmation before running:

```
MODEL: I'll delete the temp file now.
VALIDATION: Run delete_file with arguments {"path": "temp.txt"}? Type 'yes' to confirm: yes
TOOL: delete_file returned with arguments {"path": "temp.txt"}
```

## Available Tools

### Files (`tools/files.py`)

Restricted to `FILES_ROOT` — any path that resolves outside it is rejected.

`read_file`, `write_file`, `replace`, `replace_all`, `append_file`, `delete_file`\*, `list_files`, `glob`, `grep`, `exists_file`, `stat_file`, `make_directory`, `move`, `copy`, `remove_directory`\*, `copy_directory`

\* requires interactive confirmation.

### Web (`tools/web.py`)

`fetch_url` — fetches a page and extracts readable text (via [trafilatura](https://github.com/adbar/trafilatura)), stripping nav/ads/boilerplate.
`web_search` — searches the web via [ddgs](https://github.com/deedy5/ddgs) (DuckDuckGo), no API key required.

### Math (`tools/math.py`)

`add`, `substract`, `multiply`, `divide` — basic arithmetic, offloaded to avoid relying on the model's arithmetic.

## Architecture

Each turn of `run_agent_loop` in `main.py`:

1. Sends the conversation + all tool schemas (`TOOL_SCHEMAS`, merged from every module in `tools/`) to the model via litellm.
2. If the model requests tool calls, `run_tool_calls` first walks them **sequentially** in the main thread, resolving any that require confirmation (`CONFIRM_TOOLS`). Declined calls short-circuit to a "cancelled" result.
3. The remaining approved calls run **concurrently**, each dispatched via `asyncio.to_thread` so ordinary blocking tool code doesn't need to be rewritten as async, and gathered with `asyncio.gather`.
4. Tool results are appended to the conversation and the loop continues until the model returns a final answer or `max_iterations` is hit.

Note on the file tools: `FILES_ROOT` path-jailing (`tools/files.py`) prevents path traversal *within the running process*, but this is an application-level check, not OS-level isolation — see [Roadmap](#roadmap--known-limitations).

## Roadmap / Known Limitations

- **No real sandboxing yet.** The file tools are path-jailed at the application level, but the process itself runs unrestricted on the host. A Docker-based setup (for real filesystem/process isolation) is planned but not yet implemented.
- **No automated tests or CI.** Everything has been verified manually so far.
- **No dev-tooling config.** No linter/formatter (likely `ruff`) or `pytest` wired into `pyproject.toml` yet.
- **`math.py` is minimal.** Just four basic operations; a safe expression evaluator (rather than one tool per operation) is a likely next step.
- Possible future additions: document/slide creation tools (e.g. PPT), a dedicated Wikipedia summary tool.

## License

[MIT](LICENSE)
