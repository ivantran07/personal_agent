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
- **Small, composable tool modules** — file, web, math, and RAG tools each live in their own module under `tools/` and get merged into one registry automatically.
- **Context-aware conversations** — every model request is token-counted with its tool schemas and output reserve; old complete turns are summarized before they can overflow the selected model's context window.

## Installation

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/ivantran07/personal_agent.git
cd personal_agent
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
| `LOG_LEVEL` | Minimum structured log level | `INFO` |

### Model profiles (`config.yaml`)

`config.yaml` defines named profiles, each pointing at a different model/provider through litellm:

- `openrouter` — free-tier OpenRouter models, with several free fallbacks configured.
- `gemini` — Google's Gemini models.
- `llama` — a local OpenAI-compatible server (e.g. `llama.cpp`), no API key required.

Set `active_profile` in `config.yaml` to pick the default, or pass a profile name as a CLI argument (see below).

Context management uses these top-level settings, which profiles may override:

| Setting | Purpose | Default |
|---|---|---|
| `context_trigger_ratio` | Compact when input, output reserve, and safety margin reach this fraction of the context window | `0.85` |
| `context_target_ratio` | Target utilization after compaction | `0.60` |
| `context_safety_tokens` | Extra room for tokenizer/provider accounting differences | `512` |
| `compaction_max_completion_tokens` | Maximum output for the separate summary request | `1024` |
| `context_window_overrides` | Explicit model-to-context-window mapping when discovery is unavailable | `{}` |

Context limits are resolved from an explicit override first, then provider-native
metadata (OpenRouter's model catalog or llama.cpp's `/props` endpoint), and then
LiteLLM's model metadata. Generic or newly released models that cannot be
resolved require an override. Set `metadata_provider: llamacpp` on a profile to
enable `/props` discovery for an OpenAI-compatible llama.cpp server.

## Usage

```bash
uv run personal-agent            # uses active_profile from config.yaml
uv run personal-agent gemini     # overrides the profile for this run
```

`uv run python main.py` remains available when working from the repository.

Example session:

```
USER: what's in the current directory, and what's 47 * 89?
MODEL: gpt-oss-120b
{"timestamp":"...","level":"INFO","logger":"personal_agent.main","event":"tool.completed","tool_name":"list_files"}
{"timestamp":"...","level":"INFO","logger":"personal_agent.main","event":"tool.completed","tool_name":"multiply"}
MODEL: The current directory contains notes.txt and draft.md. 47 * 89 = 4183.
```

Operational events are emitted as one JSON object per line on stderr, while
prompts and streamed model output remain on stdout. Logs contain operation
metadata, registered tool names, configured model/profile identifiers, and
provider-reported token usage, and sanitized traceback locations. They do not contain prompts, unknown
model-generated tool names, tool arguments, results, user-provided paths,
URLs, or exception messages. Set
`LOG_LEVEL=DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL` to control
verbosity.

Destructive actions pause for confirmation before running:

```
MODEL: I'll delete the temp file now.
VALIDATION: Run delete_file with arguments {"path": "temp.txt"}? Type 'yes' to confirm: yes
{"timestamp":"...","level":"INFO","logger":"personal_agent.main","event":"tool.completed","tool_name":"delete_file"}
```

Declined destructive actions emit a `tool.cancelled` event without their
arguments.

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

### RAG (`tools/rag.py`)

`rag_ingest`, `rag_search` — store text in a local pgvector database and retrieve the most semantically similar stored documents.

## Architecture

Each turn of `run_agent_loop` in `main.py`, using context preflight from
`compaction.py`:

1. Discovers the candidate model's context limit and counts the complete conversation plus all tool schemas (`TOOL_SCHEMAS`, merged from every module in `tools/`). Completion tokens and a safety margin are reserved before submission.
2. If the request crosses the configured threshold, old complete turns are replaced atomically by a structured summary while the original system message and active turn remain unchanged. Compaction is a separate, non-streaming model request and therefore has its own token usage and provider cost.
3. Sends an immutable snapshot of the resulting conversation and tool schemas to the model via LiteLLM. Primary and fallback models are preflighted independently because their limits and tokenizers may differ.
4. If the model requests tool calls, `run_tool_calls` first walks them **sequentially** in the main thread, resolving any that require confirmation (`CONFIRM_TOOLS`). Declined calls short-circuit to a "cancelled" result.
5. The remaining approved calls run **concurrently**, each dispatched via `asyncio.to_thread` so ordinary blocking tool code doesn't need to be rewritten as async, and gathered with `asyncio.gather`.
6. Tool results are appended to the conversation and the loop continues until the model returns a final answer or `max_iterations` is hit. Provider usage is logged for observability, while the next request is always counted afresh rather than using a cumulative token total.

Note on the file tools: `FILES_ROOT` path-jailing (`tools/files.py`) prevents path traversal *within the running process*, but this is an application-level check, not OS-level isolation — see [Roadmap](#roadmap--known-limitations).

## Roadmap / Known Limitations

- **No real sandboxing yet.** The file tools are path-jailed at the application level, but the process itself runs unrestricted on the host. A Docker-based setup (for real filesystem/process isolation) is planned but not yet implemented.
- **`math.py` is minimal.** Just four basic operations; a safe expression evaluator (rather than one tool per operation) is a likely next step.
- Possible future additions: document/slide creation tools (e.g. PPT), a dedicated Wikipedia summary tool.

## License

[MIT](LICENSE)
