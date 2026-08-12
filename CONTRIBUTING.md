# Contributing

Contributions are welcome. This is primarily an educational project for
understanding how a tool-calling agent works, so please keep changes focused,
readable, and consistent with its intentionally small, hand-rolled design.

## Getting started

This project requires Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/ivantran07/personal_agent.git
cd personal_agent
uv sync --group dev
uv run pre-commit install
```

Copy `.env.example` to `.env`, then add an API key only if you need to run the
agent against a hosted model. Never commit `.env` or any other credentials.

On macOS or Linux:

```bash
cp .env.example .env
```

On PowerShell:

```powershell
Copy-Item .env.example .env
```

See the [README](README.md) for model configuration and how to run the agent.

## Making a change

- Keep pull requests small and focused on one clear improvement.
- Preserve the project's straightforward architecture; avoid adding frameworks
  or abstractions unless they directly support its learning goals.
- Add or update focused tests when behavior changes.
- Keep type hints and docstrings up to date, especially around agent control
  flow, tool safety, and configuration.
- Do not commit secrets, local databases, generated files, or runtime data.

## Run the checks

Run the same checks used by CI before opening a pull request:

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check .
uv run pytest
```

The pre-commit hooks run Ruff and `ty` automatically. If formatting needs
updating, run:

```bash
uv run ruff format .
```

## Pull requests

In the pull request description, briefly explain what changed and why. Include
the checks you ran, and call out any changes to configuration, tool behavior,
or safety assumptions.

Be mindful that the file tools are path-restricted within the application, not
isolated by the operating system. Changes that affect filesystem access or tool
execution should preserve or improve those safety boundaries.

## License

By submitting a contribution, you agree that it may be included in this
project under its [MIT License](LICENSE).
