"""Run the interactive tool-using personal-agent command-line application."""

import asyncio
import json
import logging
import sys
from typing import Any

import litellm
import yaml
from dotenv import load_dotenv
from litellm import ModelResponse, acompletion
from litellm.types.utils import ChatCompletionMessageToolCall, Message

from compaction import log_usage, preflight_messages
from logging_config import configure_logging, exception_metadata
from tools import TOOL_SCHEMAS, TOOLS

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"

RESET = "\033[0m"

litellm.suppress_debug_info = True

CONFIRM_TOOLS = {"delete_file", "remove_directory"}
RETRY_DELAYS = (1, 2, 4, 8, 16, 32)
RETRYABLE_API_STATUS_CODES = {408, 409, 429}

logger = logging.getLogger("personal_agent.main")


async def run_tool_call(tool: ChatCompletionMessageToolCall) -> dict[str, str]:
    """Execute one model-requested tool call and return its chat message result.

    Tool failures, invalid arguments, and unknown tool names are converted into
    results the model can use to recover on its next turn.
    """
    name = tool.function.name
    string_arguments = tool.function.arguments

    if name not in TOOLS:
        tool_content = f"Tool {name} does not exist. The list of available tools are {list(TOOLS.keys())}"
        logger.warning("tool.unknown")
    else:
        try:
            arguments = json.loads(string_arguments)
            tool_content = str(
                await asyncio.to_thread(TOOLS[name]["function"], **arguments)
            )
            logger.info("tool.completed", extra={"tool_name": name})

        except Exception as e:  # noqa: BLE001 - tool dispatcher must survive arbitrary tool failures
            tool_content = f"An exception occured: {e}. Try differently."
            logger.error(
                "tool.failed",
                extra={"tool_name": name, **exception_metadata(e)},
            )

    return {"tool_call_id": tool.id, "role": "tool", "content": tool_content}


async def run_tool_calls(message: Message) -> list[dict[str, str]]:
    """Confirm destructive calls, then execute the approved calls concurrently.

    A declined call receives a cancellation result so the model knows that the
    requested action was not performed.
    """
    tool_results = []
    approved_calls = []
    for tool in message.tool_calls or []:
        if tool.function.name in CONFIRM_TOOLS:
            answer = input(
                f"{RED}VALIDATION{RESET}: Run {tool.function.name} with arguments "
                f"{tool.function.arguments}? Type 'yes' to confirm: "
            )
            if answer.strip().lower() != "yes":
                logger.info("tool.cancelled", extra={"tool_name": tool.function.name})
                tool_results.append(
                    {
                        "tool_call_id": tool.id,
                        "role": "tool",
                        "content": "Action cancelled: user did not confirm",
                    }
                )
                continue
        approved_calls.append(tool)

    approved_tool_results = await asyncio.gather(
        *(run_tool_call(tool) for tool in approved_calls)
    )
    tool_results.extend(approved_tool_results)
    return tool_results


async def _stream_completion(
    completion_kwargs: dict[str, Any], model_shown: bool
) -> tuple[list[Any], bool]:
    """Submit and print one streamed completion while retaining every chunk."""
    response = await acompletion(**completion_kwargs)
    print(f"{GREEN}MODEL{RESET}:")
    chunks = []
    async for chunk in response:
        if not model_shown:
            model_shown = True
            print(f"{YELLOW}MODEL: {response.model}{RESET}")

        chunks.append(chunk)
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if not delta or not delta.content:
            continue
        print(delta.content, end="", flush=True)
    print()
    return chunks, model_shown


async def run_agent_loop(
    messages: list[dict[str, Any]], config: dict[str, Any]
) -> None:
    """Handle one user prompt and its model/tool exchange.

    Streams the model response, retries transient API failures and responses
    cut off by the token limit, and continues executing tool calls until the
    model produces a final reply or the configured iteration limit is reached.
    """
    user_prompt = input(f"{GREEN}USER{RESET}: ")
    messages.append({"role": "user", "content": user_prompt})
    model_shown = False
    completion_candidates = [config["model"], *(config.get("fallbacks") or [])]

    for _ in range(config["max_iterations"]):
        completed_chunks = None
        last_error = None

        for attempt in range(len(RETRY_DELAYS) + 1):
            for candidate_index, candidate in enumerate(completion_candidates):
                retry_error = None
                context_recovery_used = False
                force_compaction = False
                while True:
                    try:
                        request_messages = await preflight_messages(
                            messages,
                            candidate,
                            config,
                            force_compaction=force_compaction,
                        )
                        force_compaction = False
                        completion_kwargs = {
                            "model": candidate,
                            "messages": request_messages,
                            "max_completion_tokens": config["max_completion_tokens"],
                            "tools": TOOL_SCHEMAS,
                            "api_base": config.get("api_base"),
                            "api_key": config.get("api_key"),
                            "stream": True,
                            "stream_options": {"include_usage": True},
                            "max_retries": 0,
                            "num_retries": 0,
                        }
                        chunks, model_shown = await _stream_completion(
                            completion_kwargs, model_shown
                        )
                    except litellm.ContextWindowExceededError:
                        if context_recovery_used:
                            raise
                        context_recovery_used = True
                        force_compaction = True
                        logger.warning(
                            "context.provider_overflow", extra={"model": candidate}
                        )
                        continue
                    # Separate clauses keep every retryable LiteLLM failure explicit.
                    except litellm.RateLimitError as error:
                        retry_error = error
                    except litellm.Timeout as error:
                        retry_error = error
                    except litellm.APIConnectionError as error:
                        retry_error = error
                    except litellm.InternalServerError as error:
                        retry_error = error
                    except litellm.BadGatewayError as error:
                        retry_error = error
                    except litellm.ServiceUnavailableError as error:
                        retry_error = error
                    except litellm.APIError as error:
                        status_code = getattr(error, "status_code", None)
                        if not (
                            isinstance(status_code, int)
                            and (
                                status_code in RETRYABLE_API_STATUS_CODES
                                or status_code >= 500
                            )
                        ):
                            logger.error(
                                "llm.request.failed",
                                extra={
                                    "model": candidate,
                                    "status_code": status_code,
                                    **exception_metadata(error),
                                },
                            )
                            raise
                        retry_error = error
                    else:
                        completed_chunks = chunks
                    break

                if completed_chunks is not None:
                    break

                if retry_error is None:
                    raise RuntimeError("Completion failed without an API exception")

                last_error = retry_error
                logger.warning(
                    "llm.request.failed",
                    extra={
                        "model": candidate,
                        "attempt": attempt + 1,
                        "max_attempts": len(RETRY_DELAYS) + 1,
                        "status_code": getattr(retry_error, "status_code", None),
                        **exception_metadata(retry_error),
                    },
                )

                if candidate_index < len(completion_candidates) - 1:
                    logger.warning(
                        "llm.fallback.selected",
                        extra={
                            "model": candidate,
                            "next_model": completion_candidates[candidate_index + 1],
                        },
                    )

            if completed_chunks is not None:
                break

            if attempt == len(RETRY_DELAYS):
                logger.error(
                    "llm.retries_exhausted",
                    extra={"max_attempts": len(RETRY_DELAYS) + 1},
                )
                if last_error is None:
                    raise RuntimeError("Completion failed without an API exception")
                raise last_error

            delay = RETRY_DELAYS[attempt]
            logger.warning(
                "llm.retry.scheduled",
                extra={"attempt": attempt + 2, "delay_seconds": delay},
            )
            await asyncio.sleep(delay)

        if completed_chunks is None:
            raise RuntimeError("Completion retry loop ended without a response")

        chunks = completed_chunks

        rebuilt_response = litellm.stream_chunk_builder(chunks, messages=messages)

        if not isinstance(rebuilt_response, ModelResponse):
            logger.error("llm.response.invalid")
            return

        if not rebuilt_response.choices:
            logger.error("llm.response.empty")
            return

        log_usage(rebuilt_response, candidate, "agent")

        if rebuilt_response.choices[0].finish_reason == "length":
            messages.append(
                {
                    "role": "user",
                    "content": "Your previous response was cut off for being too long. Answer again, briefly, without re-deriving your reasoning.",
                }
            )
            continue

        message = rebuilt_response.choices[0].message
        messages.append(message.model_dump())

        if not message.tool_calls:
            break

        tool_results = await run_tool_calls(message)
        messages.extend(tool_results)

    else:
        logger.warning(
            "agent.max_iterations",
            extra={"max_iterations": config["max_iterations"]},
        )


async def main() -> None:
    """Load configuration and run consecutive interactive agent turns."""
    load_dotenv()
    configure_logging()

    with open("config.yaml", encoding="utf8") as f:  # noqa: ASYNC230 - one-time startup read, before event loop has any contention
        config = yaml.safe_load(f)

    profile_name = sys.argv[1] if len(sys.argv) > 1 else config["active_profile"]
    profile = config["profiles"][profile_name]
    config = {**config, **profile}
    logger.info("app.started", extra={"profile": profile_name})

    messages = []

    system_prompt = config["system_prompt"]
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    while True:
        await run_agent_loop(messages, config)


def cli() -> None:
    """Run the agent through the installed ``personal-agent`` command."""
    try:
        asyncio.run(main())
    except (EOFError, KeyboardInterrupt) as error:
        logger.info("app.stopped", extra={"reason": type(error).__name__})
    except Exception as error:  # noqa: BLE001 - process boundary logs all fatal failures
        logger.critical("app.failed", extra=exception_metadata(error))
        raise SystemExit(1) from None


if __name__ == "__main__":
    cli()
