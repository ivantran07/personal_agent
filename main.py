"""Run the interactive tool-using personal-agent command-line application."""

import asyncio
import json
import sys
from typing import Any

import litellm
import yaml
from dotenv import load_dotenv
from litellm import ModelResponse, acompletion
from litellm.types.utils import ChatCompletionMessageToolCall, Message

from tools import TOOL_SCHEMAS, TOOLS

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"

RESET = "\033[0m"

litellm.suppress_debug_info = True

CONFIRM_TOOLS = {"delete_file", "remove_directory"}
RETRY_DELAYS = (1, 2, 4, 8, 16, 32)
RETRYABLE_API_STATUS_CODES = {408, 409, 429}


async def run_tool_call(tool: ChatCompletionMessageToolCall) -> dict[str, str]:
    """Execute one model-requested tool call and return its chat message result.

    Tool failures, invalid arguments, and unknown tool names are converted into
    results the model can use to recover on its next turn.
    """
    name = tool.function.name
    string_arguments = tool.function.arguments

    if name not in TOOLS:
        tool_content = f"Tool {name} does not exist. The list of available tools are {list(TOOLS.keys())}"
        print(f"{RED}TOOL{RESET}: {name} does not exist")
    else:
        try:
            arguments = json.loads(string_arguments)
            tool_content = str(
                await asyncio.to_thread(TOOLS[name]["function"], **arguments)
            )
            print(
                f"{GREEN}TOOL{RESET}: {name} returned with arguments {string_arguments}"
            )

        except Exception as e:  # noqa: BLE001 - tool dispatcher must survive arbitrary tool failures
            tool_content = f"An exception occured: {e}. Try differently."
            print(f"{RED}TOOL{RESET}: {name} failed with arguments {string_arguments}")

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
                print(f"{RED}TOOL{RESET}: {tool.function.name} cancelled by user")
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
                completion_kwargs = {
                    "model": candidate,
                    "messages": messages,
                    "max_completion_tokens": config["max_completion_tokens"],
                    "tools": TOOL_SCHEMAS,
                    "api_base": config.get("api_base"),
                    "api_key": config.get("api_key"),
                    "stream": True,
                    "max_retries": 0,
                    "num_retries": 0,
                }

                retry_error = None
                try:
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
                        print(
                            f"\n{RED}NON-RETRYABLE API FAILURE{RESET}: "
                            f"{type(error).__name__} for model "
                            f"{completion_kwargs['model']}: {error}"
                        )
                        raise
                    retry_error = error
                else:
                    completed_chunks = chunks
                    break

                last_error = retry_error
                print(
                    f"\n{RED}API FAILURE{RESET}: {type(retry_error).__name__} "
                    f"for model {completion_kwargs['model']} on attempt "
                    f"{attempt + 1}/{len(RETRY_DELAYS) + 1}: {retry_error}"
                )

                if candidate_index < len(completion_candidates) - 1:
                    print(f"{YELLOW}FALLBACK{RESET}: Trying the next configured model")

            if completed_chunks is not None:
                break

            if attempt == len(RETRY_DELAYS):
                print(
                    f"{RED}API FAILURE{RESET}: All "
                    f"{len(RETRY_DELAYS) + 1} attempts exhausted; aborting"
                )
                if last_error is None:
                    raise RuntimeError("Completion failed without an API exception")
                raise last_error

            delay = RETRY_DELAYS[attempt]
            print(
                f"{YELLOW}RETRY{RESET}: All configured models failed; "
                f"retrying in {delay} seconds"
            )
            await asyncio.sleep(delay)

        if completed_chunks is None:
            raise RuntimeError("Completion retry loop ended without a response")

        chunks = completed_chunks

        rebuilt_response = litellm.stream_chunk_builder(chunks, messages=messages)

        if not isinstance(rebuilt_response, ModelResponse):
            print("No chat-completion response")
            return

        if not rebuilt_response.choices:
            print("No choices")
            return

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
        print(f"{YELLOW}Max iterations reached without a final answer{RESET}")


async def main() -> None:
    """Load configuration and run consecutive interactive agent turns."""
    load_dotenv()

    with open("config.yaml", encoding="utf8") as f:  # noqa: ASYNC230 - one-time startup read, before event loop has any contention
        config = yaml.safe_load(f)

    profile_name = sys.argv[1] if len(sys.argv) > 1 else config["active_profile"]
    profile = config["profiles"][profile_name]
    config = {**config, **profile}

    messages = []

    system_prompt = config["system_prompt"]
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    while True:
        await run_agent_loop(messages, config)


def cli() -> None:
    """Run the agent through the installed ``personal-agent`` command."""
    asyncio.run(main())


if __name__ == "__main__":
    cli()
