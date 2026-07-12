import litellm
from litellm import acompletion
import json
import sys
import yaml
from dotenv import load_dotenv
import asyncio
from tools import TOOL_SCHEMAS, TOOLS

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"

RESET = "\033[0m"

litellm.suppress_debug_info = True

CONFIRM_TOOLS = {"delete_file", "remove_directory"}


async def run_tool_call(tool):
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

        except Exception as e:
            tool_content = f"An exception occured: {e}. Try differently."
            print(f"{RED}TOOL{RESET}: {name} failed with arguments {string_arguments}")

    return {"tool_call_id": tool.id, "role": "tool", "content": tool_content}


async def run_tool_calls(message):
    tool_results = []
    approved_calls = []
    for tool in message.tool_calls:
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


async def run_agent_loop(messages, config):
    user_prompt = input(f"{GREEN}USER{RESET}: ")
    messages.append({"role": "user", "content": user_prompt})
    model_shown = False

    for _ in range(config["max_iterations"]):
        response = await acompletion(
            model=config["model"],
            messages=messages,
            max_completion_tokens=config["max_completion_tokens"],
            tools=TOOL_SCHEMAS,
            fallbacks=config.get("fallbacks", []),
            api_base=config.get("api_base"),
            api_key=config.get("api_key"),
        )

        if not model_shown:
            model_shown = True
            print(f"{YELLOW}MODEL: {response.model}{RESET}")

        if response.choices[0].finish_reason == "length":
            messages.append(
                {
                    "role": "user",
                    "content": "Your previous response was cut off for being too long. Answer again, briefly, without re-deriving your reasoning.",
                }
            )
            continue

        if not response.choices:
            print("No choices")
            return

        message = response.choices[0].message
        messages.append(message.model_dump())

        if message.content:
            print(f"{GREEN}MODEL{RESET}: {message.content}")

        if not message.tool_calls:
            break

        tool_results = await run_tool_calls(message)
        messages.extend(tool_results)

    else:
        print(f"{YELLOW}Max iterations reached without a final answer{RESET}")


async def main():
    load_dotenv()

    with open("config.yaml", encoding="utf8") as f:
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


if __name__ == "__main__":
    asyncio.run(main())
