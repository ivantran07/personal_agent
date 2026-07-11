from litellm import acompletion
import json
import yaml
from dotenv import load_dotenv
import asyncio
from tools.files import TOOL_SCHEMAS, TOOLS

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"

RESET = "\033[0m"


async def main():
    load_dotenv()
    # litellm._turn_on_debug()

    with open("config.yaml", encoding="utf8") as f:
        config = yaml.safe_load(f)

    messages = []

    system_prompt = config["system_prompt"]
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    user_prompt = input(f"{GREEN}USER{RESET}: ")
    messages.append({"role": "user", "content": user_prompt})

    for _ in range(config["max_iterations"]):
        response = await acompletion(
            model=config["model"],
            messages=messages,
            max_completion_tokens=config["max_completion_tokens"],
            tools=TOOL_SCHEMAS,
            fallbacks=config.get("fallbacks", []),
        )

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

        for tool in message.tool_calls:
            name = tool.function.name
            string_arguments = tool.function.arguments

            if name not in TOOLS:
                tool_content = f"Tool {name} does not exist. The list of available tools are {list(TOOLS.keys())}"
                print(f"{RED}TOOL{RESET}: {name} does not exist")
            else:
                try:
                    arguments = json.loads(string_arguments)
                    tool_content = str(TOOLS[name]["function"](**arguments))
                    print(
                        f"{GREEN}TOOL{RESET}: {name} returned with arguments {string_arguments}"
                    )

                except Exception as e:
                    tool_content = f"An exception occured: {e}. Try differently."
                    print(
                        f"{RED}TOOL{RESET}: {name} failed with arguments {string_arguments}"
                    )

            messages.append(
                {"tool_call_id": tool.id, "role": "tool", "content": tool_content}
            )

    else:
        print(f"{YELLOW}Max iterations reached without a final answer{RESET}")


if __name__ == "__main__":
    asyncio.run(main())
