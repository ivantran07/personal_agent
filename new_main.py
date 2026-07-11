from openai import OpenAI
import yaml
from dotenv import load_dotenv
from tools.files import TOOLS, TOOL_SCHEMAS
import os
import json

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"

RESET = "\033[0m"


def main():
    load_dotenv()

    client = OpenAI(
        api_key=os.getenv("GEMINI_API_KEY", ""),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    with open("config.yaml", encoding="utf8") as f:
        config = yaml.safe_load(f)

    messages = []

    system_prompt = config["system_prompt"]
    messages.append({"role": "system", "content": system_prompt})

    user_prompt = input(f"{GREEN}USER{RESET}: ")
    messages.append({"role": "user", "content": user_prompt})

    for iteration in range(config["max_iterations"]):
        response = client.chat.completions.create(
            model=config["model"],
            messages=messages,
            max_completion_tokens=config["max_completion_tokens"],
            tools=TOOL_SCHEMAS,
        )

        if not response.choices:
            print("No choices")
            exit()

        message = response.choices[0].message
        messages.append(message)

        if not message:
            print("No message")

        if not message.tool_calls:
            print(f"{GREEN}MODEL{RESET}: {message.content}")
            break

        for tool in message.tool_calls:
            name = tool.function.name
            string_arguments = tool.function.arguments

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

    # TODO: finish writing new version using litellm, without tools / mcps for now.


if __name__ == "__main__":
    main()
