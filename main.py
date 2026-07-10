import json
import tomllib
from openai import OpenAI, APIError
from dotenv import load_dotenv
import re
import os
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

THOUGHT_RE = re.compile(r"<(thought|think|thinking)>.*?(?:</\1>|$)", re.DOTALL)


def clean_content(message) -> str:
    content = THOUGHT_RE.sub("", message.content or "").strip()
    return content


def load_config() -> dict:
    with open("config.toml", mode="rb") as f:
        config = tomllib.load(f)
    return config


def load_system_prompt() -> str:
    with open("prompt.md", mode="r", encoding="utf8") as f:
        system_prompt = f.read()
    return system_prompt


def log(messages, message):
    messages.append(message)
    with open("log.log", mode="a", encoding="utf8") as f:
        f.write(json.dumps(message, indent=2) + "\n")


def sanitize_schema(schema: dict) -> dict:
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    schema["additionalProperties"] = False
    return schema


def build_tool_schemas(tools) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": sanitize_schema(tool.inputSchema),
            },
        }
        for tool in tools.tools
    ]

async def call_tool_safely(session, tool):
    string_kwargs = tool.function.arguments
    name = tool.function.name
    try:
        output = await session.call_tool(
            name, json.loads(string_kwargs or "{}")
        )
        content = output.content[0].text
        if output.isError:
            raise Exception(content)
        print(
            f"Tool '{name}' called with arguments {string_kwargs[: min(len(string_kwargs), 100)]}"
        )
    except Exception as e:
        content = json.dumps({"error": str(e)})
    return content

async def run_tool_calls(message, session, max_tool_calls, n_tool_calls_rounds):
    # Loop through the tools called
    limit_hit = False
    tool_call_list = []
    for tool in message.tool_calls:
        n_tool_calls_rounds += 1

        # Number of tool calls exceeded, truncate model work and force response
        if n_tool_calls_rounds > max_tool_calls:
            limit_hit = True
            tool_call_list.append( 
                {
                    "role": "tool",
                    "tool_call_id": tool.id,
                    "content": json.dumps(
                        {
                            "error": "Tool call limit reached. Answer with what you have."
                        }
                    ),
                }
            )
            continue

        # Try calling tool
        content = await call_tool_safely(session, tool)

        tool_call_list.append( 
            {
                "role": "tool",
                "tool_call_id": tool.id,
                "content": content,
            }
        )
    return tool_call_list, limit_hit, n_tool_calls_rounds

async def force_final_response(client, messages, config):
    response = client.chat.completions.create(
        model=config["model"],
        messages=messages,
        max_completion_tokens=config["max_completion_tokens"],
    )

    if response.choices and response.choices[0].message:
        content = clean_content(response.choices[0].message)
        print(f"\n{content}\n")
        log(messages, {"role": "assistant", "content": content})
    
    print()

    return



async def main():

    config = load_config()
    system_prompt = load_system_prompt()

    client = OpenAI(
        base_url=os.getenv("LLM_BASE_URL"), api_key=os.getenv("LLM_API_KEY")
    )

    messages = []

    log(messages, {"role": "system", "content": system_prompt})

    server_params = StdioServerParameters(command="python", args=["server.py"])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            tool_schemas = build_tool_schemas(tools)

            while True:
                # User prompt
                user_input = input("User: ")

                log(messages, {"role": "user", "content": user_input})

                n_tool_calls_rounds = 0
                retry_delay = 1

                # Loop until no more tools are called
                while True:
                    # Error handling on sending request
                    try:
                        response = client.chat.completions.create(
                            model=config["model"],
                            messages=messages,
                            tools=tool_schemas,
                            max_completion_tokens=config["max_completion_tokens"],
                        )
                        retry_delay = 1
                    except APIError as e:
                        print(f"\nServer error, retrying in {retry_delay}s: {e}\n")
                        await asyncio.sleep(retry_delay)
                        retry_delay = min(retry_delay * 2, 60)
                        continue

                    if not response.choices:
                        print("\nThe model did not output any choices\n")
                        break

                    # Truncate the model thinking, let the model know, and retry the prompt
                    if response.choices[0].finish_reason == "length":
                        log(messages, 
                            {
                                "role": "user",
                                "content": "Your previous response was cut off for being too long. Answer again, briefly, without re-deriving your reasoning.",
                            }
                        )
                        continue

                    message = response.choices[0].message

                    # No tools called, the agent has completed all its tasks
                    if not message.tool_calls:
                        content = clean_content(message)
                        log(messages, {"role": "assistant", "content": content})
                        print(f"\n{content}\n")
                        break

                    log(messages, 
                        {
                            "role": "assistant",
                            "content": clean_content(message),
                            "tool_calls": [
                                tool.model_dump() for tool in message.tool_calls
                            ],
                        }
                    )

                    print()

                    # RUN TOOL CALLS
                    tool_call_list, limit_hit, n_tool_calls_rounds = await run_tool_calls(message, session, config["max_tool_calls"], n_tool_calls_rounds)
                    messages.extend(tool_call_list)                 
                    if limit_hit:
                        await force_final_response(client, messages, config)   



if __name__ == "__main__":
    asyncio.run(main())
