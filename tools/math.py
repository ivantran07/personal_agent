from tools.base import ToolEntry


def add(a: int, b: int) -> int:
    return a + b


def substract(a: int, b: int) -> int:
    return a - b


def multiply(a: int, b: int) -> int:
    return a * b


def divide(a: int, b: int) -> float:
    return a / b


TOOLS: dict[str, ToolEntry] = {
    "add": {
        "function": add,
        "schema": {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two integers a and b and returns the result",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                    "additionalProperties": False,
                },
            },
        },
    },
    "substract": {
        "function": substract,
        "schema": {
            "type": "function",
            "function": {
                "name": "substract",
                "description": "Substract integers b from a and returns the result",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                    "additionalProperties": False,
                },
            },
        },
    },
    "multiply": {
        "function": multiply,
        "schema": {
            "type": "function",
            "function": {
                "name": "multiply",
                "description": "Multiply two integers a and b and returns the result",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                    "additionalProperties": False,
                },
            },
        },
    },
    "divide": {
        "function": divide,
        "schema": {
            "type": "function",
            "function": {
                "name": "divide",
                "description": "Divides integer a by integer b and returns the result",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                    "additionalProperties": False,
                },
            },
        },
    },
}

TOOL_SCHEMAS = [t["schema"] for t in TOOLS.values()]
