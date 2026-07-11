from tools.base import ToolEntry
from pathlib import Path
import os

FILES_ROOT = Path(os.environ.get("FILES_ROOT", "./files")).resolve()


def read_file(path: str) -> str:
    resolved = (FILES_ROOT / path).resolve()
    assert resolved.is_relative_to(FILES_ROOT)
    with open(resolved, "r", encoding="utf8") as f:
        return f.read()


TOOLS: dict[str, ToolEntry] = {
    "read_file": {
        "function": read_file,
        "schema": {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the file at the given path",
                "parameters": {
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path at which the file will be read",
                        }
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        },
    }
}

TOOL_SCHEMAS = [t["schema"] for t in TOOLS.values()]