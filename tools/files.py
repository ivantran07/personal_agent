from tools.base import ToolEntry
from pathlib import Path
import os
import re
import shutil

FILES_ROOT = Path(os.environ.get("FILES_ROOT", "./files")).resolve()
MAX_READ_BYTES = int(os.environ.get("MAX_READ_BYTES", 1_000_000))


def verify_path(path: str) -> Path:
    resolved = (FILES_ROOT / path).resolve()
    assert resolved.is_relative_to(FILES_ROOT)
    return resolved


def require_existing_parent(resolved: Path) -> None:
    if not resolved.parent.is_dir():
        raise ValueError(f"Parent directory '{resolved.parent}' does not exist")


def check_readable(resolved: Path) -> None:
    size = resolved.stat().st_size
    if size > MAX_READ_BYTES:
        raise ValueError(
            f"File is too large to read ({size} bytes, limit {MAX_READ_BYTES})"
        )
    with open(resolved, "rb") as f:
        chunk = f.read(8192)
    if b"\x00" in chunk:
        raise ValueError("File appears to be binary, refusing to read as text")


def read_file(path: str, start_line: int = 0, end_line: int | None = None) -> str:
    resolved = verify_path(path)
    check_readable(resolved)
    with open(resolved, "r", encoding="utf8") as f:
        lines = f.readlines()
    return "".join(lines[start_line:end_line])


def write_file(path: str, text: str) -> str:
    resolved = verify_path(path)
    require_existing_parent(resolved)
    with open(resolved, "w", encoding="utf8") as f:
        f.write(text)
        return f"Wrote {len(text.encode('utf8'))} bytes to {path}"


def replace(path: str, old_str: str, new_str: str) -> str:
    resolved = verify_path(path)
    content = resolved.read_text(encoding="utf8")
    count = content.count(old_str)
    if count == 0:
        raise ValueError(f"'{old_str}' not found in {path}")
    if count > 1:
        raise ValueError(
            f"'{old_str}' matches {count} times in {path}, must match exactly once"
        )
    new_content = content.replace(old_str, new_str, 1)
    resolved.write_text(new_content, encoding="utf8")

    lines = new_content.splitlines()
    match_line = new_content[: new_content.find(new_str)].count("\n")
    start = max(0, match_line - 2)
    end = min(len(lines), match_line + new_str.count("\n") + 3)
    snippet = "\n".join(lines[start:end])
    return f"Replaced 1 occurrence in {path}:\n{snippet}"


def append_file(path: str, text: str) -> str:
    resolved = verify_path(path)
    with open(resolved, "a", encoding="utf8") as f:
        f.write(text)
        return f"Appended {len(text.encode('utf8'))} bytes to {path}"


def delete_file(path: str) -> str:
    resolved = verify_path(path)
    resolved.unlink()
    return f"Deleted {path}"


def list_files(path: str = ".") -> list[str]:
    resolved = verify_path(path)
    return os.listdir(resolved)


def glob(pattern: str) -> list[str]:
    return [
        str(p)
        for p in FILES_ROOT.glob(pattern)
        if p.resolve().is_relative_to(FILES_ROOT)
    ]


def grep(pattern: str, path: str) -> list[str]:
    resolved = verify_path(path)
    check_readable(resolved)
    with open(resolved, "r", encoding="utf8") as f:
        return [line.rstrip("\n") for line in f if re.search(pattern, line)]


def exists_file(path: str) -> bool:
    resolved = verify_path(path)
    return resolved.exists()


def stat_file(path: str) -> dict:
    resolved = verify_path(path)
    info = resolved.stat()
    return {
        "size": info.st_size,
        "modified": info.st_mtime,
        "is_dir": resolved.is_dir(),
        "is_file": resolved.is_file(),
    }


def make_directory(path: str) -> str:
    resolved = verify_path(path)
    require_existing_parent(resolved)
    resolved.mkdir(exist_ok=True)
    return f"Created directory {path}"


def move(src_path: str, dst_path: str) -> str:
    resolved_src = verify_path(src_path)
    resolved_dst = verify_path(dst_path)
    resolved_src.rename(resolved_dst)
    return f"Moved {src_path} to {dst_path}"


def copy(src_path: str, dst_path: str) -> str:
    resolved_src = verify_path(src_path)
    resolved_dst = verify_path(dst_path)
    require_existing_parent(resolved_dst)
    shutil.copy2(resolved_src, resolved_dst)
    return f"Copied {src_path} to {dst_path}"


TOOLS: dict[str, ToolEntry] = {
    "read_file": {
        "function": read_file,
        "schema": {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the file at the given path, optionally limited to a line range",
                "parameters": {
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path at which the file will be read",
                        },
                        "start_line": {
                            "type": "integer",
                            "description": "First line to include (0-indexed, inclusive)",
                        },
                        "end_line": {
                            "type": "integer",
                            "description": "Line to stop before (0-indexed, exclusive)",
                        },
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        },
    },
    "write_file": {
        "function": write_file,
        "schema": {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write the text at the file at the given path. The parent directory must already exist",
                "parameters": {
                    "properties": {
                        "path": {"type": "string"},
                        "text": {"type": "string"},
                    },
                    "required": ["path", "text"],
                    "additionalProperties": False,
                },
            },
        },
    },
    "replace": {
        "function": replace,
        "schema": {
            "type": "function",
            "function": {
                "name": "replace",
                "description": "Replace old_str with new_str in the file at the given path. old_str must match exactly once in the file, otherwise the call fails",
                "parameters": {
                    "properties": {
                        "path": {"type": "string"},
                        "old_str": {"type": "string"},
                        "new_str": {"type": "string"},
                    },
                    "required": ["path", "old_str", "new_str"],
                    "additionalProperties": False,
                },
            },
        },
    },
    "append_file": {
        "function": append_file,
        "schema": {
            "type": "function",
            "function": {
                "name": "append_file",
                "description": "Append text to the end of the file at the given path",
                "parameters": {
                    "properties": {
                        "path": {"type": "string"},
                        "text": {"type": "string"},
                    },
                    "required": ["path", "text"],
                    "additionalProperties": False,
                },
            },
        },
    },
    "delete_file": {
        "function": delete_file,
        "schema": {
            "type": "function",
            "function": {
                "name": "delete_file",
                "description": "Delete the file at the given path",
                "parameters": {
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        },
    },
    "list_files": {
        "function": list_files,
        "schema": {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List the files available at the given directory path",
                "parameters": {
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path to list, defaults to the root",
                        }
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            },
        },
    },
    "glob": {
        "function": glob,
        "schema": {
            "type": "function",
            "function": {
                "name": "glob",
                "description": "Find files matching a glob pattern",
                "parameters": {
                    "properties": {"pattern": {"type": "string"}},
                    "required": ["pattern"],
                    "additionalProperties": False,
                },
            },
        },
    },
    "grep": {
        "function": grep,
        "schema": {
            "type": "function",
            "function": {
                "name": "grep",
                "description": "Search the file at the given path for lines matching a regex pattern",
                "parameters": {
                    "properties": {
                        "pattern": {"type": "string"},
                        "path": {"type": "string"},
                    },
                    "required": ["pattern", "path"],
                    "additionalProperties": False,
                },
            },
        },
    },
    "exists_file": {
        "function": exists_file,
        "schema": {
            "type": "function",
            "function": {
                "name": "exists_file",
                "description": "Check whether a file or directory exists at the given path",
                "parameters": {
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        },
    },
    "stat_file": {
        "function": stat_file,
        "schema": {
            "type": "function",
            "function": {
                "name": "stat_file",
                "description": "Get size, modified time, and type for the file or directory at the given path",
                "parameters": {
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        },
    },
    "make_directory": {
        "function": make_directory,
        "schema": {
            "type": "function",
            "function": {
                "name": "make_directory",
                "description": "Create a directory at the given path. The parent directory must already exist",
                "parameters": {
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        },
    },
    "move": {
        "function": move,
        "schema": {
            "type": "function",
            "function": {
                "name": "move",
                "description": "Move or rename a file from src_path to dst_path",
                "parameters": {
                    "properties": {
                        "src_path": {"type": "string"},
                        "dst_path": {"type": "string"},
                    },
                    "required": ["src_path", "dst_path"],
                    "additionalProperties": False,
                },
            },
        },
    },
    "copy": {
        "function": copy,
        "schema": {
            "type": "function",
            "function": {
                "name": "copy",
                "description": "Copy a file from src_path to dst_path. The destination's parent directory must already exist",
                "parameters": {
                    "properties": {
                        "src_path": {"type": "string"},
                        "dst_path": {"type": "string"},
                    },
                    "required": ["src_path", "dst_path"],
                    "additionalProperties": False,
                },
            },
        },
    },
}

TOOL_SCHEMAS = [t["schema"] for t in TOOLS.values()]
