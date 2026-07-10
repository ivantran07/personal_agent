from pathlib import Path
from mcp.server.fastmcp import FastMCP

FILES_DIR = Path("files").resolve()

mcp = FastMCP("tools")


def _file_names() -> list[str]:
    return sorted(p.name for p in FILES_DIR.iterdir() if p.is_file())


@mcp.tool()
def list_files() -> dict:
    """Lists the names of the files in the shared folder. Call this first to see what exists; pass the returned names directly to read_file."""
    return {"files": _file_names()}


@mcp.tool()
def read_file(path: str) -> dict:
    """Reads the full text content of one file from the shared folder. Takes a bare filename like 'notes.txt'."""
    resolved = FILES_DIR / Path(path).name
    if not resolved.is_file():
        return {"error": f"File not found. Available files: {_file_names()}"}
    return {"content": resolved.read_text(encoding="utf8")}


@mcp.tool()
def write_file(path: str, text: str) -> dict:
    """Creates or overwrites a file in the shared folder with the given text. Pass just a filename, e.g. 'output.txt'."""
    resolved = FILES_DIR / Path(path).name
    resolved.write_text(text, encoding="utf8")
    return {"status": "success", "file": resolved.name}


@mcp.tool()
def delete_file(path: str) -> dict:
    """Deletes one file from the shared folder. Takes a bare filename like 'notes.txt'. This is irreversible — use list_files or read_file first to confirm you're targeting the right file."""
    resolved = FILES_DIR / Path(path).name
    if not resolved.is_file():
        return {"error": f"File not found. Available files: {_file_names()}"}
    resolved.unlink()
    return {"status": "success", "file": resolved.name}


if __name__ == "__main__":
    mcp.run()
