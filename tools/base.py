from typing import Any, Callable, TypedDict


class ToolEntry(TypedDict):
    function: Callable[..., Any]
    schema: dict[str, Any]
