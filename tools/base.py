from collections.abc import Callable
from typing import Any, TypedDict


class ToolEntry(TypedDict):
    function: Callable[..., Any]
    schema: dict[str, Any]
