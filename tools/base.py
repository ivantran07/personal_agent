"""Define shared types used to register agent tools."""

from collections.abc import Callable
from typing import Any, TypedDict


class ToolEntry(TypedDict):
    """Describe a callable tool and the schema exposed to the language model."""

    function: Callable[..., Any]
    schema: dict[str, Any]
