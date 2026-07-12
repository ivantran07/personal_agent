from tools.base import ToolEntry
from tools.files import TOOLS as FILE_TOOLS
from tools.math import TOOLS as MATH_TOOLS
from tools.web import TOOLS as WEB_TOOLS

TOOLS: dict[str, ToolEntry] = {**FILE_TOOLS, **MATH_TOOLS, **WEB_TOOLS}
TOOL_SCHEMAS = [t["schema"] for t in TOOLS.values()]
