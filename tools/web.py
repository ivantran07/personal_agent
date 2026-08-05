import os

import requests
import trafilatura
from ddgs import DDGS

from tools.base import ToolEntry

MAX_FETCH_BYTES = int(os.environ.get("MAX_FETCH_BYTES", "2000000"))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "15"))
USER_AGENT = "personal-agent/0.1"


def fetch_url(url: str) -> str:
    response = requests.get(
        url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT}
    )
    response.raise_for_status()
    if len(response.content) > MAX_FETCH_BYTES:
        raise ValueError(
            f"Response too large ({len(response.content)} bytes, limit {MAX_FETCH_BYTES})"
        )
    text = trafilatura.extract(response.content, url=url)
    if not text:
        raise ValueError(f"Could not extract readable content from {url}")
    return text


def web_search(query: str, max_results: int = 5) -> list[dict]:
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results)
    return [
        {"title": r.get("title"), "url": r.get("href"), "snippet": r.get("body")}
        for r in results
    ]


TOOLS: dict[str, ToolEntry] = {
    "fetch_url": {
        "function": fetch_url,
        "schema": {
            "type": "function",
            "function": {
                "name": "fetch_url",
                "description": "Fetch a web page and return its main readable text content, with boilerplate (nav, ads, footers) stripped",
                "parameters": {
                    "properties": {
                        "url": {"type": "string", "description": "URL to fetch"},
                    },
                    "required": ["url"],
                    "additionalProperties": False,
                },
            },
        },
    },
    "web_search": {
        "function": web_search,
        "schema": {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web and return a list of results with title, url, and snippet",
                "parameters": {
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return, defaults to 5",
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        },
    },
}

TOOL_SCHEMAS = [t["schema"] for t in TOOLS.values()]
