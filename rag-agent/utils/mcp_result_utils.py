"""Utilities for normalizing heterogeneous MCP tool results into text."""
from __future__ import annotations

import json
from typing import Any


_TEXT_KEYS = (
    "title",
    "snippet",
    "page_content",
    "link",
    "url",
    "source",
    "answer",
    "context",
)
_NESTED_KEYS = (
    "content",
    "context",
    "result",
    "results",
    "data",
    "text",
    "output",
    "answer",
    "response",
    "messages",
    "teext",
    "outout",
)


def _parse_json_string(value: str) -> Any:
    """Parse JSON strings repeatedly while the payload still looks like JSON."""
    current: Any = value
    for _ in range(3):
        if not isinstance(current, str):
            break
        candidate = current.strip()
        if not candidate:
            break
        if candidate[0] not in "[{\"":
            break
        try:
            current = json.loads(candidate)
        except json.JSONDecodeError:
            break
    return current


def _format_mapping(entry: dict[str, Any]) -> str:
    """Convert a mapping with common web/MCP fields into readable context."""
    title = str(entry.get("title") or "").strip()
    snippet = str(entry.get("snippet") or "").strip()
    page_content = str(entry.get("page_content") or "").strip()
    link = str(entry.get("link") or entry.get("url") or entry.get("source") or "").strip()
    answer = str(entry.get("answer") or "").strip()
    context = str(entry.get("context") or "").strip()
    response = str(entry.get("response") or "").strip()
    output = str(entry.get("output") or "").strip()

    parts: list[str] = []

    if title:
        parts.append(title)

    if answer:
        parts.append(answer)
    elif context:
        parts.append(context)
    elif response:
        parts.append(response)
    elif output:
        parts.append(output)

    if page_content and not page_content.startswith("Could not retrieve content from:"):
        if snippet and snippet != title:
            parts.append(snippet)
        parts.append(page_content)
    elif snippet and snippet != title:
        parts.append(snippet)

    if link:
        parts.append(f"Source: {link}")

    return "\n".join(parts)


def extract_mcp_text(payload: Any) -> list[str]:
    """Extract human-readable text from common MCP response shapes.

    Supports:
    - plain strings
    - JSON strings nested inside strings
    - dicts with `content`, `result`, `results`, `data`, `text`, or `output`
    - lists of any of the above
    - mappings with common search fields like `title`, `snippet`, `page_content`, and `link`
    """
    chunks: list[str] = []
    seen: set[int] = set()

    def visit(value: Any) -> None:
        if value is None:
            return

        if isinstance(value, str):
            parsed = _parse_json_string(value)
            if parsed is value:
                stripped = value.strip()
                if stripped:
                    chunks.append(stripped)
                return
            visit(parsed)
            return

        if hasattr(value, "content") and not isinstance(value, (dict, list)):
            visit(getattr(value, "content"))
            return

        object_id = id(value)
        if object_id in seen:
            return
        seen.add(object_id)

        if isinstance(value, dict):
            if value.get("type") == "text" and value.get("text"):
                visit(value.get("text"))
                return

            for key in _NESTED_KEYS:
                nested = value.get(key)
                if nested is None:
                    continue
                if key == "content" and isinstance(nested, list):
                    text_blobs = [
                        item.get("text")
                        for item in nested
                        if isinstance(item, dict) and item.get("type") == "text" and item.get("text")
                    ]
                    if text_blobs:
                        for text_blob in text_blobs:
                            visit(text_blob)
                        continue
                visit(nested)

            if any(key in value for key in _TEXT_KEYS):
                formatted = _format_mapping(value)
                if formatted:
                    chunks.append(formatted)
                return

            for nested in value.values():
                if isinstance(nested, (dict, list, str)):
                    visit(nested)
            return

        if isinstance(value, list):
            for item in value:
                visit(item)
            return

        text = str(value).strip()
        if text:
            chunks.append(text)

    visit(payload)
    return chunks