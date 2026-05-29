"""
web_search.py - MCP-backed web search node with selective payload extraction.
"""
import logging
from typing import Any

from langchain_core.messages import ToolMessage

from core.config import Settings
from utils.mcp_result_utils import extract_mcp_text
from core.state import RAGState


logger = logging.getLogger(__name__)


def make_web_search_node(mcp_tools: list, settings: Settings):
    """Create a tool_call node using MCP tools and payload filtering."""
    tool_name = settings.web_search_tool_name
    logger.info("Web search node initialized with tool name: %s", tool_name)

    async def web_search_node(state: RAGState, config) -> dict:
        query = state.get("query") or ""
        last_msg = state["messages"][-1] if state.get("messages") else None
        if not query and last_msg and hasattr(last_msg, "content"):
            query = last_msg.content

        logger.info("Web search node invoked: query=%r", query[:200])

        tool = next((t for t in mcp_tools if t.name == tool_name), None)
        if tool is None:
            logger.warning("MCP web search tool '%s' not found", tool_name)
            return {
                "web_search_done": True,
                "messages": [
                    ToolMessage(
                        content="Web search tool not configured.",
                        tool_call_id="web_search_call",
                    )
                ],
            }

        result = await tool.ainvoke({"query": query})
        logger.info("Web search tool '%s' returned type=%s", tool_name, type(result).__name__)
        page_chunks = extract_mcp_text(result)
        content = "\n\n".join(page_chunks) if page_chunks else "No web results found."
        logger.info("Web search extracted %d page chunk(s)", len(page_chunks))

        return {
            "web_search_done": True,
            "messages": [ToolMessage(content=content, tool_call_id="web_search_call")],
        }

    return web_search_node
