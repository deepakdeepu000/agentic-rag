"""
router.py - LLM-driven intent routing for adaptive tool execution.
"""
import logging
from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from core.config import Settings
from core.llm_factory import get_chat_model
from core.state import RAGState


logger = logging.getLogger(__name__)


Route = Literal["retrieve", "external", "both", "direct"]


class IntentDecision(BaseModel):
    route: Route = Field(..., description="Routing decision")
    reason: str | None = Field(default=None, description="Brief justification")


ROUTER_SYSTEM_PROMPT = """You are a routing agent for a RAG system.
Your job is to inspect the user query and choose exactly one route.

Routes:
- retrieve: the answer depends on internal/private documents.
- external: the answer requires external/public information or calls to MCP-provided tools.
- both: the answer needs internal docs AND external/MCP-sourced info.
- direct: no tools needed; answer from general knowledge or reasoning.

Return a structured response only."""


def make_intent_router_node(mcp_tools: list, settings: Settings):
    """Create an intent router node that considers available MCP tools.

    The router receives a summary of available MCP tools (name + short description)
    and the recent conversation, then returns a structured routing decision.
    """
    llm = get_chat_model(settings, purpose="router")
    llm_structured = llm.with_structured_output(IntentDecision)

    tools_summary = []
    for t in (mcp_tools or []):
        desc = getattr(t, "description", "") or ""
        tools_summary.append(f"- {t.name}: {desc}")
    tools_block = "\n".join(tools_summary) if tools_summary else "(no MCP tools available)"
    logger.info("Router initialized with %d MCP tool(s)", len(mcp_tools or []))

    async def intent_router_node(state: RAGState, config) -> dict:
        # Prefer explicit query field, fallback to last non-system message
        query = state.get("query") or ""
        if not query and state.get("messages"):
            # Find last non-system message
            for m in reversed(state["messages"]):
                if not hasattr(m, "content"):
                    continue
                query = m.content
                break

        # Include a short window of recent messages for context
        recent = []
        for m in state.get("messages", [])[-6:]:
            t = type(m).__name__
            recent.append(f"[{t}] {getattr(m, 'content', '')[:200]}")
        recent_block = "\n".join(recent) if recent else "(no history)"

        logger.info(
            "Router node invoked: query=%r recent_messages=%d",
            query[:200],
            len(recent),
        )

        prompt_system = (
            ROUTER_SYSTEM_PROMPT
            + "\n\nAvailable MCP tools (name: description):\n"
            + tools_block
            + "\n\nRecent messages:\n"
            + recent_block
        )

        logger.debug("Router prompt system built \n%s", prompt_system)

        response = await llm_structured.ainvoke(
            [
                SystemMessage(content=prompt_system),
                HumanMessage(content=query),
            ]
        )

        logger.info("Router decision: route=%s reason=%s", response.route, response.reason)

        return {
            "route": response.route,
            "retrieval_done": False,
            "web_search_done": False,
        }

    return intent_router_node
