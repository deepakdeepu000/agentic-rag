"""
coordinator.py — The main LLM agent node.

Design:
- Binds the retrieve tool + all MCP tools to Ollama via .bind_tools().
- Injects session summary as a SystemMessage[0] (already in state.messages
  from load_session_into_state — coordinator does NOT re-inject it).
- Appends an iteration note when retrieval limit is nearly reached to prevent
  infinite retrieve → coordinator loops.
- Increments iterations and messages_since_last_summary counters.
"""
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from core.config import Settings
from core.llm_factory import get_chat_model
from core.state import RAGState


logger = logging.getLogger(__name__)


# ── System prompt ─────────────────────────────────────────────────────────────

COORDINATOR_SYSTEM_PROMPT = """You are an intelligent research assistant with access to a \
document retrieval system and additional tools.

## Your Capabilities
- **retrieve**: Search the internal knowledge base for relevant document chunks.
- **MCP tools**: One or more tools may be available (search, APIs, systems). Choose the best tool(s) for the user query.

## MCP Tool Preference
- Use the most specific MCP tool that matches the task before generic web search.
- Prefer specialized MCP tools when available; consult the runtime list of available tools below.

## Decision Rules
1. **Follow the route** chosen by the router (direct, retrieve, external, both).
2. **Retrieve at most 3 times** per user question. After 2 failed retrievals, \
synthesize from what you already have.
3. **Use tools purposefully** — only call the tool(s) that materially improve your answer.
4. **Cite sources**: Reference document metadata (source, page) from retrieved chunks.
5. **Be concise**: Prefer direct answers with evidence over lengthy summaries.

## Output Format
- Factual answers: state the answer, then cite [Source: filename, page X].
- Multi-part questions: use brief numbered sections.
- Cannot find information: say so clearly — never guess.
"""


def get_iteration_note(state: RAGState) -> str:
    """Appended to system prompt when retrieval attempts are exhausted."""
    attempts = state["retrieval_attempts"]
    if attempts >= 3:
        return (
            f"\n\n⚠️ RETRIEVAL LIMIT REACHED: You have already retrieved {attempts} time(s). "
            "Do NOT call 'retrieve' again. Synthesize your best answer from existing context."
        )
    return ""


def get_route_note(state: RAGState) -> str:
    """Append route-specific guidance to the system prompt."""
    if state.get("route") == "direct":
        return "\n\nYou must NOT call tools for this turn. Answer directly."
    if state.get("route") in {"external", "both"}:
        return (
            "\n\nFor external information, prefer the most specific MCP tool first. "
            "Use web search only as a fallback for broad public/current information or when the specialized tools do not apply."
        )
    return ""


def build_retrieved_chunks_message(state: RAGState) -> HumanMessage | None:
    """Build a separate message containing retrieved chunks for the LLM."""
    chunks = state.get("retrieved_chunks") or []
    if not chunks:
        return None

    lines = ["Retrieved chunks from the retriever:"]
    for idx, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata") or {}
        source = metadata.get("source") or metadata.get("filename") or "unknown"
        page = metadata.get("page") or metadata.get("page_number")
        page_text = f", page {page}" if page is not None else ""
        lines.append(
            f"{idx}. [score={chunk.get('score', 0.0):.3f} | source={source}{page_text}]\n"
            f"{chunk.get('content', '')}"
        )

    return HumanMessage(content="\n\n".join(lines))


# ── Coordinator factory ───────────────────────────────────────────────────────

def make_coordinator_node(
    all_tools: list,
    settings: Settings,
):
    """
    Factory that binds tools to Ollama and returns a coordinator_node function.
    all_tools = [retrieve_tool] + mcp_tools
    """
    llm = get_chat_model(settings, purpose="coordinator")
    llm_with_tools = llm.bind_tools(all_tools)
    logger.info("Coordinator initialized with %d tool(s)", len(all_tools))

    # Build a runtime summary of available MCP tools (exclude the 'retrieve' schema)
    tools_summary = []
    for t in (all_tools or [])[1:]:
        desc = getattr(t, "description", "") or ""
        tools_summary.append(f"- {t.name}: {desc}")
    tools_block = "\n".join(tools_summary) if tools_summary else "(no MCP tools available)"

    async def coordinator_node(state: RAGState, config: RunnableConfig) -> dict:
        """
        Main coordinator: decides whether to retrieve, call an MCP tool, or answer.
        The session summary SystemMessage is already at messages[0] from startup.
        """
        system_content = (
            COORDINATOR_SYSTEM_PROMPT
            + "\n\nAvailable MCP tools (name: description):\n"
            + tools_block
            + get_iteration_note(state)
            + get_route_note(state)
        )

        # Build final message list:
        # If messages[0] is already a SystemMessage (session context), prepend
        # the coordinator system prompt as an additional SystemMessage.
        # If no prior messages, just use the coordinator prompt.
        messages = list(state["messages"])

        logger.info(
            "Coordinator node invoked: route=%s iterations=%d retrieval_attempts=%d messages=%d",
            state.get("route", "direct"),
            state.get("iterations", 0),
            state.get("retrieval_attempts", 0),
            len(messages),
        )

        # Prepend coordinator system prompt (session context, if any, is already in messages)
        messages = [SystemMessage(content=system_content)] + [
            m for m in messages if not isinstance(m, SystemMessage)
        ]

        # Re-inject session summary after coordinator system prompt if present
        if state["session_summary"]:
            messages.insert(
                1,
                SystemMessage(content=f"[SESSION CONTEXT]\n{state['session_summary']}"),
            )

        retrieved_chunks_message = build_retrieved_chunks_message(state)
        if retrieved_chunks_message is not None:
            messages.insert(2, retrieved_chunks_message)

        if state.get("route") == "direct":
            logger.info("Coordinator invoking direct LLM call")
            response = await llm.ainvoke(messages)
        else:
            logger.info("Coordinator invoking tool-enabled LLM call")
            response = await llm_with_tools.ainvoke(messages)

        # Track which MCP tools were called in this turn
        new_tool_calls = list(state["tool_calls_made"])
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                if tc["name"] != "retrieve" and tc["name"] not in new_tool_calls:
                    new_tool_calls.append(tc["name"])

        logger.info(
            "Coordinator response received: tool_calls=%s",
            [tc["name"] for tc in response.tool_calls] if hasattr(response, "tool_calls") and response.tool_calls else [],
        )
        first_tool = (
            response.tool_calls[0]["name"]
            if hasattr(response, "tool_calls") and response.tool_calls
            else "none"
        )
        logger.info("Coordinator first tool selected: %s", first_tool)

        return {
            "messages": [response],
            "iterations": state["iterations"] + 1,
            "messages_since_last_summary": state["messages_since_last_summary"] + 1,
            "tool_calls_made": new_tool_calls,
        }

    return coordinator_node


# ── Retrieve tool definition ──────────────────────────────────────────────────
# Defined here so graph.py can import it without circular imports.

def build_retrieve_tool(vectorstore) -> StructuredTool:
    """
    Build the `retrieve` StructuredTool that the coordinator can call.
    The actual retrieval logic lives in retriever_node; this is just the
    schema/description the LLM sees when deciding to call it.
    """
    def retrieve(query: str) -> str:
        """Search the internal knowledge base for relevant document chunks."""
        # This sync function is only used for tool schema binding.
        # The actual execution is handled by retriever_node in the graph.
        return f"Searching for: {query}"

    return StructuredTool.from_function(
        func=retrieve,
        name="retrieve",
        description=(
            "Search the internal document knowledge base for relevant chunks. "
            "Use this when you need factual grounding from documents, reports, or stored knowledge. "
            "Input: a specific search query string."
        ),
    )
