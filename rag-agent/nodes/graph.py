"""
graph.py — LangGraph state machine: coordinator → retriever / tool_executor / summarizer → END.

Flow:
  User query
      │
  coordinator ──→ retriever_node       (when tool_call.name == "retrieve")
      │       ──→ tool_executor        (when any other tool call)
      │       ──→ summarizer_node      (when messages_since_last_summary hits threshold)
      └───────→ END                    (no tool call OR iterations maxed)

Every non-END node loops back to coordinator so results are processed and
the next decision can be made.
"""
import logging

from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from core.config import Settings
from nodes.coordinator import build_retrieve_tool, make_coordinator_node
from core.memory import SQLiteSessionStore, summarizer_node
from core.mcp_tools import build_mcp_tools, print_tool_summary
from nodes.retriever import build_chroma_stores, build_embeddings, make_retriever_node
from nodes.router import make_intent_router_node
from core.state import RAGState


logger = logging.getLogger(__name__)


# ── Router ────────────────────────────────────────────────────────────────────

def coordinator_router(state: RAGState) -> str:
    """
    Inspect the coordinator's last message and return the next node name.
    Priority order:
      1. Hard stop: iterations >= max_iterations → END
      2. Context bloat: messages_since_last_summary >= summarize_every → summarizer
      3. Tool call == "retrieve" → retriever
      4. Any other tool call → tool_executor
      5. No tool call → END (final answer)
    """
    if state["iterations"] >= state["max_iterations"]:
        logger.info("Coordinator router: max iterations reached -> END")
        return END

    if state["messages_since_last_summary"] >= state["summarize_every"]:
        logger.info("Coordinator router: summary threshold reached -> summarizer")
        return "summarizer"

    last = state["messages"][-1] if state["messages"] else None
    if last and hasattr(last, "tool_calls") and last.tool_calls:
        tool_name = last.tool_calls[0]["name"]
        if tool_name == "retrieve":
            logger.info("Coordinator router: tool call retrieve -> retriever")
            return "retriever"
        logger.info("Coordinator router: tool call %s -> tool_executor", tool_name)
        return "tool_executor"

    logger.info("Coordinator router: no tool call -> END")
    return END


def intent_router(state: RAGState) -> str:
    """Route based on the LLM router decision."""
    route = state.get("route", "direct")
    logger.info("Intent router edge: %s", route)
    return route


def retriever_router(state: RAGState) -> str:
    """After retrieval, return to coordinator for tool choice/final synthesis."""
    logger.info("Retriever router: returning to coordinator")
    return "coordinator"


# ── Graph factory ─────────────────────────────────────────────────────────────

async def build_graph(settings: Settings):
    """
    Build and compile the full agentic RAG graph.
    Returns (graph, vectorstore, session_store) so run.py can wire everything.
    """
    # 1. Vector store (connects to your existing Chroma DB — read only)
    #    Embeddings are built from the local embedding model configured in settings.
    logger.info("Building graph")
    embeddings = build_embeddings(settings)
    vectorstores = build_chroma_stores(settings, embeddings)
    logger.info("Initialized %d Chroma store(s)", len(vectorstores))

    # 2. Retrieve tool (schema for LLM binding)
    retrieve_tool = build_retrieve_tool(next(iter(vectorstores.values())))

    # 3. Remote MCP tools (graceful if not configured)
    mcp_tools = await build_mcp_tools(settings)
    logger.info("Available MCP tools:")
    print_tool_summary(mcp_tools)

    # 4. All tools combined
    all_tools = [retrieve_tool] + mcp_tools

    # 5. Nodes
    intent_router_node = make_intent_router_node(mcp_tools, settings)
    coordinator_node = make_coordinator_node(all_tools, settings)
    retriever_node   = make_retriever_node(vectorstores, settings)
    tool_executor    = ToolNode(tools=mcp_tools, handle_tool_errors=True)

    # 6. Session store
    session_store = SQLiteSessionStore(db_path=settings.sqlite_db_path)

    # 7. Build graph
    builder = StateGraph(RAGState)
    builder.add_node("intent_router", intent_router_node)
    builder.add_node("coordinator",   coordinator_node)
    builder.add_node("retriever",     retriever_node)
    builder.add_node("tool_executor", tool_executor)
    builder.add_node("summarizer",    summarizer_node)

    builder.set_entry_point("intent_router")

    builder.add_conditional_edges(
        "intent_router",
        intent_router,
        {
            "retrieve": "retriever",
            "external": "coordinator",
            "both": "retriever",
            "direct": "coordinator",
        },
    )

    # Coordinator fans out based on router
    builder.add_conditional_edges("coordinator", coordinator_router)

    # All worker nodes return to coordinator
    builder.add_conditional_edges("retriever", retriever_router)
    builder.add_edge("tool_executor", "coordinator")
    builder.add_edge("summarizer",    "coordinator")

    graph = builder.compile()
    logger.info("Graph compiled")
    return graph, vectorstores, session_store
