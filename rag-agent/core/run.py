"""
run.py — Entry point for the agentic RAG system.

Usage:
    python run.py                        # new session (auto UUID)
    python run.py --session my-session   # resume named session

The CLI loop:
  1. Load (or create) session from SQLite
  2. Accept user input
  3. Save user message to DB
  4. Invoke graph
  5. Save assistant response to DB
  6. Print answer
  7. Repeat
"""
import argparse
import asyncio
import logging
import uuid

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from core.config import Settings
from nodes.graph import build_graph
from utils.logging_utils import setup_logging
from core.memory import load_session_into_state
from utils.mcp_result_utils import extract_mcp_text


logger = logging.getLogger(__name__)


async def chat_loop(session_id: str, settings: Settings):
    setup_logging(settings)

    print(f"\n{'='*60}")
    print("  Agentic RAG — Chroma + Ollama + MCP")
    print(f"  Session: {session_id}")
    print(f"  Model:   {settings.ollama_model} @ {settings.ollama_base_url}")
    print(f"  Chroma:  {settings.chroma_persist_dir} / {settings.chroma_collection}")
    print(f"{'='*60}")
    print("  Type 'exit' or Ctrl-C to quit.\n")
    logger.info("Session started: %s", session_id)
    logger.info("Model: %s @ %s", settings.ollama_model, settings.ollama_base_url)
    logger.info("Chroma: %s / %s", settings.chroma_persist_dir, settings.chroma_collection)
    logger.info("Logging to file: %s", settings.log_file_path)

    # Build graph + supporting objects once
    graph, retrieval_collections, session_store = await build_graph(settings)
    logger.info("Graph ready with %d retrieval collection(s)", len(retrieval_collections))

    # RunnableConfig passes store and settings into every node via config["configurable"]
    run_config = RunnableConfig(
        configurable={
            "session_store": session_store,
            "settings": settings,
            "thread_id": session_id,
        }
    )

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Goodbye]")
            break

        if not user_input or user_input.lower() in {"exit", "quit", "q"}:
            print("[Goodbye]")
            break

        logger.info("User input received: %s", user_input)

        # ── Load fresh session state before every turn ─────────────────
        state = await load_session_into_state(session_id, session_store, settings)
        logger.info(
            "Loaded state: messages=%d retrieval_attempts=%d route=%s",
            len(state.get("messages", [])),
            state.get("retrieval_attempts", 0),
            state.get("route", "direct"),
        )

        # Save incoming user message to DB
        await session_store.save_message(session_id, "human", user_input)
        logger.debug("Saved human message to session store")

        # Merge user turn into state
        state["messages"].append(HumanMessage(content=user_input))
        state["query"] = user_input
        state["messages_since_last_summary"] += 1
        logger.debug("Prepared graph state for invocation")

        # ── Run the graph ──────────────────────────────────────────────
        print("Assistant: ", end="", flush=True)
        try:
            result = await graph.ainvoke(state, config=run_config)
        except Exception as e:
            print(f"\n[ERROR] Graph execution failed: {e}")
            logger.exception("Graph execution failed")
            continue

        # ── Extract and display final answer ───────────────────────────
        final_msg = result["messages"][-1]
        raw_answer = getattr(final_msg, "content", None)
        if raw_answer is None:
            raw_answer = getattr(final_msg, "text", None)
        if raw_answer is None:
            raw_answer = getattr(final_msg, "answer", None)
        if raw_answer is None:
            raw_answer = final_msg
        answer_parts = extract_mcp_text(raw_answer)
        answer = "\n\n".join(answer_parts) if answer_parts else str(raw_answer)

        print(answer)
        print()
        logger.info("Assistant answer: %s", answer)

        # Save assistant response to DB
        await session_store.save_message(session_id, "ai", answer)
        logger.debug("Saved assistant message to session store")


def main():
    parser = argparse.ArgumentParser(description="Agentic RAG — Chroma + Ollama + MCP")
    parser.add_argument(
        "--session",
        default=None,
        help="Session ID to resume. Omit to start a fresh session.",
    )
    args = parser.parse_args()

    session_id = args.session or f"session-{uuid.uuid4().hex[:8]}"
    settings = Settings()

    asyncio.run(chat_loop(session_id, settings))


if __name__ == "__main__":
    main()
