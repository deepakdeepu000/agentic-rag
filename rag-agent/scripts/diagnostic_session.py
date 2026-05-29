"""
Diagnostic script to validate session initialization and message history.
"""
import asyncio
import uuid

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from core.config import Settings
from core.memory import SQLiteSessionStore, load_session_into_state


def _label_message(message) -> str:
    if isinstance(message, SystemMessage):
        label = "SystemMessage"
    elif isinstance(message, HumanMessage):
        label = "HumanMessage"
    elif isinstance(message, AIMessage):
        label = "AIMessage"
    else:
        label = type(message).__name__
    return f"[{label}] {getattr(message, 'content', '')}"


async def main():
    settings = Settings()
    session_store = SQLiteSessionStore(db_path=settings.sqlite_db_path)
    session_id = f"diagnostic-{uuid.uuid4().hex[:8]}"

    state = await load_session_into_state(session_id, session_store, settings)

    print(f"session_id: {session_id}")
    print("messages:")
    for message in state["messages"]:
        print(_label_message(message))


if __name__ == "__main__":
    asyncio.run(main())
