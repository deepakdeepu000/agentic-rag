import json
import os
import tempfile
import unittest
from pathlib import Path

from core.config import Settings
from utils.mcp_result_utils import extract_mcp_text
from core.mcp_tools import build_remote_mcp_config
from core.memory import SQLiteSessionStore, maybe_generate_session_title


class DummyToolResult:
    def __init__(self, content):
        self.content = content


class TestMcpResultExtraction(unittest.TestCase):
    def test_extracts_nested_text_payload_from_web_search_style_result(self):
        payload = {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "results": [
                                {
                                    "title": "Rajahmundry Weather",
                                    "link": "https://example.com/a",
                                    "snippet": "Mostly cloudy.",
                                    "page_content": "Could not retrieve content from: https://example.com/a\n\nSource: https://example.com/a\n",
                                },
                                {
                                    "title": "WeatherBug",
                                    "link": "https://example.com/b",
                                    "snippet": "Hot and humid with storms.",
                                    "page_content": "WeatherBug forecast details here.",
                                },
                            ]
                        }
                    ),
                }
            ],
            "isError": False,
        }

        chunks = extract_mcp_text(payload)

        self.assertEqual(len(chunks), 2)
        self.assertIn("Rajahmundry Weather", chunks[0])
        self.assertIn("Mostly cloudy.", chunks[0])
        self.assertIn("Source: https://example.com/a", chunks[0])
        self.assertIn("WeatherBug", chunks[1])
        self.assertIn("WeatherBug forecast details here.", chunks[1])

    def test_extracts_dict_result_shape_from_other_mcp_tools(self):
        payload = {
            "result": {
                "title": "Flight lookup",
                "url": "https://example.com/flight",
                "snippet": "Flight delayed by 20 minutes.",
                "page_content": "Flight delayed by 20 minutes. Gate unchanged.",
            }
        }

        chunks = extract_mcp_text(payload)

        self.assertEqual(len(chunks), 1)
        self.assertIn("Flight lookup", chunks[0])
        self.assertIn("Flight delayed by 20 minutes.", chunks[0])
        self.assertIn("Source: https://example.com/flight", chunks[0])

    def test_extracts_plain_string_and_object_content(self):
        self.assertEqual(extract_mcp_text("plain text response"), ["plain text response"])
        self.assertEqual(extract_mcp_text(DummyToolResult([{"type": "text", "text": "hello"}])), ["hello"])

    def test_extracts_chat_model_content_blocks(self):
        content = [
            {
                "type": "text",
                "text": "The current weather in Rajahmundry is sunny.",
                "extras": {"signature": "abc123"},
            }
        ]

        self.assertEqual(
            extract_mcp_text(content),
            ["The current weather in Rajahmundry is sunny."],
        )

    def test_extracts_common_alias_keys(self):
        payload = {
            "context": "Context text",
            "answer": "Final answer text",
            "teext": "Typo text alias",
            "outout": "Typo output alias",
        }

        chunks = extract_mcp_text(payload)

        self.assertIn("Context text", chunks)
        self.assertIn("Final answer text", chunks)
        self.assertIn("Typo text alias", chunks)
        self.assertIn("Typo output alias", chunks)

    def test_falls_back_to_snippet_when_page_content_is_unavailable(self):
        payload = {
            "results": [
                {
                    "title": "Weather source",
                    "link": "https://example.com/weather",
                    "snippet": "Current conditions are hot and humid.",
                    "page_content": "Could not retrieve content from: https://example.com/weather\n\nSource: https://example.com/weather\n",
                }
            ]
        }

        chunks = extract_mcp_text(payload)

        self.assertEqual(len(chunks), 1)
        self.assertIn("Weather source", chunks[0])
        self.assertIn("Current conditions are hot and humid.", chunks[0])
        self.assertIn("Source: https://example.com/weather", chunks[0])


class TestRemoteMcpConfig(unittest.TestCase):
    def test_builds_multiple_server_config_and_backward_compatible_remote_url(self):
        settings = Settings.model_validate(
            {
                "REMOTE_MCP_SERVERS": [
                    {"name": "weather_mcp_server", "url": "http://localhost:8001/sse", "transport": "sse"},
                    {"name": "flight_mcp_server", "url": "http://localhost:8002/sse", "transport": "sse"},
                ],
                "REMOTE_MCP_URL": "http://localhost:8000/sse",
                "REMOTE_MCP_TOKEN": "secret-token",
                "REMOTE_MCP_TENANT_ID": "tenant-1",
            }
        )

        config = build_remote_mcp_config(settings)

        self.assertIn("weather_mcp_server", config)
        self.assertIn("flight_mcp_server", config)
        self.assertIn("remote", config)
        self.assertEqual(config["remote"]["url"], "http://localhost:8000/sse")
        self.assertEqual(config["remote"]["transport"], "sse")
        self.assertEqual(config["remote"]["headers"]["Authorization"], "Bearer secret-token")
        self.assertEqual(config["remote"]["headers"]["X-Tenant-ID"], "tenant-1")


class TestSessionListing(unittest.IsolatedAsyncioTestCase):
    async def test_list_sessions_returns_created_sessions(self):
        temp_dir = tempfile.mkdtemp()
        store = None
        db_path = str(Path(temp_dir) / "sessions.db")
        try:
            store = SQLiteSessionStore(db_path=db_path)

            await store.create_session("session-a")
            await store.create_session("session-b")
            await store.save_message("session-a", "human", "hello")
            sessions = await store.list_sessions(limit=10)

            session_ids = {session.session_id for session in sessions}
            self.assertIn("session-a", session_ids)
            self.assertIn("session-b", session_ids)

            by_id = {session.session_id: session for session in sessions}
            self.assertEqual(by_id["session-a"].title, "")
            self.assertEqual(by_id["session-a"].msg_count, 1)
            self.assertEqual(by_id["session-b"].msg_count, 0)
        finally:
            if store is not None:
                del store
            if os.path.exists(db_path):
                os.remove(db_path)
            os.rmdir(temp_dir)

class TestSessionTitleGeneration(unittest.IsolatedAsyncioTestCase):
    async def test_generates_title_only_once_for_first_user_message(self):
        temp_dir = tempfile.mkdtemp()
        store = None
        db_path = str(Path(temp_dir) / "sessions.db")
        calls = 0

        async def fake_title_provider(settings, user_question):
            nonlocal calls
            calls += 1
            return f"Title: {user_question}"

        try:
            store = SQLiteSessionStore(db_path=db_path)
            await store.create_session("session-a")
            await store.save_message("session-a", "human", "How do I connect the agent to FastAPI?")

            settings = Settings()
            title = await maybe_generate_session_title(
                "session-a",
                store,
                settings,
                "How do I connect the agent to FastAPI?",
                title_provider=fake_title_provider,
            )

            self.assertEqual(calls, 1)
            self.assertEqual(title, "How do I connect the agent to FastAPI?")

            session = await store.load_session("session-a")
            self.assertEqual(session.title, "How do I connect the agent to FastAPI?")

            second_title = await maybe_generate_session_title(
                "session-a",
                store,
                settings,
                "How do I connect the agent to FastAPI?",
                title_provider=fake_title_provider,
            )

            self.assertEqual(calls, 1)
            self.assertEqual(second_title, "How do I connect the agent to FastAPI?")
        finally:
            if store is not None:
                del store
            if os.path.exists(db_path):
                os.remove(db_path)
            os.rmdir(temp_dir)


if __name__ == "__main__":
    unittest.main()