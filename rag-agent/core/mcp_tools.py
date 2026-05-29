"""
mcp_tools.py — Remote SSE MCP client factory.

Design:
- Connects to remote MCP server(s) over HTTP/SSE transport.
- Returns a flat list of LangChain-compatible Tool objects ready for ToolNode.
- Fails gracefully: if a server is unreachable at startup, it logs a warning
  and returns an empty list so the rest of the graph still works.
- Add more servers to REMOTE_MCP_CONFIG or drive from Settings.
"""
import logging

from langchain_mcp_adapters.client import MultiServerMCPClient

from core.config import Settings


logger = logging.getLogger(__name__)


def build_remote_mcp_config(settings: Settings) -> dict:
    """
    Build the MultiServerMCPClient config dict from settings.
    Supports multiple servers and keeps single-URL backward compatibility.
    """
    config = {}

    for index, mcp_server in enumerate(settings.remote_mcp_servers, start=1):
        url = mcp_server.get("url")
        if not url:
            continue

        name = mcp_server.get("name") or f"remote_{index}"
        transport = mcp_server.get("transport") or "sse"

        headers = {}
        token = mcp_server.get("token")
        tenant_id = mcp_server.get("tenant_id")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if tenant_id:
            headers["X-Tenant-ID"] = tenant_id

        config[name] = {
            "url": url,
            "transport": transport,
            **({"headers": headers} if headers else {}),
        }

    if settings.remote_mcp_url:
        headers = {}
        if settings.remote_mcp_token:
            headers["Authorization"] = f"Bearer {settings.remote_mcp_token}"
        if settings.remote_mcp_tenant_id:
            headers["X-Tenant-ID"] = settings.remote_mcp_tenant_id

        config.setdefault(
            "remote",
            {
                "url": settings.remote_mcp_url,
                "transport": "sse",
                **({"headers": headers} if headers else {}),
            },
        )

    return config


async def build_mcp_tools(settings: Settings) -> list:
    """
    Connect to configured remote MCP servers and return all tools.
    Returns [] gracefully if no servers are configured or a server is down.
    """
    config = build_remote_mcp_config(settings)

    if not config:
        logger.info("No remote MCP servers configured. Set REMOTE_MCP_SERVERS or REMOTE_MCP_URL in .env to enable.")
        return []

    try:
        endpoints = ", ".join(
            f"{name}={server.get('url')}" for name, server in config.items()
        )
        logger.info("Connecting to remote MCP server(s): %s", endpoints)
        client = MultiServerMCPClient(config)
        tools = await client.get_tools()
        logger.info("Loaded %d MCP tool(s): %s", len(tools), [t.name for t in tools])
        return tools
    except Exception as e:
        logger.warning("Remote MCP server unavailable or not an MCP SSE endpoint — skipping. Error: %s", e)
        return []


def print_tool_summary(tools: list) -> None:
    """Debug helper: print all available tools and their descriptions."""
    if not tools:
        logger.info("  (no tools loaded)")
        return
    for t in tools:
        desc = (t.description or "").replace("\n", " ").strip()
        logger.info("  %s: %s", t.name, desc)
