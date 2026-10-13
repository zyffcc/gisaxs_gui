"""The part of MCP that tools need, independent of the transport.

``McpDispatcher`` answers one JSON-RPC message: ``initialize``, ``ping``,
``tools/list`` and ``tools/call``.  The local HTTP server (Claude Code inside
GIMaP) and the stdio server (command-line agents such as Codex) both use it.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional, Sequence

SERVER_NAME = "gimap"
DEFAULT_PROTOCOL_VERSION = "2025-06-18"


def mcp_tool(definition: dict) -> dict:
    """An API tool definition as an MCP tool."""
    return {
        "name": definition["name"],
        "description": definition.get("description", ""),
        "inputSchema": definition.get("input_schema") or {"type": "object"},
    }


def mcp_content(content: Any) -> list[dict]:
    """Tool output (a string, or API text / image blocks) as MCP content blocks."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    blocks = []
    for block in content or ():
        kind = block.get("type") if isinstance(block, dict) else None
        if kind == "image":
            source = block.get("source") or {}
            blocks.append({"type": "image", "data": source.get("data", ""), "mimeType": source.get("media_type", "image/png")})
        elif kind == "text":
            blocks.append({"type": "text", "text": str(block.get("text", ""))})
        else:
            blocks.append({"type": "text", "text": json.dumps(block, ensure_ascii=False, default=str)})
    return blocks or [{"type": "text", "text": ""}]


def rpc_error(ident, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": ident, "error": {"code": code, "message": message}}


class McpDispatcher:
    """Tools behind MCP: ``call_tool(name, arguments)`` returns an outcome with ``content`` and ``is_error``."""

    def __init__(
        self,
        tools: Sequence[dict],
        call_tool: Callable[[str, dict], Any],
        *,
        name: str = SERVER_NAME,
        instructions: str = "",
    ):
        self.name = name
        self.tools = [mcp_tool(definition) for definition in tools]
        self._names = {tool["name"] for tool in self.tools}
        self._call_tool = call_tool
        self._instructions = instructions

    def handle(self, message: dict) -> Optional[dict]:
        """The response to one JSON-RPC message; ``None`` for notifications and responses."""
        if "id" not in message or "method" not in message:
            return None
        ident, method = message["id"], message["method"]
        params = message.get("params") or {}
        if method == "initialize":
            result = {
                "protocolVersion": params.get("protocolVersion") or DEFAULT_PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": self.name, "version": "1.0"},
            }
            if self._instructions:
                result["instructions"] = self._instructions
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": self.tools}
        elif method == "tools/call":
            result = self._call(params)
        else:
            return rpc_error(ident, -32601, f"Method not found: {method}")
        return {"jsonrpc": "2.0", "id": ident, "result": result}

    def _call(self, params: dict) -> dict:
        name = params.get("name")
        if name not in self._names:
            return {"content": [{"type": "text", "text": f"Unknown tool '{name}'."}], "isError": True}
        arguments = params.get("arguments")
        try:
            outcome = self._call_tool(name, arguments if isinstance(arguments, dict) else {})
        except Exception as exc:  # reported to the model, never raised into the server
            return {"content": [{"type": "text", "text": f"The tool failed: {exc}"}], "isError": True}
        return {"content": mcp_content(outcome.content), "isError": bool(outcome.is_error)}


__all__ = ["DEFAULT_PROTOCOL_VERSION", "McpDispatcher", "SERVER_NAME", "mcp_content", "mcp_tool", "rpc_error"]
