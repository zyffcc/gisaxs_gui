"""GIMaP's tools as a local MCP server (Streamable HTTP transport, JSON responses).

Claude Code connects to it for one run.  It listens on 127.0.0.1 only, on a
free port; every request must carry the run's bearer token, and requests sent
from a web page (an ``Origin`` header that is not local) are refused.  Only the
parts of MCP that tools need are implemented (``mcp_protocol``).
"""

from __future__ import annotations

import json
import secrets
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional, Sequence

from .mcp_protocol import DEFAULT_PROTOCOL_VERSION, SERVER_NAME, McpDispatcher, mcp_content, mcp_tool, rpc_error

ENDPOINT = "/mcp"
LOCAL_ORIGINS = ("http://127.0.0.1", "http://localhost", "https://127.0.0.1", "https://localhost")


class _QuietServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address) -> None:
        # A client that closes its connection while exiting is normal; anything else is reported.
        if not isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            super().handle_error(request, client_address)


class LocalMcpServer:
    def __init__(
        self,
        tools: Sequence[dict],
        call_tool: Callable[[str, dict], Any],
        *,
        name: str = SERVER_NAME,
        instructions: str = "",
    ):
        self.name = name
        self.token = secrets.token_urlsafe(32)
        self.dispatcher = McpDispatcher(tools, call_tool, name=name, instructions=instructions)
        self.tools = self.dispatcher.tools
        handler = type("GimapMcpHandler", (_Handler,), {"server_owner": self})
        self._httpd = _QuietServer(("127.0.0.1", 0), handler)
        self._thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}{ENDPOINT}"

    def config(self) -> dict:
        """The entry for ``mcpServers`` in a Claude Code MCP configuration."""
        return {"type": "http", "url": self.url, "headers": {"Authorization": f"Bearer {self.token}"}}

    def start(self) -> "LocalMcpServer":
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="gimap-mcp", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    def __enter__(self) -> "LocalMcpServer":
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.stop()

    def handle(self, message: dict) -> Optional[dict]:
        """The response to one JSON-RPC message; ``None`` for notifications and responses."""
        return self.dispatcher.handle(message)


class _Handler(BaseHTTPRequestHandler):
    server_owner: LocalMcpServer
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args) -> None:  # no console noise from the GUI
        pass

    def _send(self, status: int, body: Optional[dict] = None, headers: Sequence[tuple[str, str]] = ()) -> None:
        data = b"" if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        if body is not None:
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for key, value in headers:
            self.send_header(key, value)
        self.end_headers()
        if data:
            self.wfile.write(data)

    def _allowed(self) -> bool:
        origin = self.headers.get("Origin")
        if origin and not origin.startswith(LOCAL_ORIGINS):
            self._send(HTTPStatus.FORBIDDEN)
            return False
        expected = f"Bearer {self.server_owner.token}"
        if not secrets.compare_digest(self.headers.get("Authorization", ""), expected):
            self._send(HTTPStatus.UNAUTHORIZED, {"jsonrpc": "2.0", "error": {"code": -32001, "message": "Unauthorized"}})
            return False
        return True

    def do_POST(self) -> None:
        # Read the body before any answer: answering first and closing with unread
        # data resets the connection on Windows, and the client never sees the status.
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length > 0 else b""
        if self.path.split("?", 1)[0] != ENDPOINT:
            self._send(HTTPStatus.NOT_FOUND)
            return
        if not self._allowed():
            return
        try:
            message = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            self._send(HTTPStatus.BAD_REQUEST, rpc_error(None, -32700, "Parse error"))
            return
        if not isinstance(message, dict):
            self._send(HTTPStatus.BAD_REQUEST, rpc_error(None, -32600, "Send one JSON-RPC message per request"))
            return
        response = self.server_owner.handle(message)
        if response is None:
            self._send(HTTPStatus.ACCEPTED)
        else:
            self._send(HTTPStatus.OK, response)

    def do_GET(self) -> None:
        # No server-initiated stream: the tools only answer requests.
        self._send(HTTPStatus.METHOD_NOT_ALLOWED, headers=[("Allow", "POST, DELETE")])

    def do_DELETE(self) -> None:
        if self._allowed():
            self._send(HTTPStatus.OK)


__all__ = ["DEFAULT_PROTOCOL_VERSION", "ENDPOINT", "LocalMcpServer", "SERVER_NAME", "mcp_content", "mcp_tool"]
