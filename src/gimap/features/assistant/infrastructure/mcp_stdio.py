"""MCP over stdio: one JSON-RPC message per line on stdin, answers on stdout.

For command-line agents (Codex, Claude Code, any MCP client) that start GIMaP's
tools as a subprocess.  Only protocol messages go to stdout, UTF-8 encoded;
everything else must go to stderr.
"""

from __future__ import annotations

import json
from typing import BinaryIO

from .mcp_protocol import McpDispatcher, rpc_error


def _answer(dispatcher: McpDispatcher, message) -> object:
    if isinstance(message, list):  # a JSON-RPC batch
        answers = [answer for answer in (_answer(dispatcher, item) for item in message) if answer is not None]
        return answers or None
    if not isinstance(message, dict):
        return rpc_error(None, -32600, "Invalid request")
    return dispatcher.handle(message)


def serve_stdio(dispatcher: McpDispatcher, reader: BinaryIO, writer: BinaryIO) -> None:
    """Answer messages until the input closes."""
    for raw in reader:
        line = raw.strip()
        if not line:
            continue
        try:
            message = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            answer = rpc_error(None, -32700, "Parse error")
        else:
            answer = _answer(dispatcher, message)
        if answer is not None:
            writer.write(json.dumps(answer, ensure_ascii=False).encode("utf-8") + b"\n")
            writer.flush()


__all__ = ["serve_stdio"]
