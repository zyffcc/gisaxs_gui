"""A stand-in for the Claude Code CLI in tests: stream-json in and out, tools over MCP HTTP.

It records its command line and environment (``FAKE_CLAUDE_RECORD``) and plays
one of these scenarios (``FAKE_CLAUDE_SCENARIO``):

- ``report``: the analysis tool calls, then ``submit_report``;
- ``late_report``: ends the first turn without a report, reports after the reminder;
- ``not_logged_in``: the error result Claude Code gives without a login;
- ``hang``: never answers (to test Stop).
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

PREFIX = "mcp__gimap__"


def emit(message: dict) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def option(arguments: list[str], name: str):
    for index, item in enumerate(arguments):
        if item == name and index + 1 < len(arguments):
            return arguments[index + 1]
        if item.startswith(name + "="):
            return item.split("=", 1)[1]
    return None


class McpClient:
    def __init__(self, config_path: str):
        with open(config_path, encoding="utf-8") as stream:
            server = json.load(stream)["mcpServers"]["gimap"]
        self.url = server["url"]
        self.headers = dict(server.get("headers") or {})
        self.next_id = 0

    def post(self, message: dict):
        request = urllib.request.Request(
            self.url,
            data=json.dumps(message).encode("utf-8"),
            headers={**self.headers, "Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            body = response.read()
        return json.loads(body) if body else None

    def request(self, method: str, params: dict) -> dict:
        self.next_id += 1
        return self.post({"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params})["result"]

    def connect(self) -> list[str]:
        self.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "fake", "version": "0"}})
        self.post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return [tool["name"] for tool in self.request("tools/list", {})["tools"]]


def use_tool(client: McpClient, name: str, arguments: dict, turn: list) -> dict:
    tool_id = f"toolu_{len(turn) + 1:03d}"
    emit({"type": "stream_event", "event": {"type": "content_block_start", "index": 0,
                                            "content_block": {"type": "tool_use", "id": tool_id, "name": PREFIX + name, "input": {}}}})
    emit({"type": "assistant", "message": {"role": "assistant", "model": "claude-opus-5", "content": [
        {"type": "tool_use", "id": tool_id, "name": PREFIX + name, "input": arguments}]}, "parent_tool_use_id": None})
    result = client.request("tools/call", {"name": name, "arguments": arguments})
    emit({"type": "user", "message": {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": tool_id, "content": result["content"], "is_error": result.get("isError", False)}]},
        "parent_tool_use_id": None})
    turn.append(name)
    return result


def report(client: McpClient, turn: list) -> None:
    use_tool(client, "submit_report", {
        "summary": "Lamellar peaks out of plane, π–π in plane.",
        "items": [{"item": "peaks", "status": "done", "findings": "Four peaks.", "evidence": "find_peaks", "reason": ""}],
        "caveats": [],
        "suggestions": [],
    }, turn)


def result(turns: int, text: str = "Done.", *, error: bool = False) -> None:
    emit({
        "type": "result", "subtype": "success", "is_error": error, "num_turns": turns, "session_id": "s1",
        "duration_ms": 10, "duration_api_ms": 5, "total_cost_usd": 0.42, "result": text,
        "usage": {"input_tokens": 1200, "output_tokens": 800, "cache_read_input_tokens": 30000, "cache_creation_input_tokens": 4000},
    })


def read_user_message():
    line = sys.stdin.readline()
    if not line:
        return None
    message = json.loads(line)
    assert message["type"] == "user" and message["message"]["role"] == "user", message
    return message["message"]["content"]


def main() -> int:
    sys.stdin.reconfigure(encoding="utf-8")  # as Claude Code does
    arguments = sys.argv[1:]
    if arguments[:1] == ["--version"]:
        print("9.9.9 (Claude Code)")
        return 0
    if arguments[:2] == ["auth", "status"]:
        print(json.dumps({"loggedIn": True, "authMethod": "claude.ai", "apiProvider": "firstParty", "subscriptionType": "max"}))
        return 0
    record = os.environ.get("FAKE_CLAUDE_RECORD")
    if record:
        prompt_file = option(arguments, "--system-prompt-file")
        with open(record, "w", encoding="utf-8") as stream:
            json.dump({
                "argv": arguments,
                "env": sorted(key for key in os.environ if key.upper().startswith(("ANTHROPIC", "CLAUDE", "MCP_"))),
                "mcp_tool_timeout": os.environ.get("MCP_TOOL_TIMEOUT"),
                "system_prompt": open(prompt_file, encoding="utf-8").read() if prompt_file else None,
                "cwd": os.getcwd(),
            }, stream)
    scenario = os.environ.get("FAKE_CLAUDE_SCENARIO", "report")
    prompt = read_user_message()
    if scenario == "not_logged_in":
        emit({"type": "system", "subtype": "init", "model": "claude-opus-5", "apiKeySource": "none", "mcp_servers": []})
        emit({"type": "result", "subtype": "success", "is_error": True, "num_turns": 0, "session_id": "s1",
              "result": "Not logged in · Please run /login", "total_cost_usd": 0, "usage": {}})
        read_user_message()
        return 1
    client = McpClient(option(arguments, "--mcp-config"))
    tools = client.connect()
    emit({"type": "system", "subtype": "init", "model": "claude-opus-5", "apiKeySource": "none",
          "tools": [PREFIX + name for name in tools], "mcp_servers": [{"name": "gimap", "status": "connected"}]})
    emit({"type": "rate_limit_event", "rate_limit_info": {"status": "allowed_warning", "rateLimitType": "five_hour", "utilization": 0.82}})
    if scenario == "hang":
        while True:
            time.sleep(0.1)
    assert "Requested results" in prompt
    turn: list = []
    emit({"type": "stream_event", "event": {"type": "content_block_delta", "index": 0,
                                            "delta": {"type": "thinking_delta", "thinking": "Radial curve first."}}})
    emit({"type": "assistant", "message": {"role": "assistant", "model": "claude-opus-5", "content": [
        {"type": "text", "text": "Finding the peaks first."}]}, "parent_tool_use_id": None})
    use_tool(client, "find_peaks", {"curve": "radial"}, turn)
    use_tool(client, "compare_sectors", {}, turn)
    use_tool(client, "ring_orientation", {"q_center": 0.4}, turn)
    use_tool(client, "crystallite_size", {"q_center": 1.2}, turn)
    use_tool(client, "find_peaks", {"curve": "nonsense"}, turn)  # invalid input comes back as an error
    if scenario == "late_report":
        result(len(turn), "Here is what I found.")
        reminder = read_user_message()
        if reminder is None:
            return 0
        assert "submit_report" in reminder
    report(client, turn)
    result(len(turn))
    while read_user_message() is not None:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
