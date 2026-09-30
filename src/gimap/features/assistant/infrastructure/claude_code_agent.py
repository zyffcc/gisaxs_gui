"""Claude Code as the assistant's brain, on the user's Claude subscription.

GIMaP runs the local Claude Code CLI headless (``claude -p`` with stream-json
input and output) and gives it only the GIMaP tools, served for the run by
``LocalMcpServer``; every built-in tool (shell, files, web) is switched off and
nothing outside the run's temporary folder is configured.  API-key variables
are removed from its environment, so Claude Code uses the account the user
signed in with (``claude auth login``): with a Pro or Max plan the run counts
against that plan's usage limits instead of being billed per token.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

from ..application import (
    BILLING_API,
    BILLING_SUBSCRIPTION,
    EFFORTS,
    AgentResult,
    LlmError,
    LlmUsage,
)
from .mcp_http import SERVER_NAME, LocalMcpServer

TOOL_PREFIX = f"mcp__{SERVER_NAME}__"
KEEP_ENV = frozenset({"CLAUDE_CONFIG_DIR", "CLAUDE_CODE_GIT_BASH_PATH"})
DROP_ENV_PREFIXES = ("CLAUDE", "ANTHROPIC_")
MCP_TOOL_TIMEOUT_MS = 30 * 60 * 1000
"""A tool call can wait for the person to answer a confirmation."""
CHECK_TIMEOUT_S = 60.0
STOP_POLL_S = 0.2
EXIT_WAIT_S = 15.0
WINDOWS = sys.platform.startswith("win")
NO_WINDOW = subprocess.CREATE_NO_WINDOW if WINDOWS else 0
INSTALL_HINT = (
    "Claude Code was not found. Install it (claude.com/claude-code), or use Browse… in "
    "Settings ▸ Assistant to choose claude.exe (the Claude desktop app keeps one under "
    "%LOCALAPPDATA%\\Packages\\Claude_…\\LocalCache\\Roaming\\Claude\\claude-code)."
)


def child_environment(base: Mapping[str, str]) -> dict[str, str]:
    """Claude Code's environment: no API credentials and nothing from a parent Claude session."""
    environment = {
        key: value
        for key, value in base.items()
        if key.upper() in KEEP_ENV or not key.upper().startswith(DROP_ENV_PREFIXES)
    }
    environment["MCP_TOOL_TIMEOUT"] = str(MCP_TOOL_TIMEOUT_MS)
    environment["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"
    return environment


def _version_key(path: Path) -> tuple:
    parts = []
    for piece in path.parent.name.split("."):
        parts.append(int(piece) if piece.isdigit() else -1)
    return tuple(parts)


def cli_candidates(environ: Mapping[str, str] = os.environ) -> list[Path]:
    """Where Claude Code usually is: PATH, the native installer, the Claude desktop app, npm.

    The Claude desktop app from the Microsoft Store is an MSIX package: the
    Claude Code it keeps under ``%APPDATA%\\Claude\\claude-code`` is visible at that
    path only to the app itself; every other program (GIMaP) finds the same file
    in the package's ``LocalCache`` folder.
    """
    home = Path(environ.get("USERPROFILE") or environ.get("HOME") or Path.home())
    found = [Path(path) for path in (shutil.which("claude", path=environ.get("PATH")),) if path]
    if WINDOWS:
        app_data = Path(environ.get("APPDATA") or home / "AppData" / "Roaming")
        local_app_data = Path(environ.get("LOCALAPPDATA") or home / "AppData" / "Local")
        bundled = [
            *(app_data / "Claude" / "claude-code").glob("*/claude.exe"),
            *local_app_data.glob("Packages/Claude_*/LocalCache/Roaming/Claude/claude-code/*/claude.exe"),
        ]
        bundled.sort(key=_version_key, reverse=True)
        found = [path for path in found if path.suffix.lower() == ".exe"]
        found += [home / ".local" / "bin" / "claude.exe", *bundled, app_data / "npm" / "claude.cmd"]
    else:
        support = home / "Library" / "Application Support" / "Claude" / "claude-code"
        bundled = sorted(support.glob("*/claude"), key=_version_key, reverse=True)
        found += [
            home / ".local" / "bin" / "claude",
            home / ".claude" / "local" / "claude",
            *bundled,
            Path("/usr/local/bin/claude"),
            Path("/opt/homebrew/bin/claude"),
            home / ".npm-global" / "bin" / "claude",
        ]
    return found


def find_claude_cli(configured: str = "", environ: Mapping[str, str] = os.environ) -> Optional[str]:
    """The Claude Code executable to use: the configured one, or the first one found."""
    if configured.strip():
        path = Path(configured.strip())
        return str(path) if path.is_file() else None
    for path in cli_candidates(environ):
        if path.is_file():
            return str(path)
    return None


def _usage(raw) -> LlmUsage:
    raw = raw if isinstance(raw, dict) else {}
    return LlmUsage(
        int(raw.get("input_tokens") or 0),
        int(raw.get("output_tokens") or 0),
        int(raw.get("cache_read_input_tokens") or 0),
        int(raw.get("cache_creation_input_tokens") or 0),
    )


def _strip(name: str) -> str:
    return name[len(TOOL_PREFIX):] if name.startswith(TOOL_PREFIX) else name


def _limit_text(info: dict) -> str:
    window = str(info.get("rateLimitType") or "usage").replace("_", " ")
    text = f"Claude plan limit ({window})"
    utilization = info.get("utilization")
    if isinstance(utilization, (int, float)):
        text += f": {utilization:.0%} used"
    if info.get("status") == "rejected":
        text += " — limit reached"
        resets = info.get("resetsAt")
        if isinstance(resets, (int, float)):
            text += f", resets {time.strftime('%H:%M', time.localtime(resets))}"
    return text


class _Session:
    """One Claude Code process: feeds it user messages and turns its output into events."""

    def __init__(self, process: subprocess.Popen, events, follow_up, cancelled):
        self.process = process
        self.events = events
        self.follow_up = follow_up
        self.cancelled = cancelled
        self.stderr: deque[str] = deque(maxlen=40)
        self.transcript: list = []
        self.model = ""
        self.billing = ""
        self.usage = LlmUsage()
        self.cost: Optional[float] = None
        self.turns = 0
        self.results = 0
        self.error = ""
        self.final_text = ""
        self.stopped = False
        self._limit: Optional[tuple] = None

    def run(self, prompt: str) -> AgentResult:
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        threading.Thread(target=self._watch_stop, daemon=True).start()
        self._send(prompt)
        for line in self.process.stdout:
            message = self._parse(line)
            if message is not None:
                self._handle(message)
        try:
            code = self.process.wait(EXIT_WAIT_S)
        except subprocess.TimeoutExpired:
            self.process.kill()
            code = self.process.wait()
        return self._result(code)

    # -- process I/O -------------------------------------------------------------------

    def _send(self, text: str) -> None:
        # ASCII-escaped JSON reads the same whatever encoding the other side assumes.
        line = json.dumps(
            {"type": "user", "message": {"role": "user", "content": text}, "parent_tool_use_id": None, "session_id": "default"},
            ensure_ascii=True,
        )
        try:
            self.process.stdin.write(line + "\n")
            self.process.stdin.flush()
        except (OSError, ValueError):
            pass

    def _close_input(self) -> None:
        try:
            self.process.stdin.close()
        except (OSError, ValueError):
            pass

    def _drain_stderr(self) -> None:
        for line in self.process.stderr:
            if line.strip():
                self.stderr.append(line.rstrip())

    def _watch_stop(self) -> None:
        while self.process.poll() is None:
            if self.cancelled():
                self.stopped = True
                self.process.kill()
                return
            time.sleep(STOP_POLL_S)

    @staticmethod
    def _parse(line: str) -> Optional[dict]:
        line = line.strip()
        if not line.startswith("{"):
            return None
        try:
            message = json.loads(line)
        except ValueError:
            return None
        return message if isinstance(message, dict) else None

    # -- messages ----------------------------------------------------------------------

    def _handle(self, message: dict) -> None:
        kind = message.get("type")
        if kind == "stream_event":
            self._stream_event(message.get("event") or {})
            return
        if kind in ("system", "assistant", "user", "result"):
            self.transcript.append(message)
        if kind == "system":
            self._system(message)
        elif kind == "assistant":
            for block in (message.get("message") or {}).get("content") or ():
                if isinstance(block, dict) and block.get("type") == "text" and str(block.get("text", "")).strip():
                    self.events.model_text(str(block["text"]).strip())
        elif kind == "rate_limit_event":
            info = message.get("rate_limit_info") or {}
            key = (info.get("status"), info.get("rateLimitType"), round(float(info.get("utilization") or 0.0), 1))
            if key != self._limit:
                self._limit = key
                self.events.notice(_limit_text(info))
        elif kind == "result":
            self._finished_turn(message)

    def _system(self, message: dict) -> None:
        subtype = message.get("subtype")
        if subtype == "init":
            self.model = str(message.get("model") or "")
            source = message.get("apiKeySource")
            # "none" is a Claude plan login; "/login managed key", ANTHROPIC_API_KEY and
            # apiKeyHelper are billed per token.
            self.billing = BILLING_SUBSCRIPTION if source in (None, "", "none") else BILLING_API
            servers = {str(item.get("name")): str(item.get("status")) for item in message.get("mcp_servers") or () if isinstance(item, dict)}
            state = servers.get(SERVER_NAME)
            if state not in (None, "connected", "pending"):
                self.events.notice(f"GIMaP tools did not connect to Claude Code ({state}).")
            account = "Claude subscription" if self.billing == BILLING_SUBSCRIPTION else f"API billing ({source})"
            self.events.notice(f"Claude Code · {self.model or 'default model'} · {account}")
        elif subtype == "api_retry":
            attempt, most = message.get("attempt"), message.get("max_retries")
            self.events.model_progress("retry", f"Claude is retrying ({attempt}/{most}): {message.get('error') or 'error'}")

    def _stream_event(self, event: dict) -> None:
        kind = event.get("type")
        if kind == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") == "text_delta":
                self.events.model_progress("text", str(delta.get("text", "")))
            elif delta.get("type") == "thinking_delta":
                self.events.model_progress("thinking", str(delta.get("thinking", "")))
        elif kind == "content_block_start":
            block = event.get("content_block") or {}
            if "tool_use" in str(block.get("type", "")):
                self.events.model_progress("tool", _strip(str(block.get("name", ""))))

    def _finished_turn(self, message: dict) -> None:
        self.results += 1
        turn = _usage(message.get("usage"))
        # Totals are cumulative over the session, so the last result counts.
        self.usage = turn
        cost = message.get("total_cost_usd")
        self.cost = float(cost) if isinstance(cost, (int, float)) else self.cost
        self.turns = int(message.get("num_turns") or self.turns)
        self.final_text = str(message.get("result") or "")
        subtype = str(message.get("subtype") or "")
        if message.get("is_error") or subtype not in ("", "success"):
            if subtype == "error_max_turns":
                self.error = "Claude Code stopped at the most turns allowed for a run (Settings ▸ Assistant)."
            else:
                errors = message.get("errors") or []
                self.error = self.final_text or "; ".join(str(item) for item in errors) or subtype or "Claude Code failed."
        self.events.usage(turn, turn)
        follow = None if self.error or self.cancelled() else self.follow_up()
        if follow:
            self._send(follow)
        else:
            self._close_input()

    def _result(self, code: int) -> AgentResult:
        common = {
            "usage": self.usage, "cost_usd": self.cost, "model": self.model, "billing": self.billing,
            "turns": self.turns, "transcript": self.transcript,
        }
        if self.stopped or self.cancelled():
            return AgentResult(False, "Stopped by the user.", **common)
        if self.error:
            return AgentResult(False, self.error, **common)
        if not self.results:
            detail = " ".join(list(self.stderr)[-3:]) or f"exit code {code}"
            return AgentResult(False, f"Claude Code ended without an answer ({detail}).", **common)
        return AgentResult(True, self.final_text, **common)


class ClaudeCodeAgent:
    def __init__(
        self,
        *,
        cli: Optional[str] = None,
        model: str = "",
        effort: str = "",
        command: Optional[Sequence[str]] = None,
        environ: Optional[Mapping[str, str]] = None,
    ):
        self._environ = dict(os.environ if environ is None else environ)
        if command is None:
            path = find_claude_cli(cli or "", self._environ)
            if path is None:
                raise LlmError(INSTALL_HINT if not cli else f"Claude Code was not found at {cli}.")
            command = [path]
        self.command = list(command)
        self.model = (model or "").strip()
        self.effort = effort if effort in EFFORTS else ""

    def environment(self) -> dict[str, str]:
        return child_environment(self._environ)

    def arguments(self, system_file: Path, tool_names: Sequence[str], mcp_config: Path, max_turns: int) -> list[str]:
        arguments = [
            *self.command,
            "-p",
            "--output-format", "stream-json",
            "--input-format", "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--system-prompt-file", str(system_file),
            "--tools", "",
            "--allowedTools", ",".join(tool_names),
            "--permission-mode", "dontAsk",
            "--mcp-config", str(mcp_config),
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--setting-sources=",
            "--no-session-persistence",
            "--max-turns", str(int(max_turns)),
        ]
        if self.model:
            arguments += ["--model", self.model]
        if self.effort:
            arguments += ["--effort", self.effort]
        return arguments

    def run(
        self,
        *,
        system: str,
        tools: Sequence[dict],
        prompt: str,
        call_tool,
        follow_up: Callable[[], Optional[str]],
        cancelled: Callable[[], bool],
        events,
        max_turns: int,
    ) -> AgentResult:
        names = [TOOL_PREFIX + tool["name"] for tool in tools]
        instructions = "Tools of GIMaP's Analyze workspace for the GIWAXS frame that is open in the GUI."
        with LocalMcpServer(tools, call_tool, instructions=instructions) as server, tempfile.TemporaryDirectory(
            prefix="gimap_claude_", ignore_cleanup_errors=True
        ) as folder:
            config = Path(folder) / "mcp.json"
            config.write_text(json.dumps({"mcpServers": {SERVER_NAME: server.config()}}), encoding="utf-8")
            system_file = Path(folder) / "system_prompt.md"
            system_file.write_text(system, encoding="utf-8")
            try:
                process = subprocess.Popen(
                    self.arguments(system_file, names, config, max_turns),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=folder,
                    env=self.environment(),
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    creationflags=NO_WINDOW,
                )
            except OSError as exc:
                return AgentResult(False, f"Claude Code could not be started: {exc}")
            return _Session(process, events, follow_up, cancelled).run(prompt)

    # -- account -----------------------------------------------------------------------

    def _run_quietly(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [*self.command, *arguments],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=CHECK_TIMEOUT_S,
            env=self.environment(),
            creationflags=NO_WINDOW,
        )

    def status(self) -> dict:
        """Version and sign-in state of Claude Code (no model request, no usage)."""
        info = {
            "cli": self.command[-1], "version": "", "logged_in": False, "auth_method": "",
            "subscription": "", "billing": "", "error": "",
        }
        try:
            version = self._run_quietly("--version")
            info["version"] = (version.stdout or "").strip().splitlines()[0] if version.stdout.strip() else ""
            status = self._run_quietly("auth", "status", "--json")
        except (OSError, subprocess.SubprocessError) as exc:
            info["error"] = f"Claude Code could not be run: {exc}"
            return info
        try:
            data = json.loads(status.stdout or "{}")
        except ValueError:
            info["error"] = (status.stderr or status.stdout or "Unreadable sign-in status.").strip()[:300]
            return info
        info["logged_in"] = bool(data.get("loggedIn"))
        info["auth_method"] = str(data.get("authMethod") or "")
        info["subscription"] = str(data.get("subscriptionType") or "")
        info["billing"] = BILLING_SUBSCRIPTION if info["auth_method"] == "claude.ai" else (BILLING_API if info["logged_in"] else "")
        return info

    def open_login(self) -> None:
        """Start ``claude auth login`` for a Claude subscription in its own console window."""
        arguments = [*self.command, "auth", "login", "--claudeai"]
        if WINDOWS:
            subprocess.Popen(arguments, env=self.environment(), creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
            subprocess.Popen(arguments, env=self.environment(), start_new_session=True)


__all__ = [
    "ClaudeCodeAgent",
    "INSTALL_HINT",
    "MCP_TOOL_TIMEOUT_MS",
    "TOOL_PREFIX",
    "child_environment",
    "cli_candidates",
    "find_claude_cli",
]
