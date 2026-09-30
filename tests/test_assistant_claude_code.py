"""Claude Code as the brain: the local MCP endpoint, the CLI runner and the run itself.

A stand-in CLI (``tests/fake_claude_code.py``) speaks Claude Code's stream-json
protocol and calls GIMaP's tools over MCP HTTP, so these tests use no account.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from src.gimap.features.assistant.application import (
    BILLING_SUBSCRIPTION,
    GOALS,
    RUN_CANCELLED,
    RUN_COMPLETED,
    RUN_FAILED,
    AnalysisGoals,
    LlmError,
    RunAgentTask,
    ToolOutcome,
    tool_specs,
)
from src.gimap.features.assistant.infrastructure import (
    ClaudeCodeAgent,
    LocalMcpServer,
    child_environment,
    find_claude_cli,
)
from tests.assistant_fakes import FakeWorkbench

FAKE_CLI = Path(__file__).with_name("fake_claude_code.py")
TOOLS = [spec.definition() for spec in tool_specs(allow_images=False)]


class Events:
    def __init__(self):
        self.log: list[tuple] = []

    def step_started(self, step):
        self.log.append(("started", step.tool))

    def step_finished(self, step):
        self.log.append(("finished", step.tool, step.ok))

    def model_text(self, text):
        self.log.append(("text", text))

    def model_progress(self, kind, text):
        self.log.append(("progress", kind, text))

    def notice(self, text):
        self.log.append(("notice", text))

    def usage(self, turn, total):
        self.log.append(("usage", total.output_tokens))


def _post(url: str, message, *, token: str | None, origin: str | None = None):
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    if origin is not None:
        headers["Origin"] = origin
    request = urllib.request.Request(url, data=json.dumps(message).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            body = response.read()
            return response.status, json.loads(body) if body else None
    except urllib.error.HTTPError as error:
        return error.code, None


def test_the_mcp_endpoint_serves_the_tools_to_its_client_only() -> None:
    calls = []

    def call_tool(name, arguments):
        calls.append((name, arguments))
        if name == "view_preview":
            return ToolOutcome([{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
                                {"type": "text", "text": "q map"}], "preview")
        if name == "get_curve":
            raise RuntimeError("no such curve")
        return ToolOutcome('{"peaks": []}', "no peaks", is_error=name == "compare_sectors")

    tools = [*TOOLS, *[spec.definition() for spec in tool_specs(allow_images=True) if spec.name == "view_preview"]]
    with LocalMcpServer(tools, call_tool) as server:
        url, token = server.url, server.token
        assert url.startswith("http://127.0.0.1:") and server.config()["headers"]["Authorization"] == f"Bearer {token}"
        status, reply = _post(url, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                    "params": {"protocolVersion": "2025-11-25", "capabilities": {}}}, token=token)
        assert status == 200 and reply["result"]["protocolVersion"] == "2025-11-25"
        assert reply["result"]["capabilities"] == {"tools": {"listChanged": False}}
        assert _post(url, {"jsonrpc": "2.0", "method": "notifications/initialized"}, token=token) == (202, None)
        listed = _post(url, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, token=token)[1]["result"]["tools"]
        assert [tool["name"] for tool in listed][:2] == ["get_status", "run_standard_pipeline"]
        report = next(tool for tool in listed if tool["name"] == "submit_report")
        assert "strict" not in report and report["inputSchema"]["required"] == ["summary", "items", "caveats", "suggestions"]

        def call(name, arguments=None):
            message = {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": name, "arguments": arguments or {}}}
            return _post(url, message, token=token)[1]["result"]

        assert call("find_peaks", {"curve": "radial"}) == {"content": [{"type": "text", "text": '{"peaks": []}'}], "isError": False}
        assert call("compare_sectors")["isError"] is True
        image, text = call("view_preview")["content"]
        assert image == {"type": "image", "data": "AAAA", "mimeType": "image/png"} and text["text"] == "q map"
        failed = call("get_curve", {"curve": "radial"})
        assert failed["isError"] and "no such curve" in failed["content"][0]["text"]
        assert call("make_coffee")["isError"]
        assert ("make_coffee", {}) not in calls
        unknown = _post(url, {"jsonrpc": "2.0", "id": 4, "method": "resources/list"}, token=token)[1]
        assert unknown["error"]["code"] == -32601
        # Only the run's own client gets in.
        assert _post(url, {"jsonrpc": "2.0", "id": 5, "method": "ping"}, token="wrong")[0] == 401
        assert _post(url, {"jsonrpc": "2.0", "id": 5, "method": "ping"}, token=None)[0] == 401
        assert _post(url, {"jsonrpc": "2.0", "id": 5, "method": "ping"}, token=token, origin="https://evil.example")[0] == 403
        assert _post(url, {"jsonrpc": "2.0", "id": 5, "method": "ping"}, token=token, origin="http://localhost:3000")[0] == 200
        with pytest.raises(urllib.error.HTTPError) as get:
            urllib.request.urlopen(urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"}), timeout=10)
        assert get.value.code == 405


def test_claude_code_gets_no_api_key_and_nothing_from_a_parent_session() -> None:
    environment = child_environment({
        "PATH": "C:/bin", "HTTPS_PROXY": "http://proxy:8080", "ANTHROPIC_API_KEY": "sk", "ANTHROPIC_AUTH_TOKEN": "t",
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:1", "CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "x",
        "CLAUDE_CODE_OAUTH_TOKEN_OF_PARENT": "t", "CLAUDE_CONFIG_DIR": "D:/claude", "CLAUDE_CODE_GIT_BASH_PATH": "C:/git/bash.exe",
    })
    assert sorted(environment) == [
        "CLAUDE_CODE_DISABLE_AUTO_MEMORY", "CLAUDE_CODE_GIT_BASH_PATH", "CLAUDE_CONFIG_DIR", "HTTPS_PROXY",
        "MCP_TOOL_TIMEOUT", "PATH",
    ]
    assert int(environment["MCP_TOOL_TIMEOUT"]) >= 600_000


@pytest.mark.skipif(not sys.platform.startswith("win"), reason="Windows install locations")
def test_claude_code_is_found_in_the_desktop_app_and_a_chosen_path_wins(tmp_path: Path) -> None:
    app_data = tmp_path / "Roaming"
    for version in ("2.1.9", "2.1.10"):
        path = app_data / "Claude" / "claude-code" / version / "claude.exe"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"")
    local = tmp_path / "Local"
    environ = {
        "APPDATA": str(app_data), "LOCALAPPDATA": str(local), "USERPROFILE": str(tmp_path), "PATH": str(tmp_path / "empty"),
    }
    assert find_claude_cli("", environ) == str(app_data / "Claude" / "claude-code" / "2.1.10" / "claude.exe")
    # Outside the Microsoft Store app the same Claude Code is in the package's LocalCache.
    packaged = local / "Packages" / "Claude_pzs8sxrjxfjjc" / "LocalCache" / "Roaming" / "Claude" / "claude-code"
    (packaged / "2.1.281").mkdir(parents=True)
    (packaged / "2.1.281" / "claude.exe").write_bytes(b"")
    assert find_claude_cli("", environ) == str(packaged / "2.1.281" / "claude.exe")
    for version in ("2.1.9", "2.1.10"):
        (app_data / "Claude" / "claude-code" / version / "claude.exe").unlink()
    assert find_claude_cli("", environ) == str(packaged / "2.1.281" / "claude.exe")
    native = tmp_path / ".local" / "bin" / "claude.exe"
    native.parent.mkdir(parents=True)
    native.write_bytes(b"")
    assert find_claude_cli("", environ) == str(native)
    chosen = tmp_path / "my" / "claude.exe"
    chosen.parent.mkdir()
    chosen.write_bytes(b"")
    assert find_claude_cli(str(chosen), environ) == str(chosen)
    assert find_claude_cli(str(tmp_path / "missing.exe"), environ) is None
    with pytest.raises(LlmError, match="not found"):
        ClaudeCodeAgent(cli=str(tmp_path / "missing.exe"), environ=environ)


def _agent(tmp_path: Path, scenario: str = "report", **options) -> tuple[ClaudeCodeAgent, Path]:
    record = tmp_path / f"record_{scenario}.json"
    environ = {
        "PATH": "", "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", ""),
        "FAKE_CLAUDE_SCENARIO": scenario, "FAKE_CLAUDE_RECORD": str(record),
        "ANTHROPIC_API_KEY": "sk-must-not-leak", "CLAUDECODE": "1",
    }
    return ClaudeCodeAgent(command=[sys.executable, str(FAKE_CLI)], environ=environ, **options), record


def test_claude_code_runs_the_gimap_tools_and_submits_the_report(tmp_path: Path) -> None:
    agent, record = _agent(tmp_path, model="opus", effort="high")
    workbench, events = FakeWorkbench(), Events()
    outcome = RunAgentTask(agent, workbench, events=events, max_turns=12)(AnalysisGoals(goals=tuple(GOALS)))

    assert outcome.state == RUN_COMPLETED, outcome.message
    assert [step.tool for step in outcome.steps] == [
        "get_status", "find_peaks", "compare_sectors", "ring_orientation", "crystallite_size", "find_peaks", "submit_report",
    ]
    assert [step.ok for step in outcome.steps] == [True, True, True, True, True, False, True]
    assert outcome.results.report.summary.startswith("Lamellar")
    assert min(abs(peak.q - 0.40) for peak in outcome.results.peak_searches["radial"].peaks) < 0.02
    assert outcome.results.rings and outcome.results.sizes
    assert ("set_chi_window", pytest.approx(outcome.results.rings[0].q_window[0]),
            pytest.approx(outcome.results.rings[0].q_window[1])) in workbench.calls
    assert outcome.billing == BILLING_SUBSCRIPTION and outcome.cost_usd == pytest.approx(0.42)
    assert outcome.usage.cache_read_input_tokens == 30000 and outcome.model == "claude-opus-5"
    notices = [entry[1] for entry in events.log if entry[0] == "notice"]
    assert notices[0] == "Claude Code · claude-opus-5 · Claude subscription"
    assert notices[1] == "Claude plan limit (five hour): 82% used"
    assert ("text", "Finding the peaks first.") in events.log
    assert ("progress", "tool", "find_peaks") in events.log and ("progress", "thinking", "Radial curve first.") in events.log
    assert any(message.get("type") == "result" for message in outcome.transcript)

    seen = json.loads(record.read_text("utf-8"))
    argv = seen["argv"]
    assert argv[:1] == ["-p"] and "--strict-mcp-config" in argv and "--setting-sources=" in argv
    assert "--disable-slash-commands" in argv and "--no-session-persistence" in argv
    assert argv[argv.index("--tools") + 1] == ""  # no shell, files or web
    assert argv[argv.index("--permission-mode") + 1] == "dontAsk"
    allowed = argv[argv.index("--allowedTools") + 1].split(",")
    assert "mcp__gimap__find_peaks" in allowed and "mcp__gimap__submit_report" in allowed and len(allowed) == len(TOOLS)
    assert argv[argv.index("--model") + 1] == "opus" and argv[argv.index("--effort") + 1] == "high"
    assert argv[argv.index("--max-turns") + 1] == "12"
    assert "mcp__gimap__" in seen["system_prompt"] and "GIWAXS" in seen["system_prompt"]
    assert "ANTHROPIC_API_KEY" not in seen["env"] and "CLAUDECODE" not in seen["env"]
    assert int(seen["mcp_tool_timeout"]) >= 600_000
    assert not Path(seen["cwd"]).exists()  # the run's temporary folder is removed


def test_a_missing_report_gets_one_reminder(tmp_path: Path) -> None:
    agent, _record = _agent(tmp_path, "late_report")
    outcome = RunAgentTask(agent, FakeWorkbench())(AnalysisGoals(goals=("peaks",)))
    assert outcome.state == RUN_COMPLETED and outcome.results.report is not None


def test_not_being_signed_in_is_reported_plainly(tmp_path: Path) -> None:
    agent, _record = _agent(tmp_path, "not_logged_in")
    outcome = RunAgentTask(agent, FakeWorkbench())(AnalysisGoals(goals=("peaks",)))
    assert outcome.state == RUN_FAILED and outcome.message == "Not logged in · Please run /login"


def test_stop_ends_claude_code_at_once(tmp_path: Path) -> None:
    agent, _record = _agent(tmp_path, "hang")
    events, stop = Events(), threading.Event()
    done: list = []
    runner = threading.Thread(
        target=lambda: done.append(RunAgentTask(agent, FakeWorkbench(), events=events)(
            AnalysisGoals(goals=("peaks",)), cancelled=stop.is_set)),
    )
    runner.start()
    deadline = time.monotonic() + 30
    while not any(entry[0] == "notice" for entry in events.log):
        assert time.monotonic() < deadline
        time.sleep(0.05)
    stop.set()
    runner.join(20)
    assert not runner.is_alive() and done[0].state == RUN_CANCELLED


def test_sign_in_state_is_read_without_asking_the_model(tmp_path: Path) -> None:
    agent, _record = _agent(tmp_path)
    info = agent.status()
    assert info["version"] == "9.9.9 (Claude Code)"
    assert info["logged_in"] and info["auth_method"] == "claude.ai" and info["subscription"] == "max"
    assert info["billing"] == BILLING_SUBSCRIPTION and not info["error"]
