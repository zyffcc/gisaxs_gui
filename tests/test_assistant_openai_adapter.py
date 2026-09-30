"""Other AI providers through the OpenAI-compatible adapter, against a fake server (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from src.gimap.features.assistant.application import (
    GOALS,
    PERMISSION_AUTO,
    RUN_COMPLETED,
    AnalysisGoals,
    LlmError,
    RunAssistantTask,
    provider,
)
from src.gimap.features.assistant.infrastructure import OpenAICompatibleLlm, ProviderKeyStore, openai_messages, openai_tools
from tests.assistant_fakes import FakeWorkbench, report

openai = pytest.importorskip("openai")


def chunk(delta=None, finish=None, usage=None, model="deepseek-chat") -> dict:
    body = {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": model, "choices": []}
    if delta is not None or finish is not None:
        body["choices"] = [{"index": 0, "delta": delta or {}, "finish_reason": finish}]
    if usage is not None:
        body["usage"] = usage
    return body


def sse(*events: dict) -> bytes:
    return b"".join(b"data: " + json.dumps(event).encode("utf-8") + b"\n\n" for event in events) + b"data: [DONE]\n\n"


def tool_call_events(name: str, arguments: dict, *, call_id="call_1", text="", reasoning="") -> list[dict]:
    raw = json.dumps(arguments)
    events = []
    if reasoning:
        events.append(chunk({"role": "assistant", "content": None, "reasoning_content": reasoning}))
    if text:
        events.append(chunk({"role": "assistant", "content": text}))
    events.append(chunk({"tool_calls": [{"index": 0, "id": call_id, "type": "function", "function": {"name": name, "arguments": ""}}]}))
    middle = len(raw) // 2  # the arguments arrive in pieces
    events.append(chunk({"tool_calls": [{"index": 0, "function": {"arguments": raw[:middle]}}]}))
    events.append(chunk({"tool_calls": [{"index": 0, "function": {"arguments": raw[middle:]}}]}))
    events.append(chunk({}, finish="tool_calls"))
    events.append(chunk(usage={"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150,
                               "prompt_tokens_details": {"cached_tokens": 100}}))
    return events


class FakeServer:
    """Answers /chat/completions from a script and records every request."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": [{"id": "deepseek-reasoner", "object": "model"},
                                                                       {"id": "deepseek-chat", "object": "model"}]})
        body = json.loads(request.content.decode("utf-8"))
        self.requests.append({"headers": dict(request.headers), "body": body})
        reply = self.replies.pop(0)
        if isinstance(reply, httpx.Response):
            return reply
        if body.get("stream"):
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse(*reply))
        return httpx.Response(200, json=reply)


def llm_for(server: FakeServer, key="deepseek", model="deepseek-chat") -> OpenAICompatibleLlm:
    client = openai.OpenAI(
        base_url="https://fake.example/v1", api_key="sk-test", max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(server)),
    )
    return OpenAICompatibleLlm(preset=provider(key), model=model, client=client)


def test_the_conversation_is_translated_to_chat_messages_and_tools() -> None:
    messages = [
        {"role": "user", "content": "Analyse the frame."},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Finding peaks."},
            {"type": "tool_use", "id": "t1", "name": "find_peaks", "input": {"curve": "radial"}},
            {"type": "tool_use", "id": "t2", "name": "get_status", "input": {}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": '{"peaks": []}'},
            {"type": "tool_result", "tool_use_id": "t2", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAA"}},
                {"type": "text", "text": "q map"},
            ], "is_error": True},
        ]},
    ]
    converted = openai_messages("You are GIMaP.", messages)
    assert converted[0] == {"role": "system", "content": "You are GIMaP."}
    assert converted[1] == {"role": "user", "content": "Analyse the frame."}
    assistant = converted[2]
    assert assistant["content"] == "Finding peaks." and [call["id"] for call in assistant["tool_calls"]] == ["t1", "t2"]
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"curve": "radial"}
    assert converted[3] == {"role": "tool", "tool_call_id": "t1", "content": '{"peaks": []}'}
    assert converted[4]["role"] == "tool" and "image omitted" in converted[4]["content"] and "q map" in converted[4]["content"]
    tools = openai_tools([{"name": "find_peaks", "description": "Peaks.", "input_schema": {"type": "object", "properties": {}}}])
    assert tools == [{"type": "function", "function": {"name": "find_peaks", "description": "Peaks.",
                                                       "parameters": {"type": "object", "properties": {}}}}]


def test_a_streamed_tool_call_with_reasoning_and_usage_becomes_one_turn() -> None:
    server = FakeServer([tool_call_events("find_peaks", {"curve": "radial"}, text="Looking at the peaks.", reasoning="The radial curve first.")])
    seen = []
    turn = llm_for(server).respond(system="S", tools=[{"name": "find_peaks", "input_schema": {"type": "object"}}],
                                   messages=[{"role": "user", "content": "go"}], progress=lambda kind, text: seen.append((kind, text)))
    assert turn.stop_reason == "tool_use" and turn.text == "Looking at the peaks."
    assert [(call.id, call.name, call.input) for call in turn.tool_calls] == [("call_1", "find_peaks", {"curve": "radial"})]
    assert turn.content[-1] == {"type": "tool_use", "id": "call_1", "name": "find_peaks", "input": {"curve": "radial"}}
    assert (turn.usage.input_tokens, turn.usage.cache_read_input_tokens, turn.usage.output_tokens) == (20, 100, 30)
    assert ("thinking", "The radial curve first.") in seen and ("tool", "find_peaks") in seen
    request = server.requests[0]["body"]
    assert request["stream"] and request["stream_options"] == {"include_usage": True}
    assert request["tools"][0]["function"]["name"] == "find_peaks" and request["messages"][0]["role"] == "system"


def test_provider_errors_become_clear_messages() -> None:
    def failing(status, message):
        return FakeServer([httpx.Response(status, json={"error": {"message": message, "type": "x"}})])

    with pytest.raises(LlmError, match="rejected the API key"):
        llm_for(failing(401, "bad key")).respond(system="", tools=[], messages=[{"role": "user", "content": "x"}])
    with pytest.raises(LlmError) as limited:
        llm_for(failing(429, "slow down")).respond(system="", tools=[], messages=[{"role": "user", "content": "x"}])
    assert limited.value.retryable
    with pytest.raises(LlmError, match="tool calling"):
        llm_for(failing(400, "this model does not support tools")).respond(system="", tools=[], messages=[{"role": "user", "content": "x"}])
    # An older server that does not know stream_options is asked again without it.
    older = FakeServer([httpx.Response(400, json={"error": {"message": "unknown field stream_options"}}),
                        [chunk({"role": "assistant", "content": "OK"}), chunk({}, finish="stop")]])
    turn = llm_for(older).respond(system="", tools=[], messages=[{"role": "user", "content": "x"}])
    assert turn.text == "OK" and "stream_options" not in older.requests[1]["body"]
    with pytest.raises(LlmError, match="Choose a model"):
        OpenAICompatibleLlm(preset=provider("deepseek"), model="", client=object())
    with pytest.raises(LlmError, match="No API key for DeepSeek"):
        OpenAICompatibleLlm(preset=provider("deepseek"), model="deepseek-chat", api_key=None)


def test_models_are_listed_and_the_connection_test_uses_a_tool() -> None:
    server = FakeServer([{"id": "r1", "object": "chat.completion", "created": 0, "model": "deepseek-chat",
                          "choices": [{"index": 0, "message": {"role": "assistant", "content": None, "tool_calls": [
                              {"id": "c", "type": "function", "function": {"name": "ping", "arguments": '{"reply": "ok"}'}}]},
                              "finish_reason": "tool_calls"}]}])
    llm = llm_for(server)
    assert llm.list_models() == ["deepseek-chat", "deepseek-reasoner"]
    assert llm.check() == "deepseek-chat" and server.requests[0]["body"]["tools"][0]["function"]["name"] == "ping"


def test_keys_are_kept_per_provider_and_environment_variables_work(tmp_path: Path) -> None:
    store = ProviderKeyStore(tmp_path, environ={"DASHSCOPE_API_KEY": "sk-env"})
    assert store.source("qwen", "DASHSCOPE_API_KEY") == "environment variable DASHSCOPE_API_KEY"
    assert store.load("qwen", "DASHSCOPE_API_KEY") == "sk-env" and store.load("deepseek", "DEEPSEEK_API_KEY") is None
    store.save("deepseek", "sk-saved ")
    store.save("qwen", "sk-qwen")
    assert store.load("deepseek", "DEEPSEEK_API_KEY") == "sk-saved" and store.load("qwen", "DASHSCOPE_API_KEY") == "sk-qwen"
    assert store.source("qwen", "DASHSCOPE_API_KEY") == "API key saved in GIMaP"
    assert store.delete("qwen") and not store.delete("qwen") and store.load("qwen", "") is None
    assert "sk-saved" not in store.source("deepseek")  # the key itself is never shown


def test_a_whole_run_with_another_provider() -> None:
    submit = report(("peaks", "done"), summary="Four peaks.")
    server = FakeServer([
        tool_call_events("find_peaks", {"curve": "radial"}),
        tool_call_events("submit_report", submit.input, call_id="call_2", text="Done."),
    ])
    outcome = RunAssistantTask(llm_for(server), FakeWorkbench())(
        AnalysisGoals(goals=tuple(GOALS), permission=PERMISSION_AUTO)
    )
    assert outcome.state == RUN_COMPLETED and outcome.results.report is not None
    assert [step.tool for step in outcome.steps] == ["get_status", "find_peaks", "submit_report"]
    second = server.requests[1]["body"]["messages"]
    assert [message["role"] for message in second][-2:] == ["assistant", "tool"]
    assert json.loads(second[-1]["content"])["curve"] == "radial"  # the tool result went back to the model
