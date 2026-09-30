"""The Claude adapter: request shape, streaming progress, stopping, retries and parsing."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.gimap.features.assistant.application import LlmError, LlmUsage, tool_specs
from src.gimap.features.assistant.infrastructure import (
    FALLBACK_BETA,
    AnthropicAssistantLlm,
    estimated_cost,
)
from src.gimap.features.assistant.infrastructure.anthropic_llm import api_error

TOOLS = [spec.definition() for spec in tool_specs(allow_images=False)]


class Block(SimpleNamespace):
    def to_dict(self, mode="python"):
        assert mode == "json"
        return dict(self.__dict__)


def _message(*blocks, stop_reason="tool_use", model="claude-opus-5", stop_details=None):
    usage = SimpleNamespace(input_tokens=12, output_tokens=34, cache_read_input_tokens=1000, cache_creation_input_tokens=50)
    return SimpleNamespace(content=list(blocks), usage=usage, stop_reason=stop_reason, model=model, stop_details=stop_details)


class FakeStream:
    def __init__(self, events, final):
        self.events = list(events)
        self.final = final
        self.closed = False
        self.consumed = 0

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.closed = True

    def __iter__(self):
        for event in self.events:
            if isinstance(event, Exception):
                raise event
            self.consumed += 1
            yield event

    def get_final_message(self):
        return self.final


class FakeClient:
    def __init__(self, *streams):
        self.streams = list(streams)
        self.requests: list[dict] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _stream(self, **arguments):
        self.requests.append(arguments)
        return self.streams.pop(0)


def _events():
    return [
        SimpleNamespace(type="message_start"),
        SimpleNamespace(type="content_block_start", content_block=SimpleNamespace(type="thinking")),
        SimpleNamespace(type="thinking", thinking="Checking the radial curve."),
        SimpleNamespace(type="text", text="Finding peaks."),
        SimpleNamespace(type="content_block_start", content_block=SimpleNamespace(type="tool_use", name="find_peaks")),
        SimpleNamespace(type="message_stop"),
    ]


def _final():
    return _message(
        Block(type="thinking", thinking="summary", signature="sig"),
        Block(type="text", text="Finding peaks."),
        Block(type="tool_use", id="toolu_1", name="find_peaks", input={"curve": "radial"}),
    )


def test_opus_requests_use_adaptive_summarised_thinking_caching_and_fallbacks() -> None:
    llm = AnthropicAssistantLlm(model="claude-opus-5", effort="high", client=FakeClient())
    arguments = llm.request_arguments("system", TOOLS, [{"role": "user", "content": "hi"}])
    assert arguments["model"] == "claude-opus-5"
    assert arguments["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert arguments["output_config"] == {"effort": "high"}
    assert arguments["cache_control"] == {"type": "ephemeral"}
    assert arguments["betas"] == [FALLBACK_BETA] and arguments["fallbacks"] == "default"
    tools = {tool["name"]: tool for tool in arguments["tools"]}
    assert tools["find_peaks"]["eager_input_streaming"] is True
    assert tools["submit_report"]["strict"] is True and "eager_input_streaming" not in tools["submit_report"]
    assert all("eager_input_streaming" not in tool for tool in TOOLS)  # the caller's list is untouched
    # The same conversation gives byte-identical arguments (prompt caching).
    assert arguments == llm.request_arguments("system", TOOLS, [{"role": "user", "content": "hi"}])


def test_other_models_get_only_what_they_support() -> None:
    sonnet = AnthropicAssistantLlm(model="claude-sonnet-5", effort="nonsense", client=FakeClient())
    arguments = sonnet.request_arguments("s", TOOLS, [])
    assert arguments["thinking"]["display"] == "summarized"
    assert arguments["output_config"] == {"effort": "high"}  # unknown effort → default
    assert "fallbacks" not in arguments and "betas" not in arguments
    haiku = AnthropicAssistantLlm(model="claude-haiku-4-5", client=FakeClient())
    arguments = haiku.request_arguments("s", TOOLS, [])
    assert "thinking" not in arguments and "output_config" not in arguments


def test_a_streamed_turn_reports_progress_and_parses_into_a_turn() -> None:
    stream = FakeStream(_events(), _final())
    llm = AnthropicAssistantLlm(client=FakeClient(stream))
    progress: list[tuple[str, str]] = []
    turn = llm.respond(system="s", tools=TOOLS, messages=[], progress=lambda kind, text: progress.append((kind, text)))
    assert progress == [
        ("thinking", "Checking the radial curve."),
        ("text", "Finding peaks."),
        ("tool", "find_peaks"),
    ]
    assert turn.stop_reason == "tool_use" and turn.text == "Finding peaks."
    assert [(item.id, item.name, item.input) for item in turn.tool_calls] == [("toolu_1", "find_peaks", {"curve": "radial"})]
    assert turn.content[0] == {"type": "thinking", "thinking": "summary", "signature": "sig"}
    assert turn.usage == LlmUsage(12, 34, 1000, 50)
    assert stream.closed


def test_refusals_carry_their_category() -> None:
    final = _message(stop_reason="refusal", stop_details=SimpleNamespace(category="cyber"))
    llm = AnthropicAssistantLlm(client=FakeClient(FakeStream([], final)))
    turn = llm.respond(system="s", tools=TOOLS, messages=[])
    assert turn.stop_reason == "refusal" and turn.refusal_category == "cyber" and not turn.tool_calls


def test_stopping_abandons_the_stream() -> None:
    stream = FakeStream(_events(), _final())
    llm = AnthropicAssistantLlm(client=FakeClient(stream))
    seen = []

    def cancelled():
        return len(seen) >= 1

    with pytest.raises(LlmError, match="Stopped"):
        llm.respond(system="s", tools=TOOLS, messages=[], cancelled=cancelled, progress=lambda *item: seen.append(item))
    assert stream.closed and stream.consumed < len(_events())


def test_an_unreadable_tool_input_is_requested_again_a_few_times() -> None:
    broken = [FakeStream([ValueError("bad json")], None) for _ in range(2)]
    client = FakeClient(*broken, FakeStream(_events(), _final()))
    llm = AnthropicAssistantLlm(client=client)
    progress = []
    turn = llm.respond(system="s", tools=TOOLS, messages=[], progress=lambda kind, text: progress.append(kind))
    assert turn.tool_calls and len(client.requests) == 3
    assert progress.count("retry") == 2

    client = FakeClient(*[FakeStream([ValueError("bad json")], None) for _ in range(3)])
    with pytest.raises(LlmError, match="could not be read"):
        AnthropicAssistantLlm(client=client).respond(system="s", tools=TOOLS, messages=[])


def test_cost_estimate_counts_cache_reads_and_writes() -> None:
    usage = LlmUsage(input_tokens=1_000_000, output_tokens=100_000, cache_read_input_tokens=1_000_000, cache_creation_input_tokens=0)
    assert estimated_cost("claude-opus-5", usage) == pytest.approx(5.0 + 0.5 + 2.5)
    assert estimated_cost("claude-sonnet-5", usage) == pytest.approx(2.0 + 0.2 + 1.0)
    assert estimated_cost("some-other-model", usage) is None


def test_sdk_errors_become_messages_for_the_user() -> None:
    anthropic = pytest.importorskip("anthropic")
    httpx = pytest.importorskip("httpx2")
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")

    def status(code, body=None):
        return httpx.Response(code, request=request), body

    response, body = status(401)
    error = api_error(anthropic.AuthenticationError("bad key", response=response, body=body), "claude-opus-5")
    assert "Settings" in error.message and not error.retryable
    response, body = status(529, {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})
    assert api_error(anthropic.OverloadedError("Overloaded", response=response, body=body), "m").retryable
    # An error event inside a stream arrives with HTTP 200.
    response, body = status(200, {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})
    mid_stream = api_error(anthropic.APIStatusError("Overloaded", response=response, body=body), "m")
    assert mid_stream.retryable and "overloaded_error" in mid_stream.message
    response, body = status(400)
    assert not api_error(anthropic.BadRequestError("bad", response=response, body=body), "m").retryable
    assert api_error(anthropic.APIConnectionError(request=request), "m").retryable
    assert api_error(RuntimeError("not from the SDK"), "m") is None

    stream = FakeStream([anthropic.APIConnectionError(request=request)], None)
    with pytest.raises(LlmError) as raised:
        AnthropicAssistantLlm(client=FakeClient(stream)).respond(system="s", tools=TOOLS, messages=[])
    assert raised.value.retryable


SSE_EVENTS = [
    ("message_start", {"type": "message_start", "message": {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5", "content": [],
        "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 1, "cache_read_input_tokens": 900, "cache_creation_input_tokens": 0},
    }}),
    ("content_block_start", {"type": "content_block_start", "index": 0,
                             "content_block": {"type": "thinking", "thinking": "", "signature": ""}}),
    ("content_block_delta", {"type": "content_block_delta", "index": 0,
                             "delta": {"type": "thinking_delta", "thinking": "Radial curve first."}}),
    ("content_block_delta", {"type": "content_block_delta", "index": 0,
                             "delta": {"type": "signature_delta", "signature": "c2ln"}}),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    ("content_block_start", {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}),
    ("content_block_delta", {"type": "content_block_delta", "index": 1,
                             "delta": {"type": "text_delta", "text": "Finding peaks."}}),
    ("content_block_stop", {"type": "content_block_stop", "index": 1}),
    ("content_block_start", {"type": "content_block_start", "index": 2, "content_block": {
        "type": "tool_use", "id": "toolu_1", "name": "find_peaks", "input": {}}}),
    ("content_block_delta", {"type": "content_block_delta", "index": 2,
                             "delta": {"type": "input_json_delta", "partial_json": "{\"curve\": \"rad"}}),
    ("content_block_delta", {"type": "content_block_delta", "index": 2,
                             "delta": {"type": "input_json_delta", "partial_json": "ial\"}"}}),
    ("content_block_stop", {"type": "content_block_stop", "index": 2}),
    ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                       "usage": {"output_tokens": 42}}),
    ("message_stop", {"type": "message_stop"}),
]


def test_the_real_sdk_stream_is_read_into_a_turn() -> None:
    anthropic = pytest.importorskip("anthropic")
    httpx = pytest.importorskip("httpx2")
    import json

    seen: dict = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["beta"] = request.headers.get("anthropic-beta")
        body = "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in SSE_EVENTS)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())

    client = anthropic.Anthropic(api_key="sk-test", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    llm = AnthropicAssistantLlm(model="claude-opus-5", client=client)
    progress = []
    turn = llm.respond(
        system="s", tools=TOOLS, messages=[{"role": "user", "content": "go"}],
        progress=lambda kind, text: progress.append((kind, text)),
    )

    body = seen["body"]
    assert seen["beta"] == FALLBACK_BETA and body["fallbacks"] == "default" and body["stream"] is True
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["cache_control"] == {"type": "ephemeral"} and body["output_config"] == {"effort": "high"}
    sent = {tool["name"]: tool for tool in body["tools"]}
    assert sent["find_peaks"]["eager_input_streaming"] is True and sent["submit_report"]["strict"] is True
    assert progress == [("thinking", "Radial curve first."), ("text", "Finding peaks."), ("tool", "find_peaks")]
    assert turn.stop_reason == "tool_use"
    assert [(item.name, item.input) for item in turn.tool_calls] == [("find_peaks", {"curve": "radial"})]
    assert turn.usage.cache_read_input_tokens == 900 and turn.usage.output_tokens == 42
    thinking, text, tool_use = turn.content
    assert thinking == {"type": "thinking", "thinking": "Radial curve first.", "signature": "c2ln"}
    assert text["type"] == "text" and text["text"] == "Finding peaks."
    assert tool_use["type"] == "tool_use" and tool_use["input"] == {"curve": "radial"}
    # The echoed blocks go back into the next request unchanged.
    json.dumps(list(turn.content))
