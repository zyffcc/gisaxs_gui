"""The assistant's model through any OpenAI-compatible chat API (the official ``openai`` SDK).

OpenAI, DeepSeek, Qwen, Kimi, GLM, Gemini's compatible endpoint, OpenRouter,
SiliconFlow, Azure OpenAI and local servers (Ollama, LM Studio, vLLM) all
answer ``chat.completions`` with tool calls.  The run loop keeps the
conversation in Anthropic's block format (text, tool_use, tool_result); this
adapter translates it to chat messages and back, streams the reply (text and,
where a provider sends it, reasoning) to the panel, and turns the SDK's errors
into ``LlmError``.  The SDK is imported only when a client is created.
"""

from __future__ import annotations

import json
import secrets
from typing import Any, Callable, Iterable, Optional, Sequence

from ..application import LlmError, LlmTurn, LlmUsage, ToolCall
from ..application.providers import AZURE_API_VERSION, KIND_AZURE, ProviderPreset

TIMEOUT_S = 300.0
JSON_RETRIES = 2
STOP_REASONS = {
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "stop": "end_turn",
    "length": "max_tokens",
    "content_filter": "refusal",
}
PING_TOOL = {
    "name": "ping",
    "description": "Answer the connection test.",
    "input_schema": {"type": "object", "properties": {"reply": {"type": "string"}}, "required": ["reply"]},
}
Progress = Optional[Callable[[str, str], None]]


def _sdk():
    try:
        import openai
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise LlmError("The OpenAI SDK is not installed. Run: python -m pip install openai") from exc
    return openai


# -- translation ----------------------------------------------------------------------------------


def openai_tools(tools: Sequence[dict]) -> list[dict]:
    """Tool definitions (name, description, input_schema) as chat-completions functions."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
            },
        }
        for tool in tools
    ]


def _text_of(content: Any) -> str:
    """User or tool content as plain text; images stay behind (these providers get numbers)."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content or ():
        kind = block.get("type") if isinstance(block, dict) else None
        if kind == "text":
            parts.append(str(block.get("text", "")))
        elif kind == "image":
            parts.append("[image omitted: this provider receives the numbers only]")
        else:
            parts.append(json.dumps(block, ensure_ascii=False, default=str))
    return "\n".join(parts)


def openai_messages(system: str, messages: Sequence[dict]) -> list[dict]:
    """The run loop's conversation (Anthropic blocks) as chat-completions messages."""
    converted: list[dict] = [{"role": "system", "content": system}] if system else []
    for message in messages:
        role, content = message.get("role"), message.get("content")
        if role == "assistant":
            blocks = content if isinstance(content, list) else [{"type": "text", "text": str(content or "")}]
            text = "".join(str(block.get("text", "")) for block in blocks if block.get("type") == "text")
            calls = [
                {
                    "id": block["id"],
                    "type": "function",
                    "function": {"name": block["name"], "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False)},
                }
                for block in blocks
                if block.get("type") == "tool_use"
            ]
            entry: dict = {"role": "assistant", "content": text}
            if calls:
                entry["tool_calls"] = calls
            converted.append(entry)
            continue
        if isinstance(content, list) and any(isinstance(block, dict) and block.get("type") == "tool_result" for block in content):
            for block in content:
                if block.get("type") == "tool_result":
                    converted.append({"role": "tool", "tool_call_id": block["tool_use_id"], "content": _text_of(block.get("content"))})
                else:
                    converted.append({"role": "user", "content": _text_of([block])})
            continue
        converted.append({"role": "user", "content": _text_of(content)})
    return converted


def _reasoning(delta: Any) -> str:
    """Reasoning text some providers stream next to the answer (DeepSeek, Qwen: ``reasoning_content``)."""
    value = getattr(delta, "reasoning_content", None)
    if value is None:
        extra = getattr(delta, "model_extra", None) or {}
        value = extra.get("reasoning_content") or extra.get("reasoning")
    return value if isinstance(value, str) else ""


def assemble(
    chunks: Iterable[Any],
    *,
    progress: Progress = None,
    cancelled: Optional[Callable[[], bool]] = None,
) -> LlmTurn:
    """One streamed reply as an ``LlmTurn`` with Anthropic-style content blocks."""
    text: list[str] = []
    calls: dict[int, dict] = {}
    finish, model, usage = "", "", None
    for chunk in chunks:
        if cancelled is not None and cancelled():
            raise LlmError("Stopped by the user.")
        model = getattr(chunk, "model", "") or model
        usage = getattr(chunk, "usage", None) or usage
        for choice in getattr(chunk, "choices", None) or ():
            delta = getattr(choice, "delta", None)
            if delta is not None:
                piece = getattr(delta, "content", None)
                if piece:
                    text.append(piece)
                    if progress is not None:
                        progress("text", piece)
                thought = _reasoning(delta)
                if thought and progress is not None:
                    progress("thinking", thought)
                for call in getattr(delta, "tool_calls", None) or ():
                    index = call.index if getattr(call, "index", None) is not None else len(calls)
                    slot = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                    if getattr(call, "id", None):
                        slot["id"] = call.id
                    function = getattr(call, "function", None)
                    if function is not None:
                        name = getattr(function, "name", None) or ""
                        if name:
                            if not slot["name"] and progress is not None:
                                progress("tool", name)
                            # Names arrive whole (sometimes repeated in every chunk) or, rarely, in pieces.
                            slot["name"] = name if not slot["name"] or name.startswith(slot["name"]) else slot["name"] + name
                        slot["arguments"] += getattr(function, "arguments", None) or ""
            if getattr(choice, "finish_reason", None):
                finish = choice.finish_reason
    blocks: list[dict] = []
    joined = "".join(text)
    if joined:
        blocks.append({"type": "text", "text": joined})
    tool_calls = []
    for index in sorted(calls):
        slot = calls[index]
        raw = slot["arguments"].strip()
        try:
            arguments = json.loads(raw) if raw else {}
        except ValueError as exc:
            raise ValueError(f"tool '{slot['name']}' arguments are not JSON: {raw[:120]}") from exc
        if not isinstance(arguments, dict):
            raise ValueError(f"tool '{slot['name']}' arguments are not a JSON object")
        ident = slot["id"] or f"call_{index}_{secrets.token_hex(4)}"
        tool_calls.append(ToolCall(ident, slot["name"], arguments))
        blocks.append({"type": "tool_use", "id": ident, "name": slot["name"], "input": arguments})
    stop = STOP_REASONS.get(finish, finish or "end_turn")
    if tool_calls and stop == "end_turn":
        stop = "tool_use"  # a few servers end tool turns with "stop"
    return LlmTurn(
        stop_reason=stop,
        text=joined,
        tool_calls=tuple(tool_calls),
        content=tuple(blocks),
        usage=_usage(usage),
        model=model,
        refusal_category=None,
    )


def _usage(raw: Any) -> LlmUsage:
    if raw is None:
        return LlmUsage()
    prompt = int(getattr(raw, "prompt_tokens", 0) or 0)
    details = getattr(raw, "prompt_tokens_details", None)
    cached = int(getattr(details, "cached_tokens", 0) or 0) if details is not None else 0
    extra = getattr(raw, "model_extra", None) or {}
    cached = cached or int(extra.get("prompt_cache_hit_tokens", 0) or 0)  # DeepSeek
    return LlmUsage(max(0, prompt - cached), int(getattr(raw, "completion_tokens", 0) or 0), cached, 0)


# -- errors ---------------------------------------------------------------------------------------


def openai_error(exc: Exception, name: str, model: str) -> Optional[LlmError]:
    """The user-facing ``LlmError`` for an OpenAI SDK exception, ``None`` for anything else."""
    try:
        import openai
    except ImportError:  # pragma: no cover
        return None
    detail = getattr(exc, "message", None) or str(exc)
    if isinstance(exc, openai.AuthenticationError):
        return LlmError(f"{name} rejected the API key. Check it in Settings ▸ Assistant.")
    if isinstance(exc, openai.PermissionDeniedError):
        return LlmError(f"{name}: this key may not use {model} ({detail}).")
    if isinstance(exc, openai.NotFoundError):
        return LlmError(f"{name}: the model '{model}' or the address was not found ({detail}).")
    if isinstance(exc, openai.RateLimitError):
        return LlmError(f"{name} is rate-limiting this key or the account has no balance ({detail}).", retryable=True)
    if isinstance(exc, (openai.BadRequestError, openai.UnprocessableEntityError)):
        lowered = detail.lower()
        if "tool" in lowered or "function" in lowered:
            return LlmError(
                f"{name} rejected the tool definitions: the model '{model}' may not support tool calling, "
                f"which GIMaP needs ({detail})."
            )
        return LlmError(f"{name} rejected the request: {detail}")
    if isinstance(exc, openai.APITimeoutError):
        return LlmError(f"{name} did not answer in time.", retryable=True)
    if isinstance(exc, openai.APIConnectionError):
        return LlmError(f"Cannot reach {name}: check the address and the network.", retryable=True)
    if isinstance(exc, openai.APIStatusError):
        return LlmError(f"{name} error ({exc.status_code}): {detail}", retryable=exc.status_code >= 500)
    if isinstance(exc, openai.OpenAIError):
        return LlmError(f"{name}: {detail}")
    return None


# -- the model ------------------------------------------------------------------------------------


class OpenAICompatibleLlm:
    def __init__(
        self,
        *,
        preset: ProviderPreset,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        max_tokens: Optional[int] = None,
        client: Any = None,
    ):
        self.preset = preset
        self.model = (model or "").strip()
        self.max_tokens = max_tokens
        self._stream_usage = True
        if not self.model:
            raise LlmError(f"Choose a model for {preset.name} in Settings ▸ Assistant.")
        url = (base_url or preset.base_url or "").strip()
        if client is None:
            if not url:
                raise LlmError(f"Enter the address of {preset.name} in Settings ▸ Assistant.")
            if preset.needs_key and not api_key:
                where = f" or set {preset.env_key}" if preset.env_key else ""
                raise LlmError(f"No API key for {preset.name}: save one in Settings ▸ Assistant{where}.")
            openai = _sdk()
            try:
                if preset.kind == KIND_AZURE:
                    client = openai.AzureOpenAI(
                        azure_endpoint=url, api_key=api_key, api_version=api_version or AZURE_API_VERSION,
                        timeout=TIMEOUT_S, max_retries=2,
                    )
                else:
                    client = openai.OpenAI(base_url=url, api_key=api_key or "not-needed", timeout=TIMEOUT_S, max_retries=2)
            except openai.OpenAIError as exc:
                raise LlmError(f"The {preset.name} client could not start: {exc}") from exc
        self._client = client

    def request_arguments(self, system: str, tools: Sequence[dict], messages: Sequence[dict]) -> dict:
        arguments: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages(system, messages),
            "stream": True,
        }
        if tools:
            arguments["tools"] = openai_tools(tools)
        if self._stream_usage:
            arguments["stream_options"] = {"include_usage": True}
        if self.max_tokens:
            arguments["max_tokens"] = int(self.max_tokens)
        return arguments

    def respond(
        self,
        *,
        system: str,
        tools: Sequence[dict],
        messages: Sequence[dict],
        cancelled: Optional[Callable[[], bool]] = None,
        progress: Progress = None,
    ) -> LlmTurn:
        failures = 0
        while True:
            arguments = self.request_arguments(system, tools, messages)
            try:
                stream = self._client.chat.completions.create(**arguments)
                try:
                    return assemble(stream, progress=progress, cancelled=cancelled)
                finally:
                    close = getattr(stream, "close", None)
                    if close is not None:
                        close()
            except LlmError:
                raise
            except ValueError as exc:
                # Tool arguments that are not JSON cannot be answered: ask again.
                failures += 1
                if failures > JSON_RETRIES:
                    raise LlmError(f"{self.preset.name}'s tool input could not be read: {exc}") from exc
                if progress is not None:
                    progress("retry", "A tool input arrived garbled; asking again.")
            except Exception as exc:
                error = openai_error(exc, self.preset.name, self.model)
                if error is None:
                    raise
                if self._stream_usage and "stream_options" in error.message:
                    self._stream_usage = False  # an older server: ask again without usage in the stream
                    continue
                raise error from exc

    def list_models(self) -> list[str]:
        """The provider's model names (``/models``)."""
        try:
            page = self._client.with_options(timeout=20.0, max_retries=0).models.list()
            return sorted({str(item.id) for item in page})
        except Exception as exc:
            error = openai_error(exc, self.preset.name, self.model)
            if error is None:
                raise
            raise error from exc

    def check(self) -> str:
        """A tiny request with one tool: the key, the model and tool calling all work; returns the model name."""
        try:
            response = self._client.with_options(timeout=60.0, max_retries=0).chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Call the ping tool with reply 'ok'."}],
                tools=openai_tools([PING_TOOL]),
            )
        except Exception as exc:
            error = openai_error(exc, self.preset.name, self.model)
            if error is None:
                raise
            raise error from exc
        return str(getattr(response, "model", "") or self.model)


__all__ = ["OpenAICompatibleLlm", "assemble", "openai_error", "openai_messages", "openai_tools"]
