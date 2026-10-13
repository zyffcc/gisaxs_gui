"""The assistant's model: Claude through the official Anthropic Python SDK.

One ``respond`` call is one streamed Messages API request.  Streaming lets the
panel show a summary of Claude's reasoning while it works and lets Stop end a
turn early.  The request uses adaptive thinking with summarized display,
top-level prompt caching (the tools, the system prompt and the growing
conversation are reused between turns), eager input streaming for the tools
(the application validates every tool input before it runs) and, for the models
that support it, server-side refusal fallbacks (``fallbacks="default"``).  The
SDK is imported only when a run starts, so GIMaP works without it.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from ..application import DEFAULT_EFFORT, DEFAULT_MODEL, EFFORTS, LlmError, LlmTurn, LlmUsage, ToolCall

MAX_TOKENS = 32000
"""Per turn, thinking included; streaming keeps long turns within the SDK's time limits."""
JSON_RETRIES = 2
FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODELS = ("claude-opus-5", "claude-fable-5", "claude-fable-5-1")
ADAPTIVE_PREFIXES = (
    "claude-opus-5", "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6",
    "claude-sonnet-5", "claude-sonnet-4-6", "claude-fable-5",
)
SUMMARY_PREFIXES = ("claude-opus-5", "claude-opus-4-8", "claude-opus-4-7", "claude-sonnet-5", "claude-fable-5")
"""Models whose thinking text is omitted unless ``display: "summarized"`` is requested."""
PRICES_PER_MTOK = {"claude-opus-5": (5.0, 25.0), "claude-sonnet-5": (2.0, 10.0)}
"""Input / output USD per million tokens, for the cost estimate shown in the panel."""

Progress = Optional[Callable[[str, str], None]]


def estimated_cost(model: str, usage: LlmUsage) -> Optional[float]:
    """USD estimate: cache reads at 0.1×, cache writes at 1.25× the input price."""
    prices = PRICES_PER_MTOK.get(model)
    if prices is None:
        return None
    input_price, output_price = prices
    return (
        usage.input_tokens * input_price
        + usage.cache_read_input_tokens * input_price * 0.1
        + usage.cache_creation_input_tokens * input_price * 1.25
        + usage.output_tokens * output_price
    ) / 1e6


def _sdk():
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise LlmError(
            "The Anthropic SDK is not installed. Run: python -m pip install anthropic"
        ) from exc
    return anthropic


RETRYABLE_TYPES = ("overloaded_error", "api_error", "timeout_error", "rate_limit_error")


def _error_type(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    error = body.get("error") if isinstance(body, dict) else None
    return str(error.get("type", "")) if isinstance(error, dict) else ""


def api_error(exc: Exception, model: str) -> Optional[LlmError]:
    """The user-facing ``LlmError`` for an SDK exception, ``None`` for anything else."""
    try:
        import anthropic
    except ImportError:  # pragma: no cover - a client exists only with the SDK
        return None
    if isinstance(exc, anthropic.AuthenticationError):
        return LlmError(
            "The Claude API rejected the credentials. Set an API key in Settings ▸ Assistant, "
            "the ANTHROPIC_API_KEY environment variable, or run `ant auth login`."
        )
    if isinstance(exc, anthropic.PermissionDeniedError):
        return LlmError(f"This API key may not use {model}: {exc.message}")
    if isinstance(exc, anthropic.NotFoundError):
        return LlmError(f"The model '{model}' was not found.")
    if isinstance(exc, anthropic.RateLimitError):
        return LlmError("The Claude API is rate-limiting this key.", retryable=True)
    if isinstance(exc, anthropic.BadRequestError):
        return LlmError(f"The Claude API rejected the request: {exc.message}")
    if isinstance(exc, anthropic.APIStatusError):
        # Errors inside a stream arrive with HTTP 200; their type says what happened.
        retryable = exc.status_code >= 500 or _error_type(exc) in RETRYABLE_TYPES
        return LlmError(f"Claude API error ({_error_type(exc) or exc.status_code}): {exc.message}", retryable=retryable)
    if isinstance(exc, anthropic.APIConnectionError):
        return LlmError("Cannot reach the Claude API: check the network connection.", retryable=True)
    if isinstance(exc, anthropic.AnthropicError):
        return LlmError(f"Claude API: {exc}")
    return None


def _streamed_tool(tool: dict) -> dict:
    # Strict tools keep the server's schema guarantee; the others stream unbuffered.
    return tool if tool.get("strict") else {**tool, "eager_input_streaming": True}


class AnthropicAssistantLlm:
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        effort: str = DEFAULT_EFFORT,
        api_key: Optional[str] = None,
        max_tokens: int = MAX_TOKENS,
        client: Any = None,
    ):
        self.model = (model or DEFAULT_MODEL).strip()
        self.effort = effort if effort in EFFORTS else DEFAULT_EFFORT
        self.max_tokens = int(max_tokens)
        if client is None:
            anthropic = _sdk()
            try:
                client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
            except anthropic.AnthropicError as exc:
                raise LlmError(f"The Claude API client could not start: {exc}") from exc
        self._client = client

    def request_arguments(self, system: str, tools: Sequence[dict], messages: Sequence[dict]) -> dict:
        arguments: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "tools": [_streamed_tool(tool) for tool in tools],
            "messages": list(messages),
            "cache_control": {"type": "ephemeral"},
        }
        if self.model.startswith(ADAPTIVE_PREFIXES):
            thinking = {"type": "adaptive"}
            if self.model.startswith(SUMMARY_PREFIXES):
                thinking["display"] = "summarized"
            arguments["thinking"] = thinking
            arguments["output_config"] = {"effort": self.effort}
        if self.model in FALLBACK_MODELS:
            arguments["betas"] = [FALLBACK_BETA]
            arguments["fallbacks"] = "default"
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
        arguments = self.request_arguments(system, tools, messages)
        failures = 0
        while True:
            try:
                return self.parse(self._stream(arguments, cancelled, progress))
            except LlmError:
                raise
            except ValueError as exc:
                # A streamed tool input the SDK could not parse at all: there is no
                # tool_use block to answer, so the same request is sent again.
                failures += 1
                if failures > JSON_RETRIES:
                    raise LlmError(f"Claude's tool input could not be read: {exc}") from exc
                if progress is not None:
                    progress("retry", "A tool input arrived garbled; asking again.")
            except Exception as exc:
                error = api_error(exc, self.model)
                if error is None:
                    raise
                raise error from exc

    def _stream(self, arguments: dict, cancelled, progress: Progress):
        with self._client.beta.messages.stream(**arguments) as stream:
            for event in stream:
                if cancelled is not None and cancelled():
                    raise LlmError("Stopped by the user.")  # leaving the block closes the stream
                if progress is None:
                    continue
                kind = getattr(event, "type", "")
                if kind == "thinking":
                    progress("thinking", event.thinking)
                elif kind == "text":
                    progress("text", event.text)
                elif kind == "content_block_start" and getattr(event.content_block, "type", "") == "tool_use":
                    progress("tool", event.content_block.name)
            return stream.get_final_message()

    @staticmethod
    def parse(response: Any) -> LlmTurn:
        blocks = list(getattr(response, "content", None) or [])
        text = "\n".join(block.text for block in blocks if getattr(block, "type", "") == "text")
        calls = tuple(
            ToolCall(block.id, block.name, block.input)
            for block in blocks
            if getattr(block, "type", "") == "tool_use"
        )
        raw_usage = getattr(response, "usage", None)
        usage = LlmUsage(
            int(getattr(raw_usage, "input_tokens", 0) or 0),
            int(getattr(raw_usage, "output_tokens", 0) or 0),
            int(getattr(raw_usage, "cache_read_input_tokens", 0) or 0),
            int(getattr(raw_usage, "cache_creation_input_tokens", 0) or 0),
        )
        stop_reason = str(getattr(response, "stop_reason", "") or "")
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None) if stop_reason == "refusal" and details else None
        return LlmTurn(
            stop_reason=stop_reason,
            text=text,
            tool_calls=calls,
            content=tuple(block.to_dict(mode="json") for block in blocks),
            usage=usage,
            model=str(getattr(response, "model", "") or ""),
            refusal_category=category,
        )

    def check(self) -> str:
        """A cheap request (model lookup) to confirm credentials and model; returns the model name."""
        try:
            info = self._client.with_options(timeout=20.0, max_retries=0).models.retrieve(self.model)
        except Exception as exc:
            error = api_error(exc, self.model)
            if error is None:
                raise
            raise error from exc
        return str(getattr(info, "display_name", None) or self.model)


__all__ = [
    "AnthropicAssistantLlm",
    "DEFAULT_EFFORT",
    "DEFAULT_MODEL",
    "EFFORTS",
    "FALLBACK_BETA",
    "MAX_TOKENS",
    "api_error",
    "estimated_cost",
]
