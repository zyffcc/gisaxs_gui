"""What the assistant's presentation needs from outside, built by the feature bootstrap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from ..application import AgentRuntime, AssistantLlm, FileExplorer, CurveFitter, GeometryCalibrator, LlmUsage, ResultStore


@dataclass(frozen=True)
class ProviderServices:
    """Other AI providers (OpenAI-compatible): clients, keys, model lists."""

    create: Callable[[str, str, str], AssistantLlm]
    """``(provider key, model, address override)`` → a model client; raises ``LlmError``."""
    source: Callable[[str], str]
    """Where the provider's key comes from ("" when none)."""
    has_saved_key: Callable[[str], bool]
    save_key: Callable[[str, str], None]
    delete_key: Callable[[str], bool]
    list_models: Callable[[str, str, str], list]
    """``(provider key, model, address)`` → model names from the provider (network; off the GUI thread)."""
    check: Callable[[str, str, str], str]
    """``(provider key, model, address)`` → model name after a request with one tool (network)."""


@dataclass(frozen=True)
class AssistantServices:
    create_llm: Callable[[str, str], AssistantLlm]
    """``(model, effort)`` → a model client; raises ``LlmError`` (e.g. SDK missing)."""
    credentials: Callable[[], str]
    """Where the credentials come from, "" when none are configured."""
    has_saved_key: Callable[[], bool]
    save_key: Callable[[str], None]
    delete_key: Callable[[], bool]
    check: Callable[[str, str], str]
    """``(model, effort)`` → the model's display name after a test request; raises ``LlmError``."""
    store: ResultStore
    save_text: Callable[[str, str], str]
    """``(path, text)`` → the written path."""
    save_run: Callable[[dict], Optional[str]]
    """Keeps the record of a run (steps, results, transcript) in the user data folder."""
    cost: Callable[[str, LlmUsage], Optional[float]]
    create_agent: Callable[[str, str, str], AgentRuntime]
    """``(cli, model, effort)`` → Claude Code on the user's Claude plan; raises ``LlmError``."""
    find_cli: Callable[[str], Optional[str]]
    """The Claude Code executable for a configured path ("" = search the usual places)."""
    code_status: Callable[[str], dict]
    """Version and sign-in state of Claude Code (runs it; call off the GUI thread)."""
    code_login: Callable[[str], None]
    """Opens ``claude auth login`` for a Claude plan in its own window."""
    explorer: Optional[FileExplorer] = None
    """Read-only search for calibration files and logs around the frame."""
    calibrator: Optional[GeometryCalibrator] = None
    """GIMaP's geometry calibration (from the Calibration feature), when available."""
    fitter: Optional[CurveFitter] = None
    """A physical fit of a GISAXS cut (from the Fitting feature), when available."""
    providers: Optional[ProviderServices] = None
    """Other AI providers (OpenAI-compatible), when the ``openai`` SDK is available."""


__all__ = ["AssistantServices", "ProviderServices"]
