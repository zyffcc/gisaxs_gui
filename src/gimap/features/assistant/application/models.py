"""Requests, turns, steps and results of one assistant run (framework neutral)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

DEFAULT_MODEL = "claude-opus-5"
MODEL_CHOICES = ("claude-opus-5", "claude-sonnet-5")
"""Offered in Settings; any other model id can be typed in."""
DEFAULT_EFFORT = "high"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
LANGUAGES = ("English", "中文")

BACKEND_CLAUDE_CODE = "claude_code"
BACKEND_API = "api"
BACKEND_PROVIDER = "provider"
"""Another AI provider through an OpenAI-compatible API (DeepSeek, Qwen, OpenAI, Ollama …)."""
BACKENDS = (BACKEND_CLAUDE_CODE, BACKEND_API, BACKEND_PROVIDER)
"""Who does the thinking: the local Claude Code (the user's Claude subscription) or the API."""
CLAUDE_CODE_MODELS = ("", "opus", "sonnet")
"""Claude Code model aliases offered in Settings; "" keeps Claude Code's own default."""
BILLING_SUBSCRIPTION = "subscription"
BILLING_API = "api"

PERMISSION_CONFIRM = "confirm"
PERMISSION_AUTO = "auto"
PERMISSION_PREVIEW = "preview"
"""Preview first: settings the model changes are restored at the end and offered as cards to apply."""
PERMISSION_MODES = (PERMISSION_CONFIRM, PERMISSION_AUTO, PERMISSION_PREVIEW)

GOAL_PEAKS = "peaks"
GOAL_ORIENTATION = "orientation"
GOAL_RING = "ring_orientation"
GOAL_SIZE = "crystallite_size"
GOAL_GISAXS_CUT = "gisaxs_cut"
GOAL_SPACING = "in_plane_spacing"
GOAL_GISAXS_FIT = "gisaxs_fit"
GOALS = {
    GOAL_PEAKS: "Peak table: q, d-spacing, FWHM and intensity of every peak (full, in-plane and out-of-plane curves)",
    GOAL_ORIENTATION: "Orientation: for each peak, in-plane versus out-of-plane intensity (face-on / edge-on / isotropic)",
    GOAL_RING: "Orientation distribution of one ring: I(χ), maxima and Herman's orientation factor",
    GOAL_SIZE: "Crystallite size (Scherrer coherence length) of the resolved peaks",
    GOAL_GISAXS_CUT: "GISAXS horizontal cut I(qy): at the Yoneda band, centred on the symmetry axis, halves chosen with the reason",
    GOAL_SPACING: "GISAXS in-plane distance: 2π/q* of a side maximum (or shoulder) of I(qy)",
    GOAL_GISAXS_FIT: "GISAXS fit of I(qy): particle family, radius, size spread and distance D, with the close alternatives",
}
"""The results a user can ask for, in dialog order."""
GIWAXS_GOALS = (GOAL_PEAKS, GOAL_ORIENTATION, GOAL_RING, GOAL_SIZE)
GISAXS_GOALS = (GOAL_GISAXS_CUT, GOAL_SPACING, GOAL_GISAXS_FIT)


def technique_of(goals) -> str:
    """``gisaxs`` when every requested result is a GISAXS one, else ``giwaxs``."""
    return "gisaxs" if goals and all(goal in GISAXS_GOALS for goal in goals) else "giwaxs"


@dataclass(frozen=True)
class AnalysisGoals:
    goals: tuple[str, ...]
    instructions: str = ""
    ring_q: Optional[float] = None
    """q (Å⁻¹) of the ring to analyse; ``None`` lets the assistant choose."""
    permission: str = PERMISSION_CONFIRM
    allow_images: bool = False
    language: str = "English"
    standing_instructions: str = ""
    """Rules the user keeps for every run (Settings ▸ Assistant)."""

    def __post_init__(self) -> None:
        unknown = [goal for goal in self.goals if goal not in GOALS]
        if unknown or not self.goals:
            raise ValueError(f"Choose at least one known result (unknown: {unknown})")
        if self.permission not in PERMISSION_MODES:
            raise ValueError(f"Unknown permission mode {self.permission!r}")


@dataclass(frozen=True)
class CurveData:
    """A reduced curve with x in Å⁻¹ (q-type) or degrees (χ)."""

    key: str
    title: str
    x: np.ndarray
    y: np.ndarray
    sigma: np.ndarray
    pixels: np.ndarray
    x_label: str
    region: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass(frozen=True)
class LlmUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def __add__(self, other: "LlmUsage") -> "LlmUsage":
        return LlmUsage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_input_tokens + other.cache_read_input_tokens,
            self.cache_creation_input_tokens + other.cache_creation_input_tokens,
        )


@dataclass(frozen=True)
class LlmTurn:
    stop_reason: str
    text: str
    tool_calls: tuple[ToolCall, ...]
    content: tuple[dict, ...]
    """The assistant message content as API blocks, echoed back unchanged next turn."""
    usage: LlmUsage = LlmUsage()
    model: str = ""
    refusal_category: Optional[str] = None


class ToolInputError(ValueError):
    """A tool call the model has to correct (reported to it, never raised further)."""


@dataclass(frozen=True)
class ToolOutcome:
    """What one tool call returned: the content the model sees and a one-line summary."""

    content: Any
    """A string, or a list of content blocks (text and image)."""
    summary: str
    is_error: bool = False
    final: bool = False
    data: Any = None


class LlmError(RuntimeError):
    """The model could not be reached or refused the request; ``message`` is for users."""

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.message = message
        self.retryable = retryable


@dataclass
class StepRecord:
    index: int
    tool: str
    arguments: dict
    summary: str = ""
    ok: bool = True
    seconds: float = 0.0
    result: Any = None
    """The JSON-able result the model saw."""


@dataclass(frozen=True)
class ReportItem:
    item: str
    status: str
    """``done``, ``partial`` or ``not_available``."""
    findings: str
    evidence: str
    reason: str


@dataclass(frozen=True)
class AssistantReport:
    summary: str
    items: tuple[ReportItem, ...]
    caveats: tuple[str, ...]
    suggestions: tuple[str, ...]


@dataclass
class RunResults:
    """Everything the tools computed; the report view shows these numbers, not retyped ones."""

    status: Optional[dict] = None
    peak_searches: dict = field(default_factory=dict)
    sector_rows: tuple = ()
    rings: list = field(default_factory=list)
    sizes: list = field(default_factory=list)
    exports: list = field(default_factory=list)
    feature_requests: list = field(default_factory=list)
    calibrations: list = field(default_factory=list)
    """Summaries of the calibrations fitted in this run (dicts, see ``GeometryCalibrator``)."""
    geometry_used: Optional[dict] = None
    """The geometry this run saved for the frame, with where it came from."""
    pipeline: Optional[dict] = None
    """The last standard-pipeline report (``run_standard_pipeline``), with its full tables."""
    operations: list = field(default_factory=list)
    """Settings changes the model made or proposed (``Operation``), for preview, apply and undo."""
    gisaxs: dict = field(default_factory=dict)
    """GISAXS findings: ``symmetry``, ``halves`` (a ``HalvesChoice``), ``spacing``, ``fit`` (solutions)."""
    report: Optional[AssistantReport] = None


@dataclass
class AgentResult:
    """How a run of an agent that owns its tool loop (Claude Code) ended."""

    ok: bool
    message: str
    usage: LlmUsage = LlmUsage()
    cost_usd: Optional[float] = None
    """The agent's own estimate at API prices (not billed on a subscription)."""
    model: str = ""
    billing: str = ""
    turns: int = 0
    transcript: list = field(default_factory=list)


RUN_COMPLETED = "completed"
RUN_INCOMPLETE = "incomplete"
RUN_CANCELLED = "cancelled"
RUN_FAILED = "failed"


@dataclass
class RunOutcome:
    state: str
    message: str
    steps: list[StepRecord]
    results: RunResults
    usage: LlmUsage
    model: str = ""
    transcript: list = field(default_factory=list)
    cost_usd: Optional[float] = None
    """Cost reported by the brain itself (Claude Code); ``None`` lets the caller estimate it."""
    billing: str = ""
    """``subscription`` (Claude plan quota) or ``api`` (billed per token)."""


__all__ = [
    "AgentResult",
    "AnalysisGoals",
    "AssistantReport",
    "BACKENDS",
    "BACKEND_API",
    "BACKEND_CLAUDE_CODE",
    "BACKEND_PROVIDER",
    "BILLING_API",
    "BILLING_SUBSCRIPTION",
    "CLAUDE_CODE_MODELS",
    "CurveData",
    "DEFAULT_EFFORT",
    "DEFAULT_MODEL",
    "EFFORTS",
    "LANGUAGES",
    "MODEL_CHOICES",
    "GOALS",
    "GISAXS_GOALS",
    "GIWAXS_GOALS",
    "GOAL_GISAXS_CUT",
    "GOAL_GISAXS_FIT",
    "GOAL_ORIENTATION",
    "GOAL_PEAKS",
    "GOAL_SPACING",
    "technique_of",
    "GOAL_RING",
    "GOAL_SIZE",
    "LlmError",
    "LlmTurn",
    "LlmUsage",
    "PERMISSION_AUTO",
    "PERMISSION_CONFIRM",
    "PERMISSION_MODES",
    "PERMISSION_PREVIEW",
    "ReportItem",
    "RUN_CANCELLED",
    "RUN_COMPLETED",
    "RUN_FAILED",
    "RUN_INCOMPLETE",
    "RunOutcome",
    "RunResults",
    "StepRecord",
    "ToolCall",
    "ToolInputError",
    "ToolOutcome",
]
