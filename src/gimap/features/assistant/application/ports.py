"""What the assistant needs from the outside: a model, a workbench, a person, storage."""

from __future__ import annotations

from typing import Callable, Optional, Protocol, Sequence

from .models import AgentResult, CurveData, LlmTurn, StepRecord, ToolOutcome


class AssistantLlm(Protocol):
    """One model turn: the conversation so far in, the assistant's next message out.

    ``progress(kind, text)`` receives the turn while it is generated: ``thinking``
    and ``text`` deltas, and ``tool`` with the name of a tool call being written.
    When ``cancelled()`` turns true the turn is abandoned with ``LlmError``.
    """

    model: str

    def respond(
        self,
        *,
        system: str,
        tools: Sequence[dict],
        messages: Sequence[dict],
        cancelled: Optional[Callable[[], bool]] = None,
        progress: Optional[Callable[[str, str], None]] = None,
    ) -> LlmTurn: ...


class AnalysisWorkbench(Protocol):
    """The Analyze workspace, operated as a user would (every call updates the GUI).

    Setters re-analyse the frame and return the new ``status()``.  Curves come
    back with q in Å⁻¹ (never 2θ) and χ in degrees.
    """

    def status(self) -> dict: ...

    def set_mode(self, mode: str) -> dict: ...

    def set_incidence(self, degrees: Optional[float]) -> dict: ...

    def set_sector_widths(self, in_plane_deg: float, out_of_plane_deg: float) -> dict: ...

    def set_radial_bins(self, bins: Optional[int]) -> dict: ...

    def set_cut_regions(self, regions: list[dict]) -> dict:
        """Replace the GIWAXS cut regions (``giwaxs_tools.region_arguments`` format); returns the status."""
        ...

    def set_custom_sector(
        self, chi: Optional[tuple[float, float]], q_range: tuple[Optional[float], Optional[float]]
    ) -> dict: ...

    def set_q_box(
        self, q_parallel: Optional[tuple[float, float]], qz: Optional[tuple[float, float]]
    ) -> dict: ...

    def set_chi_window(self, q_low: float, q_high: float) -> dict: ...

    def set_valid_range(self, minimum: Optional[float], maximum: Optional[float]) -> dict: ...

    def set_frame(self, frame_number: int, sum_count: int) -> dict:
        """Frame ``frame_number`` (1-based) of a series, ``sum_count`` frames summed from it."""
        ...

    def curve(self, key: str) -> Optional[CurveData]: ...

    def show(self, view: Optional[str] = None, lower_profile: Optional[str] = None) -> None: ...

    def export_curves(self) -> list[str]: ...

    def preview_png(self, max_size: int = 900) -> Optional[bytes]: ...

    def symmetry_center(self) -> dict:
        """GISAXS: the symmetry axis of the horizontal cut (``x_px``, ``initial_x_px``, losses); nothing changes."""
        ...

    def set_beam_center(self, x_px: float, y_px: float) -> dict: ...

    def set_halves(self, side: str) -> dict: ...

    def set_gisaxs_cuts(
        self,
        horizontal_row: Optional[float],
        horizontal_half_height: Optional[float],
        vertical_column: Optional[float],
        vertical_half_width: Optional[float],
    ) -> dict:
        """All ``None``: back to the automatic cuts."""
        ...

    def use_geometry(self, values: dict, name: Optional[str], source: str) -> dict:
        """Save ``values`` as the instrument profile of this frame's detector and re-analyse.

        ``values``: distance_mm, wavelength_angstrom, beam_center_x_px and
        beam_center_y_px (canonical pixels), optional pixel_size_x_m,
        pixel_size_y_m and incidence_deg.  Returns the new ``status()``.
        """
        ...


class FileExplorer(Protocol):
    """Read-only look at the files around the frame (for calibration material)."""

    def related_files(self, frame_path: str) -> dict:
        """``{"entries": [{path, relative, size, mtime, relation}], "folders": [...], "truncated": bool}``."""
        ...

    def search(self, folder: str, name_contains: Optional[str] = None, max_depth: int = 4) -> dict:
        """Like ``related_files`` for one folder and its subfolders (optionally by name)."""
        ...

    def kind(self, path: str) -> Optional[str]:
        """``file``, ``folder`` or ``None``."""
        ...

    def read_text(self, path: str, max_bytes: int = 65536) -> str: ...

    def image_header(self, path: str) -> dict:
        """Shape, detector name and header values (pixel size, energy, distance, beam centre)."""
        ...


class GeometryCalibrator(Protocol):
    """GIMaP's geometry calibration on an image of a standard (provided by the Calibration feature).

    Results are dicts: standard, standard_name, energy_kev, wavelength_angstrom,
    distance_mm, beam_center_px ([x, y], canonical pixels), pixel_size_um,
    detector, shape, matched_rings, rms_residual_px, confidence, score,
    warnings, rotation_deg, source_image, alternatives.
    """

    def standards(self) -> dict[str, str]: ...

    def calibrate(
        self,
        path: str,
        *,
        standard: str,
        energy_kev: float,
        distance_mm: Optional[float] = None,
        pixel_size_m: Optional[float] = None,
        cancelled: Optional[Callable[[], bool]] = None,
    ) -> dict: ...

    def read_result(self, path: str) -> dict:
        """A calibration saved by GIMaP's Geometry Calibration (.json)."""
        ...


class CurveFitter(Protocol):
    """A numerical physical fit of one curve (|q| > 0, q in Å⁻¹): solutions, best first."""

    def __call__(
        self, q_inv_angstrom, intensity, sigma, *, components=(), distance_nm=None, report=None, cancelled=None,
    ) -> list[dict]:
        """``distance_nm``: an interparticle distance to start from as well (a peak or shoulder of the curve)."""
        ...


class Chooser(Protocol):
    """Asks the person at the screen to pick an option or type a value."""

    def choose(self, question: str, options: Sequence[dict], allow_text: bool) -> Optional[dict]:
        """``{"index": int | None, "text": str}``, or ``None`` when the person declines."""
        ...


class Confirmer(Protocol):
    """Asks the person at the screen before an action that writes or changes settings."""

    def confirm(self, title: str, text: str) -> bool: ...


class ResultStore(Protocol):
    """Writes the computed tables next to the data and keeps tool requests for later."""

    def write_tables(self, folder: str, stem: str, payload: dict) -> str: ...

    def append_feature_request(self, entry: dict) -> None: ...


class RunEvents(Protocol):
    """Progress of a run (called from the thread that runs it)."""

    def step_started(self, step: StepRecord) -> None: ...

    def step_finished(self, step: StepRecord) -> None: ...

    def model_text(self, text: str) -> None: ...

    def model_progress(self, kind: str, text: str) -> None: ...

    def notice(self, text: str) -> None: ...

    def usage(self, turn_usage, total_usage) -> None: ...


class AgentRuntime(Protocol):
    """A brain that runs its own tool loop (Claude Code); GIMaP only answers its tool calls.

    ``call_tool(name, arguments)`` runs one of ``tools`` and returns its
    ``ToolOutcome``.  After each finished turn ``follow_up()`` gives the next
    user message, or ``None`` to end the session.
    """

    model: str

    def run(
        self,
        *,
        system: str,
        tools: Sequence[dict],
        prompt: str,
        call_tool: Callable[[str, dict], ToolOutcome],
        follow_up: Callable[[], Optional[str]],
        cancelled: Callable[[], bool],
        events: "RunEvents",
        max_turns: int,
    ) -> AgentResult: ...


__all__ = [
    "CurveFitter",
    "AgentRuntime",
    "AnalysisWorkbench",
    "AssistantLlm",
    "Chooser",
    "Confirmer",
    "FileExplorer",
    "GeometryCalibrator",
    "ResultStore",
    "RunEvents",
]
