"""Undo and redo of the Analyze set-up (no QWidget).

The set-up is what decides the reduction: the mode, the instrument profile chosen, αi, the beam
centre, masks and corrections, the GISAXS bands, the GIWAXS cuts and regions, the frames summed.
Its parts are immutable, so a ``Setup`` is a cheap snapshot. ``SetupHistory.observe`` is called
whenever the analysis is about to run again: a set-up that differs from the last one is one step
(“cut regions”, “mask” …). Changes of the same kind in quick succession — dragging a band, typing
a value — are one step. Undo returns the previous set-up, Redo the one undone.

The instrument profiles themselves (a calibration saved, a profile edited) are not part of it.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Optional

MAX_STEPS = 100
MERGE_SECONDS = 1.2
"""Changes of the same kind closer together than this are one step."""


@dataclass(frozen=True)
class Setup:
    mode: str
    profile_name: Optional[str]
    incidence_deg: Optional[float]
    beam_center: Optional[tuple]
    corrections: Any
    giwaxs: Any
    gisaxs: Any
    sum_count: int


def snapshot(view_model) -> Setup:
    state = view_model.state
    return Setup(
        mode=state.mode, profile_name=state.profile_name, incidence_deg=state.incidence_deg,
        beam_center=state.beam_center, corrections=state.corrections, giwaxs=state.giwaxs, gisaxs=state.gisaxs,
        sum_count=int(view_model.sum_count),
    )


def restore(view_model, setup: Setup) -> None:
    state = view_model.state
    state.mode = setup.mode
    state.profile_name = setup.profile_name
    state.incidence_deg = setup.incidence_deg
    state.beam_center = setup.beam_center
    state.corrections = setup.corrections
    state.giwaxs = setup.giwaxs
    state.gisaxs = setup.gisaxs
    view_model.set_sum_count(setup.sum_count)


_CORRECTIONS = (
    (("mask_shapes", "mask_path"), "mask"),
    (("background_path", "background_frame", "background_scale"), "background"),
    (("mirror_fill",), "mirror filling"),
    (("minimum", "maximum", "gap_guard_px", "bad_pixels"), "pixel limits"),
    (("solid_angle", "polarization", "film_thickness_nm", "attenuation_length_um"), "intensity corrections"),
)
_GIWAXS = (
    (("regions",), "cut regions"),
    (("sector", "box"), "sector and q box"),
    (("chi_q_window",), "I(χ) ring"),
)


def _changed_fields(before, after) -> set[str]:
    if before is None or after is None or not hasattr(before, "__dataclass_fields__"):
        return set() if before == after else {"*"}
    return {item.name for item in fields(before) if getattr(before, item.name) != getattr(after, item.name)}


def _grouped(changed: set[str], groups, other: str) -> list[str]:
    names = []
    for keys, name in groups:
        if changed & set(keys):
            names.append(name)
    if changed - {key for keys, _name in groups for key in keys}:
        names.append(other)
    return names


def describe(before: Setup, after: Setup) -> str:
    """What changed from ``before`` to ``after``, in a few words (English; translated where shown)."""
    names: list[str] = []
    if before.mode != after.mode:
        names.append("mode")
    if before.profile_name != after.profile_name:
        names.append("instrument profile")
    if before.incidence_deg != after.incidence_deg:
        names.append("αi")
    if before.beam_center != after.beam_center:
        names.append("beam centre")
    if before.sum_count != after.sum_count:
        names.append("frames summed")
    names += _grouped(_changed_fields(before.corrections, after.corrections), _CORRECTIONS, "corrections")
    names += _grouped(_changed_fields(before.giwaxs, after.giwaxs), _GIWAXS, "GIWAXS cuts")
    if before.gisaxs != after.gisaxs:
        names.append("GISAXS cuts")
    if not names:
        return "set-up"
    return names[0] if len(names) == 1 else ("set-up" if len(names) > 2 else f"{names[0]} and {names[1]}")


class SetupHistory:
    """The steps of the set-up: ``observe`` after changes, ``undo`` / ``redo`` to move between them."""

    def __init__(self) -> None:
        self._undo: list[tuple[Setup, str]] = []
        self._redo: list[tuple[Setup, str]] = []
        self._current: Optional[Setup] = None
        self._last_time = float("-inf")
        self._last_what: Optional[str] = None

    @property
    def current(self) -> Optional[Setup]:
        return self._current

    def observe(self, setup: Setup, now: float) -> bool:
        """``True`` when ``setup`` is a new step (or continues the last one)."""
        if self._current is None:
            self._current = setup
            return False
        if setup == self._current:
            return False
        what = describe(self._current, setup)
        continuing = (
            self._undo and not self._redo and what == self._last_what and now - self._last_time < MERGE_SECONDS
            and setup != self._undo[-1][0]  # an edit that cancels the last one (add, then remove) is a step of its own
        )
        if not continuing:
            self._undo.append((self._current, what))
            del self._undo[:-MAX_STEPS]
        self._redo.clear()
        self._current, self._last_time, self._last_what = setup, now, what
        return True

    def undo(self) -> Optional[tuple[Setup, str]]:
        if not self._undo:
            return None
        previous, what = self._undo.pop()
        self._redo.append((self._current, what))
        self._current, self._last_what = previous, None
        return previous, what

    def redo(self) -> Optional[tuple[Setup, str]]:
        if not self._redo:
            return None
        following, what = self._redo.pop()
        self._undo.append((self._current, what))
        self._current, self._last_what = following, None
        return following, what

    def next_undo(self) -> Optional[str]:
        return self._undo[-1][1] if self._undo else None

    def next_redo(self) -> Optional[str]:
        return self._redo[-1][1] if self._redo else None


__all__ = ["MAX_STEPS", "MERGE_SECONDS", "Setup", "SetupHistory", "describe", "restore", "snapshot"]
