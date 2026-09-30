"""The ``AnalysisWorkbench`` of a run: the Analyze page's automation, reached through the GUI bridge.

It also asks the person at the screen before write actions (``GuiConfirmer``).
"""

from __future__ import annotations

from typing import Callable, Optional

from PyQt5.QtWidgets import QMessageBox

from ..application import CurveData
from .gui_bridge import GuiBridge


class GuiWorkbench:
    def __init__(self, automation, bridge: GuiBridge, *, cancelled: Callable[[], bool] = lambda: False):
        self._automation = automation
        self._bridge = bridge
        self._cancelled = cancelled

    def _call(self, fn, *args):
        return self._bridge.call(fn, *args, cancelled=self._cancelled)

    def _apply(self, start) -> dict:
        ok, message = self._bridge.call_async(
            lambda done: start(lambda ok, text: done((ok, text))), cancelled=self._cancelled
        )
        if not ok:
            raise RuntimeError(message or "The analysis failed.")
        return self.status()

    def status(self) -> dict:
        return self._call(self._automation.status)

    def set_mode(self, mode: str) -> dict:
        return self._apply(lambda done: self._automation.set_mode(mode, done))

    def set_incidence(self, degrees: Optional[float]) -> dict:
        return self._apply(lambda done: self._automation.set_incidence(degrees, done))

    def set_sector_widths(self, in_plane_deg: float, out_of_plane_deg: float) -> dict:
        return self._apply(lambda done: self._automation.set_sector_widths(in_plane_deg, out_of_plane_deg, done))

    def set_radial_bins(self, bins: Optional[int]) -> dict:
        return self._apply(lambda done: self._automation.set_radial_bins(bins, done))

    def set_cut_regions(self, regions) -> dict:
        return self._apply(lambda done: self._automation.set_cut_regions(regions, done))

    def set_custom_sector(self, chi, q_range) -> dict:
        return self._apply(lambda done: self._automation.set_custom_sector(chi, q_range, done))

    def set_q_box(self, q_parallel, qz) -> dict:
        return self._apply(lambda done: self._automation.set_q_box(q_parallel, qz, done))

    def set_chi_window(self, q_low: float, q_high: float) -> dict:
        return self._apply(lambda done: self._automation.set_chi_window(q_low, q_high, done))

    def set_valid_range(self, minimum: Optional[float], maximum: Optional[float]) -> dict:
        return self._apply(lambda done: self._automation.set_valid_range(minimum, maximum, done))

    def set_frame(self, frame_number: int, sum_count: int) -> dict:
        return self._apply(lambda done: self._automation.set_frame(frame_number, sum_count, done))

    def curve(self, key: str) -> Optional[CurveData]:
        data = self._call(self._automation.curve, key)
        return None if data is None else CurveData(**data)

    def show(self, view: Optional[str] = None, lower_profile: Optional[str] = None) -> None:
        self._call(self._automation.show, view, lower_profile)

    def export_curves(self) -> list[str]:
        return list(self._call(self._automation.export_current))

    def preview_png(self, max_size: int = 900) -> Optional[bytes]:
        return self._call(self._automation.preview_png, max_size)

    def use_geometry(self, values: dict, name: Optional[str], source: str) -> dict:
        return self._apply(lambda done: self._automation.use_geometry(values, name, source, done))

    def symmetry_center(self) -> dict:
        return self._call(self._automation.symmetry_center)

    def set_beam_center(self, x_px: float, y_px: float) -> dict:
        return self._apply(lambda done: self._automation.set_beam_center(x_px, y_px, done))

    def set_halves(self, side: str) -> dict:
        return self._apply(lambda done: self._automation.set_halves(side, done))

    def set_gisaxs_cuts(self, horizontal_row, horizontal_half_height, vertical_column, vertical_half_width) -> dict:
        return self._apply(lambda done: self._automation.set_gisaxs_cuts(
            horizontal_row, horizontal_half_height, vertical_column, vertical_half_width, done,
        ))


class GuiConfirmer:
    def __init__(self, bridge: GuiBridge, parent, *, cancelled: Callable[[], bool] = lambda: False):
        self._bridge = bridge
        self._parent = parent
        self._cancelled = cancelled

    def confirm(self, title: str, text: str) -> bool:
        def ask() -> bool:
            answer = QMessageBox.question(
                self._parent, title, f"{text}\n\nAllow Claude to do this?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            return answer == QMessageBox.Yes

        return bool(self._bridge.call(ask, cancelled=self._cancelled))


__all__ = ["GuiConfirmer", "GuiWorkbench"]
