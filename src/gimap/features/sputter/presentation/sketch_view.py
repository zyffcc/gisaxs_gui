"""Structure Sketch tab: a real-space picture of one time window, and a storyboard over the series.

Side view (flat-based hemispheres on the polymer) and top view (arrangement and
coverage) on one nm ruler shared by the whole series, so growth is visible when
the time slider moves. Unreliable quantities (burial depth, TS metal fraction)
are drawn only when asked and are marked as such.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QCheckBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QSlider,
                             QVBoxLayout, QWidget)

from ..application import Sketcher, StructureResult, series_extent, smoothed_rows, storyboard_indices, window_sketch

METAL_COLOURS = {"Ag": ("#b8c0c8", "#5f6b76"), "Au": ("#e3b23c", "#8a6512"), "Cu": ("#d08a5c", "#7a4220")}
POLYMER = "#f3e3c3"; POLYMER_EDGE = "#c9a86a"; SUBSTRATE = "#9aa0a8"


def _nice_bar(width: float) -> float:
    for v in (1, 2, 5, 10, 20, 50, 100):
        if v >= width / 6:
            return float(v)
    return 100.0


def _describe(s) -> str:
    t = "–" if s.time_s is None else f"{s.time_s:.1f} s"
    if s.kind == "film":
        f = f"{100 * s.metal_fraction:.0f}%" if s.metal_fraction else "–"
        return (f"<b>t = {t}</b> · {s.stage or 'late'} · continuous film (TS model): characteristic length "
                f"<b>L = {s.L:.3g} nm</b> · metal fraction {f} <i>(conditional, unreliable)</i>")
    if s.kind != "islands":
        return f"<b>t = {t}</b> · {'; '.join(s.notes) or 'no size estimate in this window'}"
    err = f"±{100 * s.R_error:.0f}%" if s.R_error else ""
    spacing = f"{s.spacing:.3g} nm ({s.spacing_source})"
    cov = f" · coverage ≈ {100 * s.coverage:.0f}% <i>(model-derived)</i>" if s.coverage is not None else ""
    disorder = f"{s.disorder:.2f}" + ("" if s.disorder_measured else " (assumed)")
    text = (f"<b>t = {t}</b> · {s.stage or ''} · <b>R = {s.R:.3g} nm</b> {err} (grade {s.R_grade or '?'}) · "
            f"spacing <b>{spacing}</b> · size spread σ = {s.sigma:.2f} · spacing disorder {disorder}{cov}")
    if s.notes:
        text += "<br><i>" + "; ".join(s.notes) + "</i>"
    return text


def draw_window(ax_side, ax_top, s, width: float, height: float, metal: str, *, show_burial: bool,
                sketcher: Sketcher, fg: str = "black", compact: bool = False) -> None:
    """Draw one window into two axes (nm units)."""
    from matplotlib.patches import Circle, Rectangle, Wedge

    fill, edge = METAL_COLOURS.get(metal, METAL_COLOURS["Ag"])
    poly_depth = .55 * height
    for ax in (ax_side, ax_top):
        ax.clear(); ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    # -- side view -----------------------------------------------------------
    side_w = width * .5
    ax_side.add_patch(Rectangle((0, -poly_depth - .18 * height), side_w, .18 * height, color=SUBSTRATE, lw=0))
    if s.kind == "islands":
        base = -(s.burial or 0.) if (show_burial and s.burial) else 0.
        for p in sketcher.side_row(s, side_w):
            if p.x - p.r > side_w:
                continue
            ax_side.add_patch(Wedge((p.x, base), p.r, 0, 180, facecolor=fill, edgecolor=edge, lw=.8,
                                    ls="--" if show_burial and s.burial else "-", zorder=2))
        ax_side.add_patch(Rectangle((0, -poly_depth), side_w, poly_depth, facecolor=POLYMER, edgecolor=POLYMER_EDGE,
                                    lw=.6, alpha=.78 if show_burial and s.burial else 1., zorder=3 if show_burial else 1))
        if show_burial and s.burial:
            ax_side.text(side_w * .99, -poly_depth * .5, f"burial {s.burial:.2g} nm\n(unreliable)", ha="right", va="center",
                         fontsize=6.5, color="#7a4b00", zorder=4)
    elif s.kind == "film":
        ax_side.add_patch(Rectangle((0, -poly_depth), side_w, poly_depth, facecolor=POLYMER, edgecolor=POLYMER_EDGE, lw=.6))
        field = sketcher.film_field(s, width, n=160)
        if field:
            row = field[len(field) // 2]; n = len(row); dx = width / n; thick = .45 * height
            for i, metal_here in enumerate(row):
                x = i * dx
                if metal_here and x < side_w:
                    ax_side.add_patch(Rectangle((x, 0), dx, thick, facecolor=fill, edgecolor="none"))
    else:
        ax_side.add_patch(Rectangle((0, -poly_depth), side_w, poly_depth, facecolor=POLYMER, edgecolor=POLYMER_EDGE, lw=.6))
        ax_side.text(side_w / 2, height * .4, "no size estimate", ha="center", va="center", fontsize=8, color=fg)
    ax_side.set_xlim(0, side_w); ax_side.set_ylim(-poly_depth - .18 * height, height)
    ax_side.set_aspect("equal", adjustable="box")
    if not compact:  # the storyboard's top view carries the same ruler
        bar = _nice_bar(side_w)
        ax_side.plot([side_w * .03, side_w * .03 + bar], [height * .88] * 2, color=fg, lw=2)
        ax_side.text(side_w * .03 + bar / 2, height * .93, f"{bar:g} nm", ha="center", va="bottom", fontsize=6.5, color=fg)
        ax_side.set_title("side view (cut)", fontsize=8, color=fg, loc="left")
        ax_side.text(side_w * .99, -poly_depth * .9 - .18 * height * .5, "polymer (thickness not to scale)", ha="right",
                     va="center", fontsize=6, color="#5a4520")
    # -- top view ------------------------------------------------------------
    ax_top.add_patch(Rectangle((0, 0), width, width, facecolor=POLYMER, edgecolor=POLYMER_EDGE, lw=.6))
    if s.kind == "islands":
        unknown = s.spacing_source == "not measured"
        from matplotlib.collections import PatchCollection
        circles = [Circle((p.x, p.y), p.r) for p in sketcher.top_view(s, width)
                   if -p.r <= p.x <= width + p.r and -p.r <= p.y <= width + p.r]
        ax_top.add_collection(PatchCollection(circles, facecolor=fill, edgecolor=edge, linewidth=.4,
                                              linestyle=":" if unknown else "-", alpha=.55 if unknown else 1., zorder=2))
        if unknown:
            ax_top.text(width / 2, width / 2, "spacing not measured\n(illustration)", ha="center", va="center",
                        fontsize=7, color="#7a4b00", bbox=dict(facecolor="white", alpha=.7, lw=0))
    elif s.kind == "film":
        field = sketcher.film_field(s, width, n=160)
        if field:
            from matplotlib.colors import ListedColormap
            ax_top.imshow([[1 if v else 0 for v in row] for row in field], extent=(0, width, 0, width), origin="lower",
                          cmap=ListedColormap([POLYMER, fill]), vmin=0, vmax=1, interpolation="nearest", zorder=2)
    ax_top.set_xlim(0, width); ax_top.set_ylim(0, width); ax_top.set_aspect("equal", adjustable="box")
    bar = _nice_bar(width)
    ax_top.plot([width * .04, width * .04 + bar], [width * .05] * 2, color="black", lw=2.5, zorder=5)
    ax_top.text(width * .04 + bar / 2, width * .075, f"{bar:g} nm", ha="center", va="bottom", fontsize=6.5, color="black",
                bbox=dict(facecolor="white", alpha=.6, lw=0, pad=1), zorder=5)
    if not compact:
        ax_top.set_title("top view", fontsize=8, color=fg, loc="left")


class SketchView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.result: Optional[StructureResult] = None
        self.sketches: list = []; self.drawable: list = []; self.sketcher = Sketcher()
        self.width_nm = self.height_nm = 10.
        lay = QVBoxLayout(self); lay.setContentsMargins(4, 4, 4, 4)
        controls = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal); self.slider.setEnabled(False)
        self.time_label = QLabel("–"); self.time_label.setMinimumWidth(70)
        self.smooth = QCheckBox("Median of ±5 windows"); self.smooth.setChecked(True)
        self.smooth.setToolTip("Median over neighbouring windows of the same model (about ±2.5 s): hides single-window "
                               "jumps, keeps the trend. Untick to see each window's own values.")
        self.burial = QCheckBox("Show model burial depth (unreliable)")
        self.save_button = QPushButton("Save Image…"); self.story_button = QPushButton("Save Storyboard…")
        controls.addWidget(QLabel("Time")); controls.addWidget(self.slider, 1); controls.addWidget(self.time_label)
        controls.addWidget(self.smooth); controls.addWidget(self.burial)
        controls.addWidget(self.save_button); controls.addWidget(self.story_button)
        lay.addLayout(controls)
        self.info = QLabel("Open a result to see the structure."); self.info.setWordWrap(True)
        self.info.setTextFormat(Qt.RichText); lay.addWidget(self.info)
        self.canvas = None
        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            self.figure = Figure(figsize=(7, 6), dpi=100, constrained_layout=True)
            self.canvas = FigureCanvasQTAgg(self.figure)
            self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            lay.addWidget(self.canvas, 1)
        except Exception as exc:  # matplotlib missing
            lay.addWidget(QLabel(f"Drawing is unavailable: {exc}"))
        note = QLabel("Sketch of the fitted model, not an image: flat-based hemispheres (h = R) with the measured mean "
                      "size and size spread; spacing from the model D only where it agrees with the peak, otherwise "
                      "2π/q*. Burial depth and film details are not reliable.")
        note.setWordWrap(True); note.setStyleSheet("color: gray; font-size: 10px;")
        lay.addWidget(note)
        self.slider.valueChanged.connect(lambda _: self.redraw())
        self.burial.toggled.connect(lambda _: self.redraw())
        self.smooth.toggled.connect(lambda _: self._rebuild())
        self.save_button.clicked.connect(self._save_image)
        self.story_button.clicked.connect(self._save_storyboard)
        for b in (self.save_button, self.story_button):
            b.setEnabled(False)

    def set_result(self, result: StructureResult) -> None:
        self.result = result
        self._rebuild(keep_position=False)

    def _rebuild(self, keep_position: bool = True) -> None:
        if self.result is None:
            return
        position = self.slider.value() if keep_position else 0
        self.sketches = sketches_of(self.result, smooth=self.smooth.isChecked())
        self.drawable = [i for i, s in enumerate(self.sketches) if s.kind != "none"]
        self.width_nm, self.height_nm = series_extent(self.sketches)
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, len(self.drawable) - 1)); self.slider.setValue(min(position, self.slider.maximum()))
        self.slider.setEnabled(bool(self.drawable)); self.slider.blockSignals(False)
        for b in (self.save_button, self.story_button):
            b.setEnabled(bool(self.drawable))
        self.redraw()

    def current(self):
        if not self.drawable:
            return None
        return self.sketches[self.drawable[self.slider.value()]]

    def redraw(self) -> None:
        s = self.current()
        if s is None:
            self.info.setText("No window with a size estimate."); self.time_label.setText("–")
            if self.canvas is not None:
                self.figure.clear(); self.canvas.draw_idle()
            return
        self.time_label.setText("–" if s.time_s is None else f"{s.time_s:.1f} s")
        self.info.setText(_describe(s))
        if self.canvas is None:
            return
        fg = self.palette().text().color().name(); bg = self.palette().base().color().name()
        self.figure.clear(); self.figure.set_facecolor(bg)
        grid = self.figure.add_gridspec(2, 1, height_ratios=[1, 2.4])
        side, top = self.figure.add_subplot(grid[0]), self.figure.add_subplot(grid[1])
        draw_window(side, top, s, self.width_nm, self.height_nm, self.result.metal or "Ag", show_burial=self.burial.isChecked(),
                    sketcher=self.sketcher, fg=fg)
        self.canvas.draw_idle()

    def _save_image(self):
        if self.canvas is None or self.result is None:
            return
        s = self.current(); stamp = "" if s is None or s.time_s is None else f"_{s.time_s:.1f}s"
        path, _ = QFileDialog.getSaveFileName(self, "Save Sketch", str(self.result.folder / f"sketch{stamp}.png"),
                                              "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path:
            self.figure.savefig(path, dpi=200, facecolor="white")

    def _save_storyboard(self):
        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Storyboard", str(self.result.folder / "storyboard.png"),
                                              "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path:
            save_storyboard(self.result, Path(path), show_burial=self.burial.isChecked(), sketcher=self.sketcher,
                            smooth=self.smooth.isChecked())


def sketches_of(result: StructureResult, *, smooth: bool) -> list:
    rows = smoothed_rows(result.rows) if smooth else result.rows
    return [window_sketch(r) for r in rows]


def save_storyboard(result: StructureResult, path: Path, *, count: int = 6, show_burial: bool = False,
                    sketcher: Optional[Sketcher] = None, smooth: bool = True) -> list:
    """Side and top views of ``count`` windows on a logarithmic time grid, one ruler; returns the chosen row indices."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    sketches = sketches_of(result, smooth=smooth)
    picks = storyboard_indices(sketches, count)
    width, height = series_extent(sketches)
    sketcher = sketcher or Sketcher()
    n = max(1, len(picks))
    fig = Figure(figsize=(2.3 * n, 3.6), dpi=200, constrained_layout=True)
    FigureCanvasAgg(fig)
    grid = fig.add_gridspec(2, n, height_ratios=[1, 2.4])
    for k, i in enumerate(picks):
        s = sketches[i]
        side, top = fig.add_subplot(grid[0, k]), fig.add_subplot(grid[1, k])
        draw_window(side, top, s, width, height, result.metal or "Ag", show_burial=show_burial, sketcher=sketcher,
                    compact=True)
        if s.kind == "islands":
            spacing = "?" if s.spacing_source == "not measured" else f"{s.spacing:.3g}"
            label = f"t = {s.time_s:.3g} s · islands\nR = {s.R:.2g} nm ({s.R_grade}) · spacing {spacing} nm"
            if s.coverage is not None and s.coverage > .9:
                label += "\n(R and spacing inconsistent)"
        else:
            label = f"t = {s.time_s:.3g} s · film (TS)\nL = {s.L:.3g} nm"
        side.set_title(label, fontsize=6.5)
    fig.suptitle(f"{result.folder.name}: sketch of the fitted structure (hemispheres h = R; spacing from consistent D "
                 f"or 2π/q*; burial {'shown, unreliable' if show_burial else 'not shown'})", fontsize=7)
    fig.supxlabel("Model labels (islands / film) follow observation rules; a switch back is not a morphology reversal. "
                  "Spacing '?' = no resolved peak (illustration). Times on a logarithmic grid"
                  + ("; values are medians of ±5 neighbouring windows." if smooth else "; single-window values."),
                  fontsize=6, color="#555555")
    fig.savefig(str(path), dpi=200, facecolor="white")
    return picks


__all__ = ["SketchView", "draw_window", "save_storyboard"]
