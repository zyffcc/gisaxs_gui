"""Shared theming of on-screen matplotlib figures (app/presentation/theme/figures.py)."""

from __future__ import annotations

import ast
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from PyQt5.QtCore import QCoreApplication, QEvent
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget

from src.gimap.app.presentation.theme import apply_theme, theme_color, theme_manager
from src.gimap.app.presentation.theme.figures import (
    apply_figure_colors,
    exported_colors,
    restyle_figure,
    theme_figure,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _app():
    return QApplication.instance() or QApplication([])


def _figure():
    figure = Figure(figsize=(3, 2))
    axes = figure.add_subplot(111)
    axes.plot([0, 1], [1, 2])
    axes.set_xlabel("q (Å⁻¹)")
    axes.set_title("curve")
    axes.text(0.5, 0.5, "empty state", transform=axes.transAxes)
    axes.text(0.1, 0.1, "chosen colour", color="red")
    return figure, axes


def test_apply_figure_colors_restyles_screen_parts_but_keeps_chosen_colours() -> None:
    figure, axes = _figure()
    apply_figure_colors(figure, background="#101010", foreground="#e0e0e0", grid="#303030")

    assert to_hex(figure.patch.get_facecolor()) == "#101010"
    assert to_hex(axes.get_facecolor()) == "#101010"
    assert to_hex(axes.xaxis.label.get_color()) == "#e0e0e0"
    assert to_hex(axes.title.get_color()) == "#e0e0e0"
    assert to_hex(axes.spines["left"].get_edgecolor()) == "#e0e0e0"
    tick = axes.xaxis.get_major_ticks()[0]
    assert to_hex(tick.label1.get_color()) == "#e0e0e0"
    assert to_hex(tick.gridline.get_color()) == "#303030"
    assert not tick.gridline.get_visible()  # the grid keeps its on/off state
    empty, chosen = axes.texts
    assert to_hex(empty.get_color()) == "#e0e0e0"
    assert to_hex(chosen.get_color()) == "#ff0000"
    # Themed text follows the next theme too.
    apply_figure_colors(figure, background="#ffffff", foreground="#334155", grid="#e2e8f0")
    assert to_hex(empty.get_color()) == "#334155"
    assert to_hex(chosen.get_color()) == "#ff0000"


def test_theme_figure_follows_switches_once_and_disconnects_with_the_canvas() -> None:
    import gc

    import pytest

    from src.gimap.app.presentation.theme import figures

    _app()
    manager = theme_manager()
    # Canvases of windows from earlier tests that are gone must not move the baseline: a theme
    # switch drops their connections. Automatic garbage collection is off while counting, so a
    # window of an earlier module kept alive by a reference cycle cannot go (and disconnect) here.
    gc.collect()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    apply_theme("light", 9)
    gc.disable()
    try:
        baseline = manager.receivers(manager.changed)
        host = QWidget()
        figure, axes = _figure()
        canvas = FigureCanvasQTAgg(figure)
        QVBoxLayout(host).addWidget(canvas)
        try:
            theme_figure(figure, canvas)
            slot = getattr(canvas, figures._SLOT_ATTRIBUTE)
            theme_figure(figure, canvas)  # re-colouring again does not connect twice
            assert getattr(canvas, figures._SLOT_ATTRIBUTE) is slot
            assert manager.receivers(manager.changed) == baseline + 1

            apply_theme("dark", 9)
            assert to_hex(figure.patch.get_facecolor()) == theme_color("plot_bg").name()
            axes.clear()
            axes.set_title("redrawn")
            restyle_figure(figure)  # after Axes.clear(), as the tool windows do
            assert to_hex(axes.title.get_color()) == theme_color("plot_fg").name()
            apply_theme("light", 9)
            assert to_hex(figure.patch.get_facecolor()) == theme_color("plot_bg").name()
        finally:
            apply_theme("light", 9)
        host.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    finally:
        gc.enable()
    # The canvas is gone, and so is its connection (disconnecting it again fails).
    with pytest.raises(TypeError):
        manager.changed.disconnect(slot)


def test_exported_colors_save_matplotlib_colours_and_restore_the_theme(tmp_path) -> None:
    from matplotlib import rcParams
    from matplotlib.image import imread

    _app()
    apply_theme("dark", 9)
    try:
        figure, axes = _figure()
        restyle_figure(figure)
        path = tmp_path / "figure.png"
        with exported_colors(figure):
            assert to_hex(axes.xaxis.label.get_color()) == to_hex(rcParams["text.color"])
            figure.savefig(path)
        corner = imread(path)[1, 1, :3]
        assert to_hex(tuple(corner)) == to_hex(rcParams["figure.facecolor"])
        # The screen copy is themed again.
        assert to_hex(figure.patch.get_facecolor()) == theme_color("plot_bg").name()
        assert to_hex(axes.xaxis.label.get_color()) == theme_color("plot_fg").name()
    finally:
        apply_theme("light", 9)


def test_figure_helper_needs_no_feature_imports() -> None:
    source = (PROJECT_ROOT / "src/gimap/app/presentation/theme/figures.py").read_text(
        encoding="utf-8"
    )
    modules = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
    assert not any("features" in module for module in modules)
