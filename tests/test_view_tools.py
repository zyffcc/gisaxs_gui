"""Views that stay still, zoom into a rectangle, show one half of a curve, and point to Batch Export."""

from __future__ import annotations

import numpy as np

from tests.test_analyze_workspace import _app


def test_the_cursor_readout_never_resizes_the_view() -> None:
    from src.gimap.app.presentation.components import DetectorView

    _app()
    view = DetectorView()
    view.resize(400, 300)
    view.set_image(np.ones((50, 60)))
    before = view.minimumSizeHint().width()
    view._show_readout("frame 12 (a_very_long_file_name_of_an_in_situ_series_00012.nxs #12) · q = 1.2345 · I = 12.3" * 3)
    assert view.minimumSizeHint().width() == before  # the Series map jumped because this grew
    assert view.readout_label.text().endswith("…") and "q = 1.2345" in view.readout_label.toolTip()


def test_zoom_buttons_and_shift_drag_zoom_into_a_rectangle() -> None:
    import pyqtgraph as pg

    from src.gimap.app.presentation.components import CurvePlot, DetectorView

    _app()
    plot = CurvePlot("")
    box = plot.plot.getViewBox()
    plot.zoom_button.setChecked(True)
    assert box.state["mouseMode"] == pg.ViewBox.RectMode
    plot.zoom_button.setChecked(False)
    assert box.state["mouseMode"] == pg.ViewBox.PanMode
    assert pg.ViewBox.mouseDragEvent.__name__ == "mouse_drag"  # Shift + drag in every pyqtgraph view
    view = DetectorView()
    assert view.zoom_button.isCheckable()
    plot.set_curves([("I", np.linspace(0.1, 1, 50), np.linspace(1, 2, 50))])
    plot.plot.setXRange(0.2, 0.3)
    plot.reset_view()
    assert plot.plot.getViewBox().viewRange()[0][1] > 0.9
    plot.dispose()
    view.dispose()


def test_matplotlib_zoom_survives_a_redraw_and_resets_on_double_click() -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from src.gimap.app.presentation.components import MplBoxZoom

    figure = Figure()
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_subplot()
    axes.plot([0, 10], [0, 100])
    zoom = MplBoxZoom(canvas)
    canvas.draw()

    class Event:
        def __init__(self, x, y, *, dblclick=False):
            self.inaxes, self.button, self.dblclick = axes, 1, dblclick
            self.xdata, self.ydata = x, y
            self.x, self.y = axes.transData.transform((x, y))

    zoom._pressed(Event(2, 20))
    zoom._moved(Event(4, 40))
    zoom._released(Event(4, 40))
    assert zoom.zoomed and axes.get_xlim() == (2, 4)
    axes.clear()
    axes.plot([0, 10], [0, 100])  # the owner redraws its data
    canvas.draw()
    canvas.draw()  # the zoom is put back after the first draw
    assert tuple(np.round(axes.get_xlim(), 6)) == (2, 4)
    zoom._pressed(Event(3, 30, dblclick=True))
    assert not zoom.zoomed and axes.get_xlim()[1] >= 10


def test_a_signed_curve_can_show_one_half_or_both_folded() -> None:
    from src.gimap.app.presentation.components import CurvePlot

    _app()
    plot = CurvePlot("")
    plot.set_labels("qy (Å⁻¹)", "I")
    x = np.linspace(-1, 1, 41)
    plot.set_curves([("horizontal", x, np.exp(-x))])
    assert not plot.side_control.isHidden() and plot.side_control.button(3).text() == "|qy|"
    plot.set_side("positive")
    shown = plot.figure_state()["curves"]
    assert len(shown) == 1 and shown[0][1].min() > 0
    plot.set_side("folded")
    names = [name for name, _x, _y in plot.figure_state()["curves"]]
    assert names == ["horizontal (+)", "horizontal (−)"] and plot.figure_state()["curves"][1][1].min() > 0
    plot.set_curves([("I(q)", np.linspace(0.1, 1, 10), np.ones(10))])
    assert plot.side_control.isHidden()  # q ≥ 0: nothing to choose
    plot.dispose()


def test_the_two_images_are_separate_choices() -> None:
    from src.gimap.features.analyze.application import BatchChoices

    assert BatchChoices.from_dict({"images": True}).detector_image  # choices saved before the split
    choices = BatchChoices(q_map_image=True)
    assert choices.needs_map and not BatchChoices(detector_image=True).needs_map
