"""Plot area of the panel: data, model, components (log-log) and the residual below, x axes linked.

make_plot(parent, intensity_label, residual_label) returns an object with
    widget                       the QWidget to put into a layout
    show_curve(q, y, sigma, q_unit)
    show_fit(curves, components=True)        curves = result['curves'] of the kit
pyqtgraph is used when installed (the host's own plotting library in GIMaP), else matplotlib. Colours of the
background and of the axes are taken from the palette of the parent widget, so light and dark themes both work;
nothing global is configured (pg.setConfigOption is not touched).
"""
from __future__ import annotations

import math

from PyQt5 import QtCore, QtGui

MODEL = '#2a78d6'
COMPONENTS = ('#eb6834', '#1baf7a', '#8b5cf6', '#c2410c')       # populations 1-4
NEUTRAL = '#8a8a86'                                            # background and low-q term
TITLES = {'background_B': 'background', 'low_q_term': 'low-q term'}


def unit_text(unit):
    return {'nm^-1': 'nm⁻¹', 'A^-1': 'Å⁻¹'}.get(unit, unit)


def _positive(x, y):
    xs, ys = [], []
    for a, b in zip(x, y):
        if a is not None and b is not None and a > 0 and b > 0 and math.isfinite(a) and math.isfinite(b):
            xs.append(a)
            ys.append(b)
    return xs, ys


def _component_style(names):
    out, k = {}, 0
    for name in names:
        if name in TITLES:
            out[name] = (NEUTRAL, TITLES[name])
        else:
            out[name] = (COMPONENTS[k % len(COMPONENTS)], name.replace('_', ' '))
            k += 1
    return out


class PyqtgraphPlot:
    def __init__(self, parent, intensity_label, residual_label):
        import pyqtgraph as pg
        self.pg = pg
        palette = parent.palette()
        base, text = palette.color(QtGui.QPalette.Base), palette.color(QtGui.QPalette.Text)
        self.text = text
        self.widget = pg.GraphicsLayoutWidget(parent)
        self.widget.setBackground(base)
        self.main = self.widget.addPlot(row=0, col=0)
        self.residual = self.widget.addPlot(row=1, col=0)
        self.widget.ci.layout.setRowStretchFactor(0, 3)
        self.widget.ci.layout.setRowStretchFactor(1, 1)
        for plot in (self.main, self.residual):
            for side in ('left', 'bottom'):
                axis = plot.getAxis(side)
                axis.setPen(pg.mkPen(text))
                axis.setTextPen(pg.mkPen(text))
            plot.showGrid(x=True, y=True, alpha=0.15)
        self.main.setLogMode(x=True, y=True)
        self.residual.setLogMode(x=True, y=False)
        self.residual.setXLink(self.main)
        self.main.getAxis('bottom').setStyle(showValues=False)     # the linked residual plot below carries the q labels
        self.main.setLabel('left', intensity_label)
        self.residual.setLabel('left', residual_label)
        self.legend = self.main.addLegend(offset=(-10, 10), labelTextColor=text)
        self.q_unit = 'nm^-1'

    def _label(self):
        self.residual.setLabel('bottom', f"q ({unit_text(self.q_unit)})")

    def show_curve(self, q, y, sigma, q_unit):
        self.q_unit = q_unit
        self.main.clear()
        self.residual.clear()
        self.legend.clear()
        x, v = _positive(q, y)
        self.main.plot(x, v, pen=None, symbol='o', symbolSize=3, symbolBrush=self.text, symbolPen=None, name='data')
        self._label()
        self.main.enableAutoRange()

    def show_fit(self, curves, components=True):
        pg = self.pg
        q = curves['q']
        self.show_curve(q, curves['intensity'], curves['sigma'], curves['q_unit'])
        if components:
            for name, (color, title) in _component_style(curves['components']).items():
                x, v = _positive(q, curves['components'][name])
                if x and max(v) > 0:
                    self.main.plot(x, v, pen=pg.mkPen(color, width=1.2, style=QtCore.Qt.DashLine), name=title)
        x, v = _positive(q, curves['model'])
        self.main.plot(x, v, pen=pg.mkPen(MODEL, width=2), name='model')
        pairs = [(a, r) for a, r in zip(q, curves['residual']) if a > 0 and math.isfinite(r)]
        self.residual.plot([a for a, _ in pairs], [r for _, r in pairs], pen=pg.mkPen(MODEL, width=1))
        self.residual.addLine(y=0, pen=pg.mkPen(self.text, width=0.5))
        positive = [v for v in curves['intensity'] if v > 0]
        if positive:                                        # the data decide the visible range, not a vanishing component
            lo, hi = math.log10(min(positive)), math.log10(max(positive))
            self.main.setYRange(lo-0.3, hi+0.3, padding=0)


class MatplotlibPlot:
    def __init__(self, parent, intensity_label, residual_label):
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        palette = parent.palette()
        self.base = palette.color(QtGui.QPalette.Base).name()
        self.text = palette.color(QtGui.QPalette.Text).name()
        self.figure = Figure(figsize=(6, 5), facecolor=self.base)
        self.widget = FigureCanvasQTAgg(self.figure)
        self.widget.setParent(parent)
        grid = self.figure.add_gridspec(2, 1, height_ratios=(3, 1), hspace=0.05)
        self.main = self.figure.add_subplot(grid[0])
        self.residual = self.figure.add_subplot(grid[1], sharex=self.main)
        self.labels = (intensity_label, residual_label)
        self.q_unit = 'nm^-1'

    def _style(self):
        for ax in (self.main, self.residual):
            ax.set_facecolor(self.base)
            ax.tick_params(colors=self.text, which='both')
            for spine in ax.spines.values():
                spine.set_color(self.text)
            ax.grid(True, alpha=0.15)
        self.main.set_ylabel(self.labels[0], color=self.text)
        self.residual.set_ylabel(self.labels[1], color=self.text)
        self.residual.set_xlabel(f"q ({unit_text(self.q_unit)})", color=self.text)
        self.main.tick_params(labelbottom=False)

    def show_curve(self, q, y, sigma, q_unit, draw=True):
        self.q_unit = q_unit
        self.main.clear()
        self.residual.clear()
        x, v = _positive(q, y)
        self.main.loglog(x, v, '.', color=self.text, ms=2.5, label='data')
        self._style()
        if draw:
            self.widget.draw_idle()

    def show_fit(self, curves, components=True):
        q = curves['q']
        self.show_curve(q, curves['intensity'], curves['sigma'], curves['q_unit'], draw=False)
        if components:
            for name, (color, title) in _component_style(curves['components']).items():
                x, v = _positive(q, curves['components'][name])
                if x:
                    self.main.loglog(x, v, '--', color=color, lw=1.0, label=title)
        x, v = _positive(q, curves['model'])
        self.main.loglog(x, v, '-', color=MODEL, lw=1.8, label='model')
        positive = [v for v in curves['intensity'] if v > 0]
        if positive:
            self.main.set_ylim(min(positive)/2, max(positive)*2)
        self.residual.semilogx(q, curves['residual'], '-', color=MODEL, lw=0.8)
        self.residual.axhline(0, color=self.text, lw=0.5)
        self.main.legend(fontsize=8, frameon=False, labelcolor=self.text)
        self.widget.draw_idle()


def make_plot(parent, intensity_label='Intensity', residual_label='Residual / σ', backend=None):
    if backend in (None, 'pyqtgraph'):
        try:
            return PyqtgraphPlot(parent, intensity_label, residual_label)
        except ImportError:
            if backend == 'pyqtgraph':
                raise
    return MatplotlibPlot(parent, intensity_label, residual_label)
