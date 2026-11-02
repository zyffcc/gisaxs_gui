"""Fit1DPanel: an embeddable PyQt5 widget for fitting one curve (and a small stand-alone program).

    python -m gisaxs1d_kit.qt_panel [curve file]            # stand-alone window

    from gisaxs1d_kit.qt_panel import Fit1DPanel            # inside a host GUI
    panel = Fit1DPanel(language='zh')                        # 'en' | 'zh'
    layout.addWidget(panel)
    panel.set_curve(q, intensity, sigma, q_unit='A^-1', label='cut 1', qz=0.5)   # hand over a cut
    panel.fitFinished.connect(on_result)                     # dict, schema gisaxs1d_gui_result_v1
    panel.fitFailed.connect(on_error); panel.fitCancelled.connect(on_cancel); panel.busyChanged.connect(on_busy)
    panel.shutdown()                                         # when the host closes (stops the worker process)

The GUI thread never fits: calls go through gisaxs1d_kit.client.Fit1DClient in a QThread, and the fit itself runs
in the client's worker process. Plots use pyqtgraph when it is installed, else matplotlib.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets

from gisaxs1d_kit.client import Fit1DCancelled, Fit1DClient, Fit1DError
from gisaxs1d_kit.i18n_zh import translate
from gisaxs1d_kit.qt_plot import make_plot

STATUS_COLORS = {'ambiguous': '#b45309', 'weak': '#b45309', 'at_bound': '#b45309', 'inactive': '#6b7280', 'fixed': '#2563eb'}
POLICY_TEXT = (('fast', 'fast: about 0.3 s'), ('auto', 'auto: multi-start search when needed (1-8 s)'),
               ('robust', 'robust: always multi-start (3-8 s)'))
MODE_TEXT = (('gisaxs', 'GISAXS (structure factor + low-q term)'), ('saxs', 'SAXS (dilute, no low-q term)'))
COMPOSITIONS = ('sphere,vc', 'sphere', 'vc', 'sphere,vc,vc', 'vc,vc', 'vc,vc,vc', 'sphere,sphere', 'auto:vc:3')


class _Call(QtCore.QThread):
    """One blocking client call in a thread."""
    done = QtCore.pyqtSignal(str, object)
    failed = QtCore.pyqtSignal(str, str)

    def __init__(self, client, kind, function, parent=None):
        super().__init__(parent)
        self.client, self.kind, self.function = client, kind, function

    def run(self):
        try:
            self.done.emit(self.kind, self.function())
        except Fit1DCancelled:
            self.failed.emit(self.kind, '')
        except Fit1DError as exc:
            self.failed.emit(self.kind, str(exc))
        except Exception as exc:                              # noqa: BLE001 - shown in the status line
            self.failed.emit(self.kind, f'{type(exc).__name__}: {exc}')


class Fit1DPanel(QtWidgets.QWidget):
    fitFinished = QtCore.pyqtSignal(dict)          # result of a fit or preview
    fitFailed = QtCore.pyqtSignal(str)
    fitCancelled = QtCore.pyqtSignal()
    busyChanged = QtCore.pyqtSignal(bool)

    def __init__(self, parent=None, *, language=None, client=None, python=None):
        super().__init__(parent)
        self.language = language or ('zh' if QtCore.QLocale.system().name().startswith('zh') else 'en')
        self.client = client or Fit1DClient(python=python)
        self._owns_client = client is None
        self.curve = None                          # dict(q, intensity, sigma, q_unit, label)
        self.result = None
        self._call = None
        self._threads = set()                      # every started thread until it has really finished
        self._started = 0.0
        self._build()
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self._tick)
        self._set_busy(False)
        self._status(self.tr_('Ready'))

    # ------------------------------------------------------------------ texts
    def tr_(self, text):
        return translate(text, self.language)

    # ------------------------------------------------------------------ layout
    def _build(self):
        t = self.tr_
        self.setWindowTitle(t('GISAXS / SAXS 1D fit'))
        self.open_button = QtWidgets.QPushButton(t('Open curve…'))
        self.file_unit = QtWidgets.QComboBox()
        self.file_unit.addItems(['nm^-1', 'A^-1'])
        self.file_unit.setToolTip(t('q unit of the file'))
        self.curve_label = QtWidgets.QLabel(t('No curve loaded'))
        self.curve_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.open_button)
        top.addWidget(self.file_unit)
        top.addWidget(self.curve_label, 1)

        self.composition = QtWidgets.QComboBox()
        self.composition.setEditable(True)
        self.composition.addItems(COMPOSITIONS)
        self.composition.setToolTip(t('Shapes: sphere, vc (vertical cylinder), rc (random cylinder), 1-4 populations, e.g. sphere,vc'))
        self.mode = QtWidgets.QComboBox()
        for key, text in MODE_TEXT:
            self.mode.addItem(t(text), key)
        self.qz = QtWidgets.QDoubleSpinBox()
        self.qz.setDecimals(4)
        self.qz.setRange(0.0, 50.0)
        self.qz.setSingleStep(0.01)
        self.qz.setToolTip(t('Out-of-plane momentum of the cut; spheres are evaluated at |q| = sqrt(q∥² + qz²). '
                             'Vertical cylinders do not use it.'))
        self.policy = QtWidgets.QComboBox()
        for key, text in POLICY_TEXT:
            self.policy.addItem(t(text), key)
        self.network = QtWidgets.QComboBox()
        self.network.addItem('general', 'general')
        self.floor = QtWidgets.QDoubleSpinBox()
        self.floor.setRange(0.0, 50.0)
        self.floor.setDecimals(1)
        self.floor.setSingleStep(1.0)
        self.floor.setToolTip(t('Relative systematic error added to the error bars; 2-5 % is sensible for bright measured curves.'))
        self.q_min = QtWidgets.QLineEdit()
        self.q_max = QtWidgets.QLineEdit()
        for edit, hint in ((self.q_min, 'min'), (self.q_max, 'max')):
            edit.setPlaceholderText(t(hint))
            edit.setValidator(QtGui.QDoubleValidator(0.0, 1e6, 6, edit))
            edit.setMaximumWidth(90)
        q_range = QtWidgets.QHBoxLayout()
        q_range.addWidget(self.q_min)
        q_range.addWidget(self.q_max)
        q_range.addStretch(1)
        form = QtWidgets.QFormLayout()
        form.addRow(t('Composition'), self.composition)
        form.addRow(t('Mode'), self.mode)
        form.addRow(t('qz (nm⁻¹)'), self.qz)
        form.addRow(t('Policy'), self.policy)
        form.addRow(t('Network'), self.network)
        form.addRow(t('Error floor (%)'), self.floor)
        form.addRow(t('q range'), q_range)
        box = QtWidgets.QGroupBox(t('Model'))
        box.setLayout(form)

        self.preview_button = QtWidgets.QPushButton(t('Preview'))
        self.fit_button = QtWidgets.QPushButton(t('Fit'))
        self.fit_button.setDefault(True)
        self.cancel_button = QtWidgets.QPushButton(t('Cancel'))
        self.save_button = QtWidgets.QPushButton(t('Save result…'))
        self.components_box = QtWidgets.QCheckBox(t('Show components'))
        self.components_box.setChecked(True)
        buttons = QtWidgets.QHBoxLayout()
        for w in (self.preview_button, self.fit_button, self.cancel_button, self.save_button):
            buttons.addWidget(w)
        self.status_label = QtWidgets.QLabel()
        self.status_label.setWordWrap(True)
        self.messages = QtWidgets.QPlainTextEdit()
        self.messages.setReadOnly(True)

        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels([t('Group'), t('Parameter'), t('Value'), t('Unit'),
                                              t('Range of equivalent solutions'), t('Status')])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)

        left = QtWidgets.QVBoxLayout()
        left.addLayout(top)
        left.addWidget(box)
        left.addLayout(buttons)
        left.addWidget(self.components_box)
        left.addWidget(self.status_label)
        left.addWidget(self.messages, 1)
        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(left)
        left_widget.setMinimumWidth(380)
        left_widget.setMaximumWidth(520)

        self.plot = make_plot(self, intensity_label=t('Intensity'), residual_label=t('Residual / σ'))
        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)          # plot above, parameter table below
        right.addWidget(self.plot.widget)
        right.addWidget(self.table)
        right.setStretchFactor(0, 3)
        right.setStretchFactor(1, 1)
        right.setSizes([520, 240])
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([440, 810])
        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.addWidget(splitter)

        self.open_button.clicked.connect(self.open_file_dialog)
        self.fit_button.clicked.connect(self.run_fit)
        self.preview_button.clicked.connect(self.run_preview)
        self.cancel_button.clicked.connect(self.cancel)
        self.save_button.clicked.connect(self.save_dialog)
        self.components_box.toggled.connect(self._redraw)
        self.mode.currentIndexChanged.connect(lambda _i: self.qz.setEnabled(self.mode.currentData() == 'gisaxs'))

    # ------------------------------------------------------------------ public API for a host GUI
    def set_curve(self, q, intensity, sigma=None, *, q_unit='nm^-1', label='', qz=None):
        """Hand a curve over (arrays). q_unit 'nm^-1' or 'A^-1'; qz in nm^-1 when known."""
        as_list = lambda a: None if a is None else [float(v) for v in a]
        self.curve = dict(q=as_list(q), intensity=as_list(intensity), sigma=as_list(sigma), q_unit=q_unit, label=str(label))
        if qz is not None:
            self.qz.setValue(float(qz))
        self.result = None
        n = len(self.curve['q'])
        self.curve_label.setText(f"{label or self.tr_('Curve')} · {n} {self.tr_('points')} · q [{q_unit}]")
        self._fill_table([])
        self.messages.setPlainText('')
        self.plot.show_curve(self.curve['q'], self.curve['intensity'], self.curve['sigma'], q_unit)
        self._set_busy(self.busy)

    def set_settings(self, **settings):
        """composition, mode, qz, policy, model, sigma_floor (fraction), q_range (lo, hi)."""
        if 'composition' in settings:
            self.composition.setCurrentText(str(settings['composition']))
        for combo, key in ((self.mode, 'mode'), (self.policy, 'policy'), (self.network, 'model')):
            if key in settings:
                index = combo.findData(settings[key])
                if index < 0 and combo is self.network:
                    combo.addItem(str(settings[key]), settings[key])
                    index = combo.count()-1
                if index >= 0:
                    combo.setCurrentIndex(index)
        if 'qz' in settings:
            self.qz.setValue(float(settings['qz'] or 0.0))
        if 'sigma_floor' in settings:
            self.floor.setValue(100.0*float(settings['sigma_floor'] or 0.0))
        if 'q_range' in settings:
            lo, hi = settings['q_range'] or ('', '')
            self.q_min.setText('' if lo in ('', None) else f'{float(lo):g}')
            self.q_max.setText('' if hi in ('', None) else f'{float(hi):g}')

    def settings(self):
        out = dict(composition=self.composition.currentText().strip(), mode=self.mode.currentData(), qz=float(self.qz.value()),
                   policy=self.policy.currentData(), model=self.network.currentData(), sigma_floor=float(self.floor.value())/100.0)
        lo, hi = self.q_min.text().strip(), self.q_max.text().strip()
        if lo or hi:
            out['q_range'] = [float(lo) if lo else 0.0, float(hi) if hi else 1e30]
        return out

    @property
    def busy(self):
        return self._call is not None

    def shutdown(self):
        """Stop a running call and the worker process (call it when the host window closes)."""
        self.client.cancel()
        for thread in list(self._threads):
            thread.wait(3000)
        if self._owns_client:
            self.client.close()

    # ------------------------------------------------------------------ actions
    def open_file_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, self.tr_('Open curve…'), '',
                                                        self.tr_('Curves (*.npz *.csv *.txt *.dat);;All files (*)'))
        if path:
            self.open_file(path)

    def open_file(self, path):
        unit = self.file_unit.currentText()
        self._start('load', lambda: self.client.load(path, q_unit=unit), self.tr_('Loading…'))

    def run_fit(self):
        self._run('fit')

    def run_preview(self):
        self._run('preview')

    def _run(self, kind):
        if self.curve is None or self.busy:
            return
        curve, settings = dict(self.curve), self.settings()
        function = self.client.fit if kind == 'fit' else self.client.preview
        self._start(kind, lambda: function(curve['q'], curve['intensity'], curve['sigma'], q_unit=curve['q_unit'], **settings),
                    self.tr_('Fitting…' if kind == 'fit' else 'Previewing…'))

    def cancel(self):
        if self.busy:
            self.client.cancel()

    def _start(self, kind, function, text):
        if self.busy:
            return
        call = _Call(self.client, kind, function, self)
        call.done.connect(self._on_done)
        call.failed.connect(self._on_failed)
        call.finished.connect(lambda c=call: self._threads.discard(c))
        self._threads.add(call)
        self._call = call
        self._started = time.perf_counter()
        self._busy_text = text if self.client.running else self.tr_('Starting the fitting process…')
        self._set_busy(True)
        self._timer.start()
        call.start()

    def _release(self):
        """The call is over for the panel (its thread may still be returning): a slot connected to fitFinished
        can start the next fit at once."""
        self._timer.stop()                              # the result text must not be overwritten by a late tick
        self._call = None
        self._set_busy(False)

    def _tick(self):
        if self.busy and self._timer.isActive():
            self._status(f'{self._busy_text} {time.perf_counter()-self._started:.1f} s')

    def _on_failed(self, kind, message):
        self._release()
        if message:
            self._status(f"{self.tr_('Failed')}: {message}", error=True)
            self.fitFailed.emit(message)
        else:
            self._status(self.tr_('Cancelled'))
            self.fitCancelled.emit()

    def _on_done(self, kind, value):
        self._release()
        if kind == 'load':
            self.set_curve(value['q'], value['intensity'], value['sigma'], q_unit=value['q_unit'], label=value.get('label', ''),
                           qz=value.get('qz'))
            self._status(self.tr_('Ready'))
            return
        self.result = value
        self.save_button.setEnabled(True)
        summary = value['summary']
        self._status(f"{self.tr_('Done')} · {self.tr_('chi² per point')} {summary['chi2_per_point']:.3g} · {summary['composition']} · "
                     f"{summary['policy']} · {summary['seconds']:.2f} s")
        lines = [self.tr_(m) for m in summary.get('messages') or []]
        self.messages.setPlainText('\n'.join(lines))
        self._fill_table(value['table'])
        self._redraw()
        self.fitFinished.emit(value)

    # ------------------------------------------------------------------ display
    def _redraw(self):
        if self.result is not None:
            self.plot.show_fit(self.result['curves'], components=self.components_box.isChecked())

    def _fill_table(self, rows):
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            rng = row.get('range')
            # a range narrower than 1 % is the scatter of one solution, not a second one: not shown
            rng_text = '' if not rng or abs(rng[1]-rng[0]) <= 1e-2*abs(row['value'] or 1.0) else f'{rng[0]:.4g} – {rng[1]:.4g}'
            cells = (row['group'], self.tr_(row['parameter']), row['text'], row['unit'], rng_text, self.tr_(row['status']))
            color = STATUS_COLORS.get(row['status'])
            for j, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                if j == 2:
                    item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                if color and j in (2, 4, 5):
                    item.setForeground(QtGui.QBrush(QtGui.QColor(color)))
                self.table.setItem(i, j, item)
        self.table.resizeColumnsToContents()

    def _status(self, text, error=False):
        self.status_label.setStyleSheet('color: #b91c1c;' if error else '')
        self.status_label.setText(text)

    def _set_busy(self, busy):
        ready = self.curve is not None and not busy
        self.fit_button.setEnabled(ready)
        self.preview_button.setEnabled(ready)
        self.open_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        self.save_button.setEnabled(self.result is not None and not busy)
        self.busyChanged.emit(busy)

    def load_catalogue(self, info):
        """Fill the network selector from client.info() (called by main(); a host may call it after info())."""
        current = self.network.currentData()
        self.network.clear()
        catalogue = info['models']
        for name, item in catalogue['models'].items():
            self.network.addItem(f"{name} · {item.get('title', '')}", name)
            self.network.setItemData(self.network.count()-1, item.get('description', ''), QtCore.Qt.ToolTipRole)
        index = self.network.findData(current if current in catalogue['models'] else catalogue['default'])
        self.network.setCurrentIndex(max(index, 0))

    # ------------------------------------------------------------------ export
    def save_dialog(self):
        if self.result is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, self.tr_('Save result…'), 'fit1d_result.json', self.tr_('Result (*.json)'))
        if path:
            written = save_result(self.result, path)
            self._status(f"{self.tr_('Saved')}: {', '.join(Path(p).name for p in written)}")


def save_result(result, path):
    """result.json plus <name>_parameters.csv and <name>_curves.csv next to it; returns the written paths."""
    path = Path(path)
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding='utf-8')
    table = path.with_name(path.stem+'_parameters.csv')
    with open(table, 'w', encoding='utf-8', newline='') as f:
        f.write('group,parameter,key,value,error,unit,range_lo,range_hi,status\n')
        for r in result['table']:
            rng = r.get('range') or ['', '']
            f.write(','.join(str(v) for v in (r['group'], r['parameter'], r['key'], r['value'],
                                              '' if r['error'] is None else r['error'], r['unit'], rng[0], rng[1], r['status']))+'\n')
    curves = result['curves']
    names = list(curves['components'])
    curve_path = path.with_name(path.stem+'_curves.csv')
    with open(curve_path, 'w', encoding='utf-8', newline='') as f:
        f.write(','.join(['q_nm_inv', 'intensity', 'sigma', 'model', 'residual_over_sigma']+names)+'\n')
        for i in range(len(curves['q_nm_inv'])):
            row = [curves[k][i] for k in ('q_nm_inv', 'intensity', 'sigma', 'model', 'residual')]+[curves['components'][n][i] for n in names]
            f.write(','.join(repr(float(v)) for v in row)+'\n')
    return [str(path), str(table), str(curve_path)]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    language = os.environ.get('GISAXS1D_LANG')
    panel = Fit1DPanel(language=language)
    panel.resize(1250, 760)
    panel.show()

    def warm_up():                                     # start the worker and read the network catalogue
        call = _Call(panel.client, 'info', panel.client.info, panel)
        call.done.connect(lambda _kind, info: panel.load_catalogue(info))
        call.finished.connect(lambda: (argv and panel.open_file(argv[0])))
        panel._warm = call
        call.start()
    QtCore.QTimer.singleShot(0, warm_up)
    app.aboutToQuit.connect(panel.shutdown)
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
