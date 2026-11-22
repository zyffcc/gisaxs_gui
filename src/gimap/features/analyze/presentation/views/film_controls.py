"""Film absorption controls in Analyze's intensity corrections section."""

from pathlib import Path

from PyQt5.QtCore import QLocale, QSignalBlocker, pyqtSignal
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                             QHBoxLayout, QLabel, QLineEdit, QToolButton, QVBoxLayout, QWidget)

from src.gimap.app.presentation.i18n import tr

from .command_bar import ElidedLabel


def _length(parent, name, minimum, step, unit, decimals):
    spin = QDoubleSpinBox(parent)
    spin.setObjectName(name)
    spin.setDecimals(decimals)
    spin.setRange(minimum, 1e6)
    spin.setValue(100.)
    spin.setSingleStep(step)
    spin.setSuffix(unit)
    spin.setKeyboardTracking(False)
    return spin


class FilmControls(QWidget):
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.check = QCheckBox("Absorption in the film", self)
        self.check.setObjectName("analyzeFilmCheck")
        self.check.setToolTip("Divide each pixel by the film factor relative to the specular exit angle.")
        layout.addWidget(self.check)
        self.details = QWidget(self)
        form = QFormLayout(self.details)
        form.setContentsMargins(0, 0, 0, 0)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.model = QComboBox(self)
        self.model.setObjectName("analyzeFilmModel")
        for label, key in (("Attenuation length", "length"), ("Enter δ and β", "typed"), ("Film CIF", "cif")):
            self.model.addItem(label, key)
        form.addRow("Film model", self.model)
        self.thick = QCheckBox("Thick film limit", self)
        self.thick.setChecked(True)
        form.addRow(self.thick)
        self.thickness = _length(self, "analyzeFilmThickness", .1, 10., " nm", 1)
        self.attenuation = _length(self, "analyzeAttenuationLength", .001, 1., " µm", 3)
        self.attenuation.setToolTip("Attenuation length 1/µ of the film at this energy (tables such as CXRO or NIST)")
        form.addRow("Thickness", self.thickness)
        form.addRow("Attenuation length", self.attenuation)
        self.delta, self.beta = QLineEdit(self), QLineEdit(self)
        locale = QLocale.c()
        locale.setNumberOptions(QLocale.RejectGroupSeparator)
        for edit, label in ((self.delta, "δ (10⁻⁶)"), (self.beta, "β (10⁻⁶)")):
            validator = QDoubleValidator(edit)
            if edit is self.beta:
                validator.setBottom(0.)
            validator.setLocale(locale)
            edit.setValidator(validator)
            edit.setPlaceholderText("Enter value")
            form.addRow(label, edit)
            edit.textChanged.connect(self._changed)
        self.cif = QWidget(self)
        row = QHBoxLayout(self.cif)
        row.setContentsMargins(0, 0, 0, 0)
        choose = QToolButton(self.cif)
        choose.setText("Choose CIF…")
        choose.clicked.connect(self._choose)
        self.file_name = ElidedLabel(parent=self.cif, padding=0)
        row.addWidget(choose)
        row.addWidget(self.file_name, 1)
        form.addRow(self.cif)
        self.cif_path = None
        self.hint = QLabel("Enter both optical constants to apply the correction.", self)
        self.hint.setWordWrap(True)
        self.hint.setProperty("gimapRole", "muted")
        form.addRow(self.hint)
        layout.addWidget(self.details)
        self.check.toggled.connect(self._changed)
        self.model.currentIndexChanged.connect(self._changed)
        self.thick.toggled.connect(self._changed)
        self.thickness.valueChanged.connect(self.changed)
        self.attenuation.valueChanged.connect(self.changed)
        self._update()

    def _choose(self):
        path, _ = QFileDialog.getOpenFileName(self, tr("Film CIF"), self.cif_path or "",
                                              tr("CIF files (*.cif);;All files (*)"))
        if path:
            self.cif_path = path
            self._changed()

    def _update(self):
        mode = self.model.currentData()
        self.details.setVisible(self.check.isChecked())
        self.thick.setVisible(mode != "length")
        form = self.details.layout()
        for widget, visible in ((self.thickness, mode == "length" or not self.thick.isChecked()),
                                (self.attenuation, mode == "length"),
                                (self.delta, mode == "typed"), (self.beta, mode == "typed")):
            widget.setVisible(visible)
            form.labelForField(widget).setVisible(visible)
        self.cif.setVisible(mode == "cif")
        self.file_name.setText(Path(self.cif_path).name if self.cif_path else "")
        self.file_name.setToolTip(self.cif_path or "")
        self.hint.setVisible(mode == "typed" and not self._constants())

    def _changed(self, *_args):
        self._update()
        self.changed.emit()

    def _constants(self):
        if self.delta.hasAcceptableInput() and self.beta.hasAcceptableInput():
            return tuple(float(edit.text()) * 1e-6 for edit in (self.delta, self.beta))
        return None

    def values(self):
        mode = self.model.currentData()
        thickness = length = delta = beta = cif_path = None
        if self.check.isChecked():
            if mode == "length":
                thickness, length = self.thickness.value(), self.attenuation.value()
            elif mode == "typed" and self._constants():
                delta, beta = self._constants()
            elif mode == "cif":
                cif_path = self.cif_path
            if (delta is not None or cif_path) and not self.thick.isChecked():
                thickness = self.thickness.value()
        return dict(film_thickness_nm=thickness, attenuation_length_um=length, film_cif_path=cif_path,
                    film_delta=delta, film_beta=beta)

    def restore(self, corrections):
        """Restore the panel after loading settings or undoing an edit."""
        mode = "cif" if corrections.film_cif_path else "typed" if corrections.film_delta is not None else "length"
        with QSignalBlocker(self):
            self.model.setCurrentIndex(self.model.findData(mode))
            self.check.setChecked(mode != "length" or (corrections.film_thickness_nm is not None
                                                       and corrections.attenuation_length_um is not None))
            self.thick.setChecked(corrections.film_thickness_nm is None)
            for spin, value in ((self.thickness, corrections.film_thickness_nm),
                                (self.attenuation, corrections.attenuation_length_um)):
                if value is not None:
                    spin.setValue(value)
            for edit, value in ((self.delta, corrections.film_delta), (self.beta, corrections.film_beta)):
                edit.setText("" if value is None else format(value * 1e6, ".15g"))
            self.cif_path = corrections.film_cif_path
            self._update()
