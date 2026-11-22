"""Measured geometry and optional substrate of a Predict run, shared by the single and batch panels."""

from __future__ import annotations

import math

from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QVBoxLayout,
)

from src.gimap.app.presentation.i18n import language_changed, tr, trf

from ..application.substrate_level import CUSTOM, HC_KEV_NM, SUBSTRATE_CHOICES, Substrate

EDIT_CUT = "Cut geometry and substrate…"
ON_SUBSTRATE = "four-term DWBA on {substrate}"
BAD_ANGLES = "Enter both cut angles between 0° and 90° for the known substrate."
BAD_NUMBER = "Enter a finite number for {field}."
TABLE_RANGE = "The material table covers 4.9–30 keV. Enter δ and β at this energy."


class PredictSubstrateDialog(QDialog):
    def __init__(self, cut, parent=None, *, batch=False):
        super().__init__(parent)
        self.setWindowTitle(tr("Cut geometry and substrate"))
        self.setMinimumWidth(360)
        self.batch = batch
        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)
        self.form = form
        self.support = QComboBox()
        for key, label in SUBSTRATE_CHOICES:
            self.support.addItem(tr(label), key)
        form.addRow(tr("Substrate"), self.support)
        substrate = cut.get("substrate") or {}
        self.support.setCurrentIndex(max(0, self.support.findData(substrate.get("material", ""))))
        self.delta = self._field("Substrate δ", substrate.get("delta"))
        self.beta = self._field("Substrate β", substrate.get("beta"))
        grazing = cut.get("grazing") or {}
        self.incidence = self._field("Incidence angle αi (°)", grazing.get("alpha_i"))
        self.exit = self._field("Exit angle αf (°)", grazing.get("alpha_f"))
        self.beam_unit = QComboBox()
        self.beam_unit.addItem(tr("Wavelength (nm)"), "wavelength")
        self.beam_unit.addItem(tr("Photon energy (keV)"), "energy")
        self.beam = QLineEdit(str(grazing.get("wavelength", "")))
        form.addRow(self.beam_unit, self.beam)
        self.coated = QCheckBox(tr("Top layer"))
        form.addRow(self.coated)
        layers = substrate.get("layers") or ()
        layer = layers[0] if layers else {}
        self.lower_layers = list(layers[1:])
        self.coated.setChecked(bool(layers))
        self.thickness = self._field("Layer thickness (nm)", layer.get("thickness_nm"))
        self.layer_delta = self._field("Layer δ", layer.get("delta"))
        self.layer_beta = self._field("Layer β", layer.get("beta"))
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.setProperty("gimapRole", "warning")
        layout.addWidget(self.note)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.ok = buttons.button(QDialogButtonBox.Ok)
        self.beam_unit.currentIndexChanged.connect(self._convert)
        self.support.currentIndexChanged.connect(self.refresh)
        self.coated.toggled.connect(self.refresh)
        for field in self.findChildren(QLineEdit):
            field.textChanged.connect(self.refresh)
        language_changed().connect(self.refresh)
        self.refresh()

    def _field(self, title, value):
        field = QLineEdit("" if value is None else str(value))
        self.form.addRow(tr(title), field)
        return field

    def _convert(self):
        try:
            value = float(self.beam.text().replace(",", "."))
        except ValueError:
            return
        if math.isfinite(value) and value > 0:
            self.beam.setText(str(HC_KEV_NM / value))

    def refresh(self, *_):
        material = self.support.currentData()
        self.coated.setVisible(bool(material))
        rows = ((material == CUSTOM, (self.delta, self.beta)),
                (material and self.coated.isChecked(), (self.thickness, self.layer_delta, self.layer_beta)),
                (material or not self.batch, (self.incidence, self.exit, self.beam)))
        for visible, fields in rows:
            for field in fields:
                self.form.labelForField(field).setVisible(bool(visible))
                field.setVisible(bool(visible))
        try:
            self.cut()
            message = ""
        except ValueError as exc:
            message = str(exc)
        self.note.setText(message)
        self.note.setVisible(bool(message))
        self.ok.setEnabled(not message)

    def _number(self, field, label, *, blank=None):
        text = field.text().strip().replace(",", ".")
        if not text and blank is not None:
            return blank
        try:
            value = float(text)
        except ValueError:
            value = math.nan
        if not math.isfinite(value):
            raise ValueError(trf(BAD_NUMBER, field=tr(label)))
        return value

    def cut(self):
        material = self.support.currentData()
        if not material and self.batch:
            return {}
        alpha_i = self._number(self.incidence, "Incidence angle αi (°)", blank=0.)
        alpha_f = self._number(self.exit, "Exit angle αf (°)", blank=0.)
        beam = self._number(self.beam, self.beam_unit.currentText())
        if beam <= 0:
            raise ValueError(tr("Wavelength and photon energy must be positive."))
        wavelength = beam if self.beam_unit.currentData() == "wavelength" else HC_KEV_NM / beam
        if not (0 <= alpha_i < 90 and 0 <= alpha_f < 90):
            raise ValueError(tr("Cut angles must be at least 0° and below 90°."))
        cut = dict(grazing=dict(wavelength=wavelength, alpha_i=alpha_i, alpha_f=alpha_f))
        if material:
            if not (alpha_i and alpha_f):
                raise ValueError(tr(BAD_ANGLES))
            support = {"material": material}
            if material == CUSTOM:
                support.update(delta=self._number(self.delta, "Substrate δ"), beta=self._number(self.beta, "Substrate β"))
            if self.coated.isChecked():
                top = dict(thickness_nm=self._number(self.thickness, "Layer thickness (nm)"),
                           delta=self._number(self.layer_delta, "Layer δ"), beta=self._number(self.layer_beta, "Layer β"))
                support["layers"] = [top, *self.lower_layers]
            elif self.lower_layers:
                support["layers"] = self.lower_layers
            try:
                known = Substrate.from_dict(support)
                known.optical(wavelength)
            except ValueError as exc:
                raise ValueError(tr(TABLE_RANGE if str(exc).startswith("The table of substrates") else str(exc))) from exc
            cut["substrate"] = known.to_dict()
        return cut
