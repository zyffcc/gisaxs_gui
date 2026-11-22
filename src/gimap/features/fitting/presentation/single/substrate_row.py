"""The substrate in Model's grazing card and the level the spheres are evaluated at (Physics and Methods 9.10, 9.17).

The substrate is unknown (the default: the effective interface, the Born term or the two paths with a free r, as the
angles and |r| say), a material of the table (Si, SiO₂, plate glass, Au: δ and β at the cut's λ from Henke's f₁, f₂)
or δ and β typed. With a substrate and both angles the spheres are in the four-term DWBA with its Fresnel
coefficients (random cylinders at the Born term of the cut's q_z; upright cylinders are the same at every level).
The line under the row says which level is in use and, for the DWBA, δ and β.
"""

from __future__ import annotations

from PyQt5.QtCore import QSignalBlocker
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import QComboBox, QHBoxLayout, QLabel, QLineEdit, QVBoxLayout

from src.gimap.app.presentation.i18n import tr, trf

from ...application.substrate_level import CUSTOM, SUBSTRATE_CHOICES, level

SUBSTRATE_TIP = ("The substrate under the particles. Unknown: the spheres at q_z = 0, at the Born term or with two "
                 "paths and a free r, as αi, αf and |r| say. A material of the table or δ and β typed: with αi and "
                 "αf the spheres are in the four-term DWBA with the substrate's Fresnel coefficients (|r| and arg r "
                 "are then not used). The table holds the bare materials (Henke's tables, 4.9–30 keV); an oxide, a "
                 "polymer film or roughness are not part of it.")
LEVEL_LINE = "Level: {level}"
DWBA_LINE = "Level: {level} on {substrate} (δ = {delta}, β = {beta} at λ = {wavelength} nm)"
NEEDS_ANGLES = "Level: {level}. The substrate is used once αi and αf are given."
TABLE_ERROR = "The substrate cannot be used: {reason}"


class SubstrateControls:
    """The combo, δ and β, and the level line of the editor's grazing card."""

    def __init__(self, editor, card):
        self.editor = editor
        self.layout = QVBoxLayout()
        self.layout.setSpacing(2)
        row = QHBoxLayout()
        self.combo = QComboBox(card)
        self.combo.setObjectName("fitSubstrate")
        for key, text in SUBSTRATE_CHOICES:
            self.combo.addItem(tr(text), key)
        self.combo.setToolTip(tr(SUBSTRATE_TIP))
        self.delta = QLineEdit(card)
        self.delta.setObjectName("fitSubstrateDelta")
        self.beta = QLineEdit(card)
        self.beta.setObjectName("fitSubstrateBeta")
        for field, tip in ((self.delta, "δ of the substrate at the cut's wavelength (n = 1 − δ + iβ)"),
                           (self.beta, "β of the substrate at the cut's wavelength")):
            field.setValidator(QDoubleValidator(0.0, 1.0, 12, field))
            field.setToolTip(tr(tip))
            field.setMaximumWidth(84)
            field.editingFinished.connect(self._typed)
        self.delta_label, self.beta_label = QLabel("δ", card), QLabel("β", card)
        label = QLabel(tr("Substrate"), card)
        label.setToolTip(tr(SUBSTRATE_TIP))
        row.addWidget(label)
        row.addWidget(self.combo, 1)
        for widget in (self.delta_label, self.delta, self.beta_label, self.beta):
            row.addWidget(widget)
        self.layout.addLayout(row)
        self.line = QLabel("", card)
        self.line.setObjectName("fitLevel")
        self.line.setWordWrap(True)
        self.line.setProperty("gimapRole", "muted")
        self.layout.addWidget(self.line)
        self.combo.currentIndexChanged.connect(self._chosen)
        self.refresh()

    def refresh(self) -> None:
        """Show the model's substrate and level (after an edit or a new model)."""
        substrate = self.editor.model.substrate
        key = "" if substrate is None else substrate.material
        blocker = QSignalBlocker(self.combo)
        self.combo.setCurrentIndex(max(0, self.combo.findData(key)))
        del blocker
        custom = key == CUSTOM
        for widget in (self.delta_label, self.delta, self.beta_label, self.beta):
            widget.setVisible(custom)
        if custom:
            for field, value in ((self.delta, substrate.delta), (self.beta, substrate.beta)):
                if not field.hasFocus():
                    field.setText(f"{value:.6g}")
        self.line.setText(self.level_text())

    def level_text(self) -> str:
        model = self.editor.model
        found = level(model)
        if found.error:
            return trf(TABLE_ERROR, reason=found.error)
        if found.key == "dwba":
            return trf(DWBA_LINE, level=tr(found.label), substrate=tr(found.substrate.label), delta=f"{found.delta:.4g}",
                       beta=f"{found.beta:.3g}", wavelength=f"{model.globals['wavelength'].value:.5g}")
        if model.substrate is not None:
            return trf(NEEDS_ANGLES, level=tr(found.label))
        return trf(LEVEL_LINE, level=tr(found.label))

    def _chosen(self, _index: int) -> None:
        key = str(self.combo.currentData() or "")
        if key == CUSTOM:
            delta, beta = self._numbers()
            if not delta > 0:
                for widget in (self.delta_label, self.delta, self.beta_label, self.beta):
                    widget.setVisible(True)
                self.delta.setFocus()
                self.line.setText(tr("Type δ (above 0) and β of the substrate at the cut's wavelength."))
                return
            self.editor.set_substrate(CUSTOM, delta, beta)
        else:
            self.editor.set_substrate(key)

    def _numbers(self) -> tuple[float, float]:
        try:
            delta = float(self.delta.text().replace(",", "."))
        except ValueError:
            delta = 0.0
        try:
            beta = float(self.beta.text().replace(",", "."))
        except ValueError:
            beta = 0.0
        return delta, max(beta, 0.0)

    def _typed(self) -> None:
        if self.combo.currentData() == CUSTOM:
            delta, beta = self._numbers()
            if delta > 0:
                self.editor.set_substrate(CUSTOM, delta, beta)


__all__ = ["SubstrateControls"]
