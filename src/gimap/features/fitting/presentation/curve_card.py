"""The curve being fitted: where it came from and how to open another one."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from src.gimap.app.presentation.theme import set_role

from .layout_primitives import detach_from_parent_layout

EMPTY_HINT = (
    "Send a cut from Analyze (Send to Fitting), or open a curve file "
    "with q (Å⁻¹), I and optional σ columns."
)


class CurveSourceCard(QFrame):
    """Show the loaded curve; ``ui.fitImport1dFileButton`` opens another one."""

    def __init__(self, ui, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("fittingCurveCard")
        self.setProperty("gimapSection", True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(4)

        header = QHBoxLayout()
        header.setSpacing(8)
        title = QLabel("Curve", self)
        title.setProperty("gimapSectionTitle", True)
        header.addWidget(title)
        header.addStretch(1)
        button = ui.fitImport1dFileButton
        detach_from_parent_layout(button)
        button.setParent(self)
        button.setText("Open Curve…")
        button.setToolTip("Open a 1D curve file (q in Å⁻¹; columns q, I[, σ[, pixels]])")
        header.addWidget(button)
        layout.addLayout(header)

        self.name_label = QLabel("No curve", self)
        self.name_label.setObjectName("fittingCurveName")
        self.name_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        set_role(self.name_label, "strong")
        self.detail_label = QLabel(EMPTY_HINT, self)
        self.detail_label.setObjectName("fittingCurveDetail")
        self.detail_label.setWordWrap(True)
        set_role(self.detail_label, "muted")
        layout.addWidget(self.name_label)
        layout.addWidget(self.detail_label)

        # The path editor stays for the binding (Enter loads a typed path) but
        # the card shows the file name; the full path is in the tooltip.
        path_edit = ui.fitImport1dFileValue
        detach_from_parent_layout(path_edit)
        path_edit.setParent(self)
        path_edit.hide()
        ui.fittingCurveCard = self

    def show_curve(self, path, q, *, observation=None, unit: str = "angstrom") -> None:
        """Describe a loaded curve: points, q range and its origin (q in nm⁻¹ like the plot, Å⁻¹ as in Analyze).

        ``unit`` is the unit of ``q`` as read from the file: ``"angstrom"`` (Å⁻¹) or ``"nm"`` (nm⁻¹).
        """
        path = Path(str(path))
        q = np.asarray(q, dtype=float)
        if str(unit).lower() == "nm":
            q = q / 10.0  # to Å⁻¹
        finite = q[np.isfinite(q)]
        self.name_label.setText(path.name)
        self.name_label.setToolTip(str(path))
        parts = [f"{q.size} points"]
        if finite.size:
            parts.append(
                f"q {10 * finite.min():.4g} … {10 * finite.max():.4g} nm⁻¹ ({finite.min():.4g} … {finite.max():.4g} Å⁻¹)"
            )
        observation = observation or {}
        if observation.get("source") == "native_detector_columns":
            detector = str(observation.get("file_format", "")).upper()
            summed = int(observation.get("summed_frames", 1) or 1)
            origin = f"Analyze cut ({detector} detector columns"
            origin += f", sum of {summed} frames)" if summed > 1 else ")"
            parts.append(origin)
        elif observation:
            parts.append("Analyze curve")
        self.detail_label.setText(" · ".join(parts))


__all__ = ["CurveSourceCard", "EMPTY_HINT"]
