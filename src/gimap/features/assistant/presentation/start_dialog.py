"""“Process with AI”: choose the results, add notes and permissions, then start."""

from __future__ import annotations

import html
from typing import Callable, Optional

from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ..application import (
    BACKEND_API,
    BACKEND_CLAUDE_CODE,
    BACKEND_PROVIDER,
    GOAL_RING,
    GISAXS_GOALS,
    GIWAXS_GOALS,
    GOALS,
    LANGUAGES,
    PERMISSION_AUTO,
    PERMISSION_CONFIRM,
    PERMISSION_PREVIEW,
    AnalysisGoals,
)
from src.gimap.app.presentation.i18n import tr

from . import preferences

GOAL_LABELS = {
    "peaks": "Peak table (q, d, FWHM, intensity)",
    "orientation": "Orientation: in-plane vs out-of-plane",
    "ring_orientation": "Orientation distribution of one ring",
    "crystallite_size": "Crystallite size (Scherrer)",
    "gisaxs_cut": "Horizontal cut at Yoneda, centred, halves chosen",
    "in_plane_spacing": "In-plane distance 2π/q*",
    "gisaxs_fit": "Fit of I(qy): shape, radius, distance",
}
"""Short labels; the full description of each result is the tooltip."""
BRAINS = (
    (BACKEND_CLAUDE_CODE, "Claude Code — your Claude plan (Pro / Max), no API key"),
    (BACKEND_API, "Claude API — API key, billed per token"),
    (BACKEND_PROVIDER, "Other AI provider — DeepSeek, Qwen, OpenAI, Kimi, GLM, Gemini, Ollama … (Set Up AI…)"),
)
PRIVACY_NOTE = (
    "The frame's status and reduced curves are sent to the AI as numbers; the detector file is "
    "not (only a small q-map image, if allowed above)."
)


def _muted(text: str, parent: QWidget) -> QLabel:
    label = QLabel(text, parent)
    label.setWordWrap(True)
    label.setProperty("gimapRole", "muted")
    return label


class AssistantStartDialog(QDialog):
    """Choices for one run; the owner reports whether the chosen brain is ready (``set_status``)."""

    backendChanged = pyqtSignal(str)

    def __init__(
        self,
        settings,
        *,
        status: dict,
        configure: Optional[Callable[[], None]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("assistantStartDialog")
        self.setWindowTitle(tr("Process with AI"))
        self.settings = settings
        self._ready = False
        self._configure = configure
        self._has_frame = status.get("file") is not None
        layout = QVBoxLayout(self)
        frame = status.get("file") or tr("no frame")
        if status.get("file") and status.get("geometry") is None:
            kind = tr("NO GEOMETRY YET")
        else:
            measurement = status.get("measurement")
            kind = str(measurement).upper() if measurement else tr("NOT ANALYSED")
        layout.addWidget(QLabel(f"<b>{html.escape(frame)}</b> · {html.escape(kind)}", self))
        explanation = [tr(
            "The AI operates the Analyze tools on this frame while you watch each step, then "
            "reports every result you tick below. What cannot be determined is reported with the "
            "reason (e.g. no signal above the noise)."
        )]
        if status.get("file") and status.get("geometry") is None:
            explanation.append(tr(
                "This frame has no geometry yet: the AI looks for calibration files (standard "
                "images, .poni) and logs around it, fits or reads the geometry, and asks you when "
                "it cannot tell which calibration is right."
            ))
        gisaxs = status.get("measurement") == "gisaxs" or status.get("mode") == "gisaxs"
        if gisaxs:
            explanation.append(tr("GISAXS: the horizontal cut at the Yoneda band, its halves, the spacing and a fit."))
        layout.addWidget(_muted(" ".join(explanation), self))

        goals_box = QGroupBox(tr("Results"), self)
        goals_layout = QVBoxLayout(goals_box)
        offered = GISAXS_GOALS if gisaxs else GIWAXS_GOALS
        chosen = set(preferences.read(settings, "goals") or GOALS) & set(offered) or set(offered)
        self.goal_checks: dict[str, QCheckBox] = {}
        self.ring_spin = QDoubleSpinBox(goals_box)
        self.ring_spin.setObjectName("assistantRingSpin")
        self.ring_spin.setRange(0.0, 10.0)
        self.ring_spin.setDecimals(3)
        self.ring_spin.setSingleStep(0.01)
        self.ring_spin.setSuffix(" Å⁻¹")
        self.ring_spin.setSpecialValueText(tr("choose automatically"))
        self.ring_spin.setValue(float(preferences.read(settings, "ring_q") or 0.0))
        for key in offered:
            check = QCheckBox(tr(GOAL_LABELS[key]), goals_box)
            check.setObjectName(f"assistantGoal_{key}")
            check.setToolTip(tr(GOALS[key]))
            check.setChecked(key in chosen)
            goals_layout.addWidget(check)
            self.goal_checks[key] = check
            if key == GOAL_RING:
                ring_row = QHBoxLayout()
                ring_row.addSpacing(24)
                ring_row.addWidget(QLabel(tr("Ring at q ="), goals_box))
                ring_row.addWidget(self.ring_spin)
                ring_row.addStretch(1)
                goals_layout.addLayout(ring_row)
        if GOAL_RING in self.goal_checks:
            self.goal_checks[GOAL_RING].toggled.connect(self.ring_spin.setEnabled)
            self.ring_spin.setEnabled(self.goal_checks[GOAL_RING].isChecked())
        else:
            self.ring_spin.hide()
        layout.addWidget(goals_box)

        layout.addWidget(QLabel(tr("Sample and instructions (optional)"), self))
        self.notes_edit = QPlainTextEdit(self)
        self.notes_edit.setObjectName("assistantNotesEdit")
        self.notes_edit.setPlaceholderText(tr(
            "e.g. P3HT film on Si, annealed at 150 °C; judge edge-on versus face-on from the "
            "(100) and (010) peaks"
        ))
        self.notes_edit.setFixedHeight(72)
        layout.addWidget(self.notes_edit)

        options = QFormLayout()
        self.brain_combo = QComboBox(self)
        self.brain_combo.setObjectName("assistantBrainCombo")
        for key, title in BRAINS:
            self.brain_combo.addItem(tr(title), key)
        self.brain_combo.setCurrentIndex(max(0, self.brain_combo.findData(preferences.read(settings, "backend"))))
        self.brain_combo.currentIndexChanged.connect(self._brain_chosen)
        options.addRow(tr("Brain"), self.brain_combo)
        permission_row = QVBoxLayout()
        self.preview_radio = QRadioButton(tr(
            "Preview first — the AI's changes come back as cards: look at the picture, then apply or dismiss"
        ), self)
        self.preview_radio.setObjectName("assistantPermissionPreview")
        self.confirm_radio = QRadioButton(tr("Ask me before writing files or changing corrections"), self)
        self.auto_radio = QRadioButton(tr("Fully automatic (everything is logged and can be undone)"), self)
        group = QButtonGroup(self)
        radios = {PERMISSION_PREVIEW: self.preview_radio, PERMISSION_CONFIRM: self.confirm_radio, PERMISSION_AUTO: self.auto_radio}
        for radio in radios.values():
            group.addButton(radio)
            permission_row.addWidget(radio)
        radios.get(preferences.read(settings, "permission"), self.preview_radio).setChecked(True)
        options.addRow(tr("Permissions"), permission_row)
        self.images_check = QCheckBox(tr("Let the AI see a small image of the q map (qualitative only; Claude brains)"), self)
        self.images_check.setChecked(bool(preferences.read(settings, "allow_images")))
        options.addRow(tr("Images"), self.images_check)
        self.language_combo = QComboBox(self)
        self.language_combo.setObjectName("assistantReportLanguage")
        for name in LANGUAGES:  # the names stay as they are (never translated): the report language is read from them
            self.language_combo.addItem(name, name)
        self.language_combo.setCurrentIndex(max(0, self.language_combo.findData(preferences.report_language(settings))))
        options.addRow(tr("Report language"), self.language_combo)
        layout.addLayout(options)

        self.credentials_label = _muted("", self)
        self.credentials_label.setObjectName("assistantCredentialsNote")
        layout.addWidget(self.credentials_label)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.start_button = self.buttons.button(QDialogButtonBox.Ok)
        self.start_button.setText(tr("Start"))
        self.buttons.button(QDialogButtonBox.Cancel).setText(tr("Cancel"))
        self.setup_button = self.buttons.addButton(tr("Set Up AI…"), QDialogButtonBox.ResetRole)
        self.setup_button.setObjectName("assistantSetupButton")
        self.setup_button.setVisible(configure is not None)
        self.setup_button.clicked.connect(self._set_up)
        self.buttons.accepted.connect(self._accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        for check in self.goal_checks.values():
            check.toggled.connect(self._sync_start)
        self.set_status(False, tr("Checking…"))

    # Wrapped labels need more height when the dialog is narrow; Qt does not
    # grow a window for that by itself and squeezes the other rows instead.
    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._fit_height()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit_height()

    def _fit_height(self) -> None:
        needed = self.layout().heightForWidth(self.width())
        if needed > self.height():
            self.resize(self.width(), needed)

    def backend(self) -> str:
        return str(self.brain_combo.currentData())

    def set_status(self, ready: bool, text: str) -> None:
        """Whether the chosen brain can start, and what to tell the person about it."""
        self._ready = bool(ready)
        self.credentials_label.setText(f"{text} {tr(PRIVACY_NOTE)}".strip())
        self._sync_start()
        if self.isVisible():  # a longer answer (e.g. where to find Claude Code) needs more height, not squeezed rows
            QTimer.singleShot(0, self._fit_height)

    def _brain_chosen(self, _index: int) -> None:
        preferences.write(self.settings, "backend", self.backend())
        self.set_status(False, tr("Checking…"))
        self.backendChanged.emit(self.backend())

    def _sync_start(self, *_args) -> None:
        chosen = any(check.isChecked() for check in self.goal_checks.values())
        self.start_button.setEnabled(self._ready and self._has_frame and chosen)
        if not self._has_frame:
            self.start_button.setToolTip(tr("Open and analyse a frame in Analyze first."))

    def _set_up(self) -> None:
        if self._configure is not None:
            self._configure()
            self.backendChanged.emit(self.backend())

    def goals(self) -> AnalysisGoals:
        ring = self.ring_spin.value()
        return AnalysisGoals(
            goals=tuple(key for key, check in self.goal_checks.items() if check.isChecked()),
            instructions=self.notes_edit.toPlainText().strip(),
            ring_q=ring if ring > 0 and GOAL_RING in self.goal_checks and self.goal_checks[GOAL_RING].isChecked() else None,
            permission=(
                PERMISSION_AUTO if self.auto_radio.isChecked()
                else PERMISSION_PREVIEW if self.preview_radio.isChecked() else PERMISSION_CONFIRM
            ),
            allow_images=self.images_check.isChecked(),
            language=str(self.language_combo.currentData() or self.language_combo.currentText()),
        )

    def _accept(self) -> None:
        goals: Optional[AnalysisGoals]
        try:
            goals = self.goals()
        except ValueError:
            return
        others = [goal for goal in preferences.read(self.settings, "goals") or () if goal not in self.goal_checks]
        preferences.write(self.settings, "goals", others + list(goals.goals))  # keeps the other technique's choice
        preferences.write(self.settings, "ring_q", goals.ring_q or 0.0)
        preferences.write(self.settings, "permission", goals.permission)
        preferences.write(self.settings, "allow_images", goals.allow_images)
        preferences.write(self.settings, "language", goals.language)
        preferences.write(self.settings, "backend", self.backend())
        self.accept()


__all__ = ["AssistantStartDialog", "BRAINS", "GOAL_LABELS"]
