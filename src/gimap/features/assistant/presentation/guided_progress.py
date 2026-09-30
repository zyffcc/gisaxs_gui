"""What the automatic analysis is doing now: the current step, the phases done, time, and Stop.

Shown at the top of the Results tab while a run lasts, like the AI's panel: one line says what is
being done (and, for a slow step, that it can take a while), the phases are ticked off as the run
goes, a bar shows how far it is. **Stop** asks the run to end before its next step; a step that is
running cannot be interrupted, so the line then says the run is waiting for it. After a stop the
panel says what was kept and offers **Save Report…** and **Discard**; the results found so far are
shown below it either way until discarded.
"""

from __future__ import annotations

import time
from typing import Optional

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton, QToolButton, QVBoxLayout, QWidget

from src.gimap.app.presentation.i18n import tr
from src.gimap.app.presentation.theme import theme_manager

from ..application import PHASES, SLOW, phase_of, step_text


def _seconds(value: float) -> str:
    return f"{value:.0f} s" if value >= 10 else f"{value:.1f} s"


class GuidedProgressPanel(QFrame):
    stopRequested = pyqtSignal()
    saveRequested = pyqtSignal()
    discardRequested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("guidedProgress")
        self.setProperty("gimapInfoCard", True)
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        header = QHBoxLayout()
        self.title_label = QLabel(tr("Automatic analysis"), self)
        self.title_label.setProperty("gimapRole", "strong")
        self.elapsed_label = QLabel("", self)
        self.elapsed_label.setProperty("gimapRole", "muted")
        self.stop_button = QPushButton(tr("Stop"), self)
        self.stop_button.setObjectName("guidedStopButton")
        self.stop_button.setProperty("gimapDangerAction", True)
        self.stop_button.setToolTip(tr("End the run before its next step; what was found so far is kept"))
        self.steps_button = QToolButton(self)
        self.steps_button.setObjectName("guidedStepsToggle")
        self.steps_button.setText(tr("Steps"))
        self.steps_button.setCheckable(True)
        self.steps_button.setAutoRaise(True)
        self.steps_button.setToolTip(tr("Show the steps of the finished run and how long each took"))
        self.steps_button.hide()
        header.addWidget(self.title_label, 1)
        header.addWidget(self.elapsed_label)
        header.addWidget(self.steps_button)
        header.addWidget(self.stop_button)
        layout.addLayout(header)
        self.now_label = QLabel("", self)
        self.now_label.setObjectName("guidedNow")
        self.now_label.setWordWrap(True)
        layout.addWidget(self.now_label)
        self.bar = QProgressBar(self)
        self.bar.setObjectName("guidedProgressBar")
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(6)
        layout.addWidget(self.bar)
        self.phases_label = QLabel("", self)
        self.phases_label.setObjectName("guidedPhases")
        self.phases_label.setTextFormat(Qt.RichText)
        self.phases_label.setWordWrap(True)
        layout.addWidget(self.phases_label)
        after = QHBoxLayout()
        self.save_button = QPushButton(tr("Save Report…"), self)
        self.save_button.setObjectName("guidedProgressSave")
        self.discard_button = QPushButton(tr("Discard"), self)
        self.discard_button.setObjectName("guidedProgressDiscard")
        self.discard_button.setToolTip(tr("Remove the results of the stopped run from the Results tab"))
        after.addWidget(self.save_button)
        after.addWidget(self.discard_button)
        after.addStretch(1)
        self.after_row = QWidget(self)
        self.after_row.setLayout(after)
        self.after_row.hide()
        layout.addWidget(self.after_row)
        self.stop_button.clicked.connect(self._stop_clicked)
        self.steps_button.toggled.connect(self.phases_label.setVisible)
        self.save_button.clicked.connect(self.saveRequested)
        self.discard_button.clicked.connect(self.discardRequested)
        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._tick)
        self._started = 0.0
        self._step_started = 0.0
        self._current: Optional[dict] = None
        self._phases: list[str] = []
        self._done_phases: set[str] = set()
        self._phase_seconds: dict[str, float] = {}
        self._expected: tuple = PHASES["giwaxs"]
        self._stopping = False
        self._steps = 0
        self.hide()

    # -- a run ---------------------------------------------------------------------------

    def start(self, message: str, *, geometry_only: bool = False) -> None:
        self._expected = PHASES["geometry"] if geometry_only else PHASES["giwaxs"]
        self._started = self._step_started = time.monotonic()
        self._current, self._phases, self._done_phases, self._phase_seconds = None, [], set(), {}
        self._stopping, self._steps = False, 0
        self.title_label.setText(tr("Automatic analysis — running"))
        self.now_label.setText(tr(message))
        self.bar.setRange(0, len(self._expected))
        self.bar.setValue(0)
        self.stop_button.setEnabled(True)
        self.stop_button.setText(tr("Stop"))
        self.stop_button.show()
        self.after_row.hide()
        self.steps_button.hide()
        for widget in (self.now_label, self.bar, self.phases_label):
            widget.show()
        self._render_phases()
        self.show()
        self._timer.start()

    def step(self, event: dict) -> None:
        """A tool call started or finished (``StandardPipeline`` events)."""
        tool = str(event.get("tool") or "")
        phase = phase_of(tool)
        if tool in ("refine_beam_center_symmetry", "in_plane_spacing", "fit_horizontal_cut"):
            self._expected = PHASES["gisaxs"]
        if event.get("state") == "start":
            self._current, self._step_started = event, time.monotonic()
            if phase is not None and phase not in self._phases:
                for earlier in self._phases:
                    self._done_phases.add(earlier)
                self._phases.append(phase)
            self._show_now()
        else:
            self._steps += 1
            if phase is not None:
                self._phase_seconds[phase] = self._phase_seconds.get(phase, 0.0) + float(event.get("seconds") or 0.0)
            self._current = None
        self.bar.setRange(0, len(self._expected))
        self.bar.setValue(len(self._done_phases))
        self._render_phases()

    def stopping(self) -> None:
        """Stop was pressed: the run ends after the step in progress."""
        self._stopping = True
        self.stop_button.setEnabled(False)
        self.stop_button.setText(tr("Stopping…"))
        self._show_now()

    def finish(self, report: Optional[dict], *, failed: str = "") -> None:
        """The run ended: done, stopped (results kept, Save / Discard) or failed."""
        self._timer.stop()
        self._current = None
        total = _seconds(time.monotonic() - self._started)
        self.stop_button.hide()
        self.elapsed_label.setText(total)
        if failed:
            self.title_label.setText(tr("Automatic analysis — stopped by an error"))
            self.now_label.setText(failed)
            self.after_row.hide()
        elif report is not None and report.get("stopped"):
            self.title_label.setText(tr("Automatic analysis — stopped"))
            self.now_label.setText(tr(
                "Stopped after {steps} steps, before: {next}. What was found so far is below — keep it, "
                "save it as a report, or discard it."
            ).format(steps=self._steps, next=step_text(str(report["stopped"]), None, tr)))
            self.after_row.show()
        else:
            for phase in self._phases:
                self._done_phases.add(phase)
            self.bar.setValue(self.bar.maximum())
            self.title_label.setText(tr("Automatic analysis — done"))
            self.elapsed_label.setText(tr("{steps} steps · {time}").format(steps=self._steps, time=total))
            text = tr("It finished the last step before it could stop.") if self._stopping else ""
            self.now_label.setText(text)
            self.now_label.setVisible(bool(text))
            self.after_row.hide()
            # Done: one line; the steps stay one click away.
            self.bar.hide()
            self.steps_button.setChecked(False)
            self.phases_label.hide()
            self.steps_button.show()
        self._render_phases()

    def dismiss(self) -> None:
        self._timer.stop()
        self.hide()

    # -- display ---------------------------------------------------------------------------

    def _stop_clicked(self) -> None:
        self.stopping()
        self.stopRequested.emit()

    def _tick(self) -> None:
        self.elapsed_label.setText(_seconds(time.monotonic() - self._started))
        self._show_now()

    def _show_now(self) -> None:
        event = self._current
        if event is None:
            if self._stopping:
                self.now_label.setText(tr("Stopping: the run ends before its next step …"))
            return
        tool = str(event.get("tool") or "")
        text = step_text(tool, event.get("arguments"), tr)
        running = time.monotonic() - self._step_started
        line = tr("Now: {step} … {time}").format(step=text, time=_seconds(running))
        slow = SLOW.get(tool)
        if self._stopping:
            line = tr("Stopping: waiting for “{step}” to finish ({time} so far").format(step=text, time=_seconds(running))
            line += tr("; this step {slow})").format(slow=tr(slow)) if slow else ")"
        elif slow and running > 2.0:
            line += " — " + tr("this step {slow}").format(slow=tr(slow))
        self.now_label.setText(line)

    def _render_phases(self) -> None:
        manager = theme_manager()
        muted, faint = manager.color("text_muted").name(), manager.color("text_faint").name()
        rows = []
        visited = [index for index, phase in enumerate(self._expected) if phase in self._phases]
        last = max(visited) if visited else -1
        for index, phase in enumerate(self._expected):
            seconds = self._phase_seconds.get(phase)
            if phase not in self._phases and index < last:  # passed over: e.g. the geometry was already known
                rows.append(f"<span style='color:{faint}'>– {tr(phase)}: {tr('not needed')}</span>")
                continue
            time_text = f" <span style='color:{muted}'>{_seconds(seconds)}</span>" if seconds else ""
            if phase in self._done_phases:
                rows.append(f"✓ {tr(phase)}{time_text}")
            elif phase in self._phases:
                rows.append(f"<b>▸ {tr(phase)}</b>{time_text}")
            else:
                rows.append(f"<span style='color:{faint}'>· {tr(phase)}</span>")
        self.phases_label.setText("<br>".join(rows))


__all__ = ["GuidedProgressPanel"]
