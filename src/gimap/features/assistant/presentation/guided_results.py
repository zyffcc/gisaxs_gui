"""What the automatic analysis found, for the Results tab of the Analyze workspace.

Most useful first: a one-line outcome, the peaks (a table whose headers say how each value is
obtained; **Fit details**, closed until opened, show how the selected peak was fitted and where it
lies on the q map), then the checks (artefacts, shadow, missing wedge, the analysed rings on the
q map), the geometry and its quality, the series comparison and the report. Every picture is drawn
by Analyze from the frame on screen.
"""

from __future__ import annotations

from typing import Callable, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QFormLayout,
    QLabel,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from src.gimap.app.presentation.components import AdvancedSection
from src.gimap.app.presentation.i18n import tr

from ..application import ring_overlays, series_changes
from .guided_details import PeakDetails
from .guided_gisaxs import GisaxsResults, outcome_text
from .guided_report import report_markdown
from .guided_text import (
    GOOD,
    NOTE,
    PEAK_COLUMNS,
    WARN,
    badge,
    clear_layout,
    coverage_notes,
    label,
    peak_rows,
    ring_summary,
    table,
)

PEAK_MAP_PX = 460
Q_MAP_PX = 520
GEOMETRY_DECISIONS = (
    "geometry", "calibration", "calibration candidates", "calibration standard", "energy", "incidence angle", "pixel size",
)


def automatic_outcome(report: dict) -> tuple[str, str]:
    """(state, one line) of an automatic analysis, for the Results step of Analyze."""
    attention = report.get("needs_attention") or []
    if report.get("stopped"):
        return "warn", "Stopped by you; what was found so far is kept (Results tab)."
    if not report.get("ok"):
        return "warn", f"Needs {len(attention)} answer(s) only you can give (Results step)."
    if report.get("procedure") == "geometry":
        quality = (report.get("calibration_quality") or {}).get("assessment") or "from the instrument profile"
        return "ok", f"Geometry {quality}."
    if report.get("procedure") == "gisaxs":
        text = outcome_text(report).removeprefix("Done: ").rstrip(".")
    else:
        peaks, rings = report.get("peaks") or [], report.get("rings") or []
        reliable = sum(1 for peak in peaks if not peak.get("caveat"))
        text = f"{len(peaks)} peaks ({reliable} reliable), {len(rings)} rings analysed"
    return "ok", text + (f"; {len(attention)} question(s)" if attention else "") + "."


def _heading(text: str, parent: QWidget) -> QLabel:
    heading = QLabel(text, parent)
    heading.setProperty("gimapInspectorTitle", True)
    return heading


class GuidedResultsPanel(QWidget):
    compareRequested = pyqtSignal()
    saveRequested = pyqtSignal()
    refineRequested = pyqtSignal()
    """GISAXS: open the prepared curve in Fitting."""
    details_open = False
    """Whether Fit details are open: kept for the next report (a person who opened them wants them)."""
    solutionRequested = pyqtSignal(dict)
    """GISAXS: … with the selected solution drawn (``guided_details.candidate_row``)."""

    def __init__(self, automation: Callable[[], object], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("guidedResults")
        self._automation = automation
        self.report: Optional[dict] = None
        self.start_report: Optional[dict] = None
        self.peak_table = None
        self.peak_details: Optional[AdvancedSection] = None
        self.peak_map: Optional[QLabel] = None
        self.q_map: Optional[QLabel] = None
        self.gisaxs: Optional[GisaxsResults] = None
        self.layout_ = QVBoxLayout(self)
        self.layout_.setContentsMargins(0, 0, 0, 0)
        self.layout_.setSpacing(10)
        self.hide()

    # -- the whole report -------------------------------------------------------------

    def show_report(self, report: dict, start_report: Optional[dict] = None) -> None:
        self.report, self.start_report = report, start_report
        if self.gisaxs is not None:
            self.gisaxs.dispose()
        clear_layout(self.layout_)
        self.peak_table = self.peak_details = self.peak_map = self.q_map = self.gisaxs = None
        self._outcome(report)
        if report.get("procedure") == "gisaxs":
            if report.get("ok"):
                self.gisaxs = GisaxsResults(report, self, details=lambda: self._details_section("guidedFitDetailsSection"))
                self.gisaxs.refineRequested.connect(self.refineRequested)
                self.gisaxs.solutionRequested.connect(self.solutionRequested)
                self.layout_.addWidget(self.gisaxs)
        elif report.get("peaks"):
            self._peaks(report)
        if report.get("ok") and report.get("procedure") != "gisaxs":
            self._checks(report)
        self._geometry(report)
        frames = report.get("frames") or {}
        if report.get("ok") and int(frames.get("total") or 1) > 1:
            self._series(report)
        self._report(report)
        self.layout_.addStretch(1)
        self.show()

    def _outcome(self, report: dict) -> None:
        attention = report.get("needs_attention") or []
        peaks = report.get("peaks") or []
        rings = report.get("rings") or []
        if report.get("stopped"):
            done = len(report.get("steps") or [])
            self.layout_.addWidget(badge(
                f"Stopped by you after {done} steps: what was found so far is below "
                f"({len(peaks)} peaks, {len(rings)} rings analysed).", NOTE, self,
            ))
        elif report.get("ok"):
            reliable = sum(1 for peak in peaks if not peak.get("caveat"))
            text = f"Done: {len(peaks)} peaks ({reliable} reliable), {len(rings)} rings analysed."
            if report.get("procedure") == "gisaxs":
                text = outcome_text(report)
            if attention:
                text += f" {len(attention)} question(s) in the Results step."
            self.layout_.addWidget(badge(text, GOOD, self))
            for item in attention if report.get("procedure") == "gisaxs" else ():
                self.layout_.addWidget(badge(f"{item['item']}: {item['why']}", NOTE, self))
                self.layout_.addWidget(label(item.get("hint") or "", self, role="muted"))
        else:
            self.layout_.addWidget(badge(
                "The analysis needs your answers in the Results step before it can give results.", WARN, self,
            ))
            for item in attention:
                self.layout_.addWidget(label(f"• {item['item']}: {item['why']}", self, role="muted"))

    def _peaks(self, report: dict) -> None:
        peaks = report.get("peaks") or []
        self.layout_.addWidget(_heading("Peaks", self))
        self.peak_table = table(PEAK_COLUMNS, peak_rows(peaks), self, "guidedPeakTable")
        self.peak_table.setMinimumHeight(min(360, 60 + 30 * len(peaks)))
        self.layout_.addWidget(self.peak_table)
        self.peak_details = self._details_section("guidedPeakDetailsSection")
        details = PeakDetails(self._picture, self.peak_details)
        self.peak_details.add_widget(details)
        self.peak_map = details.where
        self.layout_.addWidget(self.peak_details)
        self.peak_table.itemSelectionChanged.connect(lambda: self._show_peak(self.peak_table.currentRow(), report))
        self.peak_details.expandedChanged.connect(lambda _open: self._show_peak(self.peak_table.currentRow(), report))
        self.peak_table.selectRow(next((index for index, peak in enumerate(peaks) if not peak.get("caveat")), 0))
        for ring in report.get("rings") or ():
            text, detail = ring_summary(ring)
            line = label(text, self)
            line.setToolTip(detail)
            self.layout_.addWidget(line)
        self.layout_.addWidget(label(
            "Sizes are Scherrer lower bounds. Phases are not assigned automatically: ask the AI or compare "
            "q with the lines of your material.", self, role="muted",
        ))

    def _details_section(self, name: str) -> AdvancedSection:
        """Fit details of the selected row: closed until opened, then open for the next reports too."""
        section = AdvancedSection(tr("Fit details"), "", self, expanded=GuidedResultsPanel.details_open)
        section.setObjectName(name)
        section.toggle_button.setToolTip(tr("How the selected row was fitted: the fit on its points, every parameter "
                                            "with its error, the settings and warnings"))
        section.expandedChanged.connect(lambda on: setattr(GuidedResultsPanel, "details_open", bool(on)))
        return section

    def _show_peak(self, row: int, report: dict) -> None:
        """Fit details of the selected peak (drawn only while they are open: the q map takes a moment)."""
        peaks = report.get("peaks") or []
        if self.peak_details is None or not self.peak_details.is_expanded() or not 0 <= row < len(peaks):
            return
        self.peak_details.findChild(PeakDetails).show_peak(peaks[row], report, map_width=PEAK_MAP_PX)

    def _picture(self, target: QLabel, rings: list, width: int) -> bool:
        try:
            png = self._automation().preview_png(width, rings=rings)
        except Exception:
            png = None
        pixmap = QPixmap()
        if png and pixmap.loadFromData(png):
            target.setPixmap(pixmap.scaledToWidth(min(pixmap.width(), width), Qt.SmoothTransformation))
            return True
        target.setText("No q map for this frame.")
        return False

    def _checks(self, report: dict) -> None:
        self.layout_.addWidget(_heading("Checks", self))
        found = 0
        for peak in report.get("peaks") or ():
            if peak.get("caveat"):
                self.layout_.addWidget(badge(f"q = {peak['q']:.4g} Å⁻¹: {peak['caveat']}", WARN, self))
                found += 1
        for text, detail in coverage_notes(report.get("rings") or ()):
            note = badge(text, NOTE, self)
            note.setToolTip(detail)
            self.layout_.addWidget(note)
            found += 1
        if any("shadowed" in str(peak.get("orientation") or "") for peak in report.get("peaks") or ()):
            self.layout_.addWidget(badge(
                "One of the two standard sectors lies in a shadow at some peaks: those peaks are compared "
                "with the other sector only.", NOTE, self,
            ))
            found += 1
        if not found:
            self.layout_.addWidget(badge("No artefacts, shadows or unmeasured ranges that change the results.", GOOD, self))
        self.q_map = QLabel(self)
        self.q_map.setObjectName("guidedQMap")
        self.q_map.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        if self._picture(self.q_map, ring_overlays(report), Q_MAP_PX):
            self.layout_.addWidget(self.q_map)
            self.layout_.addWidget(label(
                "The analysed rings on the q map: solid white = measured, orange = in a shadow, red dashed = "
                "not measured (detector gaps, the missing wedge next to qz).", self, role="muted",
            ))

    def _geometry(self, report: dict) -> None:
        self.layout_.addWidget(_heading("Geometry", self))
        geometry = report.get("geometry") or {}
        quality = report.get("calibration_quality") or {}
        if not geometry.get("distance_mm"):
            self.layout_.addWidget(badge("No geometry yet: nothing can be put in q.", WARN, self))
        else:
            assessment = str(quality.get("assessment") or "")
            color = GOOD if assessment.startswith(("good", "a saved")) or not assessment else NOTE if assessment.startswith("usable") else WARN
            self.layout_.addWidget(badge(assessment or "Instrument profile of this detector", color, self))
            centre = geometry.get("beam_center_px") or [None, None]
            form = QFormLayout()
            for name, value in (
                ("From", geometry.get("source") or "instrument profile"),
                ("Distance", f"{geometry.get('distance_mm')} mm"),
                ("Beam centre", f"({centre[0]}, {centre[1]}) px"),
                ("Wavelength", f"{geometry.get('wavelength_A')} Å"),
                ("Incidence angle αi", f"{geometry.get('incidence_deg')}°"),
            ):
                form.addRow(name, label(str(value), self))
            self.layout_.addLayout(form)
            confirmed = assessment.startswith("good") and quality.get("lines_checked")
            for warning in quality.get("warnings") or ():
                text = f"Note: {warning}" + (" — the line check above already confirms the calibration." if confirmed else "")
                self.layout_.addWidget(label(text, self, role="muted"))
        for item in report.get("decisions") or ():
            if item["what"] in GEOMETRY_DECISIONS:
                self.layout_.addWidget(label(f"• {item['what']}: {item['decision']} — {item['why']}", self, role="muted"))

    def _series(self, report: dict) -> None:
        self.layout_.addWidget(_heading("Series", self))
        frames = report.get("frames") or {}
        first, summed = int(frames.get("first") or 1), int(frames.get("summed") or 1)
        self.layout_.addWidget(label(
            f"A series of {frames.get('total')} frames: these results are frames {first}–{first + summed - 1} "
            "(the final state).", self,
        ))
        compare = QPushButton("Compare with the Start of the Series", self)
        compare.setObjectName("guidedCompareButton")
        compare.clicked.connect(self.compareRequested)
        self.layout_.addWidget(compare, 0, Qt.AlignLeft)
        if self.start_report is None or not self.start_report.get("ok"):
            return
        changes = sorted(series_changes(self.start_report, report), key=lambda item: item["change"] == "present at both")
        rows = [[f"{item['q']:.4g}", item["start"], item["end"], item["change"]] for item in changes]
        headers = (("q (Å⁻¹)", ""), ("at the start", "Frames 1–10 summed."), ("at the end", "The frames in the results above."),
                   ("change", "Reliable peaks only. Peaks closer than half their width are the same line; "
                               "it shifted if it moved by more than 0.3 % in q. \"weak\": a tentative peak."))
        series_table = table(headers, rows, self, "guidedSeriesTable")
        series_table.setMinimumHeight(min(300, 60 + 30 * len(rows)))
        self.layout_.addWidget(series_table)

    def _report(self, report: dict) -> None:
        self.layout_.addWidget(_heading("Report", self))
        save = QPushButton("Save Report…", self)
        save.setObjectName("guidedSaveReport")
        save.setToolTip("A web page with the pictures (opens in any browser), or the text as Markdown.")
        save.clicked.connect(self.saveRequested)
        self.layout_.addWidget(save, 0, Qt.AlignLeft)
        self.report_view = QTextBrowser(self)
        self.report_view.setObjectName("guidedReport")
        self.report_view.setMarkdown(report_markdown(report, self.start_report))
        self.report_view.setMinimumHeight(240)
        self.report_view.hide()
        toggle = QPushButton("Show the Full Text", self)
        toggle.setCheckable(True)
        toggle.toggled.connect(self.report_view.setVisible)
        toggle.toggled.connect(lambda on: toggle.setText("Hide the Full Text" if on else "Show the Full Text"))
        self.layout_.addWidget(toggle, 0, Qt.AlignLeft)
        self.layout_.addWidget(self.report_view)


__all__ = ["GuidedResultsPanel", "automatic_outcome"]
