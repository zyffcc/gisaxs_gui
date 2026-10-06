"""What the automatic analysis found, for the Results tab of the Analyze workspace.

Most useful first: a one-line outcome, the peaks (a table whose headers say how each value is
obtained; **Fit details**, closed until opened, show how the selected peak was fitted and where it
lies on the q map), then the checks (artefacts, shadow, missing wedge, the analysed rings on the
q map), the geometry and its quality, the series comparison and the report. Every picture is drawn
by Analyze from the frame on screen. Everything is composed in the interface language (numbers and units
as they are); ``refresh_language`` composes it again after a switch, with the same rows selected.
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
from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_to, current_language, tr, trf

from ..application import ring_overlays, series_changes
from .guided_details import PeakDetails, SolutionDetails
from .guided_gisaxs import GisaxsResults, outcome_summary
from .guided_report import report_markdown
from .guided_tables import PEAK_LEGEND, peak_table, selected_row
from .guided_text import (
    GOOD,
    NOTE,
    WARN,
    ScaledPixmapLabel,
    answerable,
    attention_title,
    badge,
    clear_layout,
    coverage_notes,
    label,
    ring_summary,
    table,
    to_check,
)
from .guided_words import words

PEAK_MAP_PX = 460
Q_MAP_PX = 520
GEOMETRY_DECISIONS = {
    "geometry": "Geometry", "calibration": "Calibration", "calibration candidates": "Calibration candidates",
    "calibration standard": "Calibration standard", "energy": "X-ray energy", "incidence angle": "Incidence angle αi",
    "pixel size": "Pixel size",
}
"""The decisions about the geometry shown under it: the report's key and its title (in the interface language)."""
FRAME_ACTIONS = ("guidedRefineFit", "guidedSaveReport", "guidedCompareButton")
"""Buttons that act on the frame shown in Analyze: off while the results are another frame's (with Show in Fitting)."""
SERIES_HEADERS = (("q (Å⁻¹)", ""), ("at the start", "Frames 1–10 summed."), ("at the end", "The frames in the results above."),
                  ("change", "Reliable peaks only. Peaks closer than half their width are the same line; it shifted if it "
                             "moved by more than 0.3 % in q. \"weak\": a tentative peak."))
PROFILE_SOURCE = "instrument profile '"
"""How the pipeline names a geometry from a profile: ``instrument profile '<name>'``."""
SHIFTED = "shifted "
"""How ``series_changes`` starts the change of a line that moved: ``shifted +0.012 Å⁻¹ (+0.5%) from 1.23``."""
KEPT_PROFILE = ("kept the instrument profile '", "' this detector already has")
"""How the pipeline says it kept a detector's saved profile (the ``geometry`` decision)."""
MOVED, MOVED_WHY = "moved? ", ": one line shifting further than its width, or one line replacing another"
"""``moved? 3.899 → 3.884 (-0.4%): one line …``: a line gone next to a new one (``series_changes``)."""


def giwaxs_summary(report: dict) -> str:
    peaks, rings = report.get("peaks") or [], report.get("rings") or []
    reliable = sum(1 for peak in peaks if not peak.get("caveat"))
    return tr("{peaks} peaks ({reliable} reliable), {rings} rings analysed").format(
        peaks=len(peaks), reliable=reliable, rings=len(rings))


def sentence(text: str) -> str:
    """``text`` with the full stop of the interface language."""
    return text + ("。" if current_language() == "zh" else ".")


def _separator() -> str:
    return "；" if current_language() == "zh" else "; "


def technique_line(report: dict) -> str:
    """“GISAXS — …” / “GIWAXS — …” / “Geometry — …”: what was found, with what the run did (no full stop).

    A geometry-only run (Find Geometry) analysed no technique: its line is the calibration's assessment.
    """
    procedure = report.get("procedure")
    if procedure == "gisaxs":
        return f"GISAXS — {outcome_summary(report)}"
    if procedure == "geometry":
        quality = (report.get("calibration_quality") or {}).get("assessment") or "from the instrument profile"
        return f"{tr('Geometry')} — {words(quality)}"
    return f"GIWAXS — {giwaxs_summary(report)}"


def automatic_outcome(report: dict) -> tuple[str, str]:
    """(state, one line) of an automatic analysis, for the Results step of Analyze.

    ``warn`` when there are questions to answer in the Results step (the workspace then shows it),
    after Find Geometry too (αi); points that only need a look (the model, the fit …) keep ``ok``.
    """
    attention = report.get("needs_attention") or []
    questions, points = answerable(attention), to_check(attention)
    if report.get("stopped"):
        return "warn", tr("Stopped by you; what was found so far is kept (Results tab).")
    if not report.get("ok"):
        if questions:
            return "warn", tr("Needs {n} answer(s) only you can give (Results step).").format(n=len(questions))
        return "warn", tr("No results yet: see the Results tab for why.")
    text = technique_line(report)
    if questions:
        return "warn", sentence(text + _separator() + tr("{n} question(s)").format(n=len(questions)))
    if points:
        return "ok", sentence(text + _separator() + tr("{n} point(s) to check").format(n=len(points)))
    return "ok", sentence(text)


def _heading(text: str, parent: QWidget) -> QLabel:
    heading = QLabel(tr(text), parent)
    heading.setProperty("gimapInspectorTitle", True)
    return heading


def change_text(change: str) -> str:
    """How a line changed along the series, in the interface language (the numbers of a shift as they are)."""
    if change.startswith(SHIFTED):
        move, _sep, origin = change[len(SHIFTED):].partition(" from ")
        return trf("shifted {move} from {q}", move=move, q=origin) if origin else change
    if change.startswith(MOVED) and change.endswith(MOVED_WHY):
        return trf("moved? {move}: one line shifting further than its width, or one line replacing another",
                   move=change[len(MOVED):-len(MOVED_WHY)])
    return tr(change)


def source_text(source) -> str:
    """Where the geometry came from, in the interface language (a file or profile name as it is)."""
    source = str(source or "instrument profile")
    if source.startswith(PROFILE_SOURCE) and source.endswith("'"):
        return trf("instrument profile “{name}”", name=source[len(PROFILE_SOURCE):-1])
    return tr(source)


def decision_text(decision) -> str:
    """A decision of the run in the interface language (a profile's name as it is)."""
    decision = str(decision or "")
    start, end = KEPT_PROFILE
    if decision.startswith(start) and decision.endswith(end) and len(decision) > len(start) + len(end):
        return trf("kept the instrument profile “{name}” this detector already has", name=decision[len(start):-len(end)])
    return words(decision)


def _point(item: dict) -> str:
    """“Title: why” of a point a run asks about, in the interface language."""
    return trf("{title}: {why}", title=attention_title(item), why=words(item.get("why")))


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
        self.actions_enabled = True
        """Whether the buttons that act on the frame in Analyze work (off while another frame is shown)."""
        self.layout_ = QVBoxLayout(self)
        self.layout_.setContentsMargins(0, 0, 0, 0)
        self.layout_.setSpacing(10)
        self.hide()

    # -- the whole report -------------------------------------------------------------

    def show_report(self, report: dict, start_report: Optional[dict] = None) -> None:
        self.report, self.start_report = report, start_report
        self._dispose()
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
        if report.get("ok") and report.get("procedure") not in ("gisaxs", "geometry"):  # Find Geometry analyses no rings
            self._checks(report)
        self._geometry(report)
        frames = report.get("frames") or {}
        if report.get("ok") and int(frames.get("total") or 1) > 1:
            self._series(report)
        self._report(report)
        self.layout_.addStretch(1)
        if current_language() != DEFAULT_LANGUAGE:  # built after the window was translated
            apply_to(self, current_language())
        self.set_actions_enabled(self.actions_enabled)
        self.show()

    def _dispose(self) -> None:
        """Before the sections are rebuilt (a new report, a switch of the language): the plots let go of the theme,
        and a q map that was never put in the layout (no picture) is deleted with the rest."""
        try:
            if self.gisaxs is not None:
                self.gisaxs.dispose()
            details = self.peak_details.findChild(PeakDetails) if self.peak_details is not None else None
            if details is not None:
                details.dispose()
            if self.q_map is not None and self.layout_.indexOf(self.q_map) < 0:
                self.q_map.setParent(None)
                self.q_map.deleteLater()
        except RuntimeError:  # already gone
            pass

    def set_actions_enabled(self, enabled: bool) -> None:
        """Refine in Fitting, Show in Fitting, Save Report and Compare: only while these results are the frame's in Analyze."""
        self.actions_enabled = bool(enabled)
        for button in self.findChildren(QPushButton):
            if button.objectName() in FRAME_ACTIONS:
                button.setEnabled(self.actions_enabled)
        for details in self.findChildren(SolutionDetails):
            details.set_allowed(self.actions_enabled)

    def _outcome(self, report: dict) -> None:
        attention = report.get("needs_attention") or []
        questions, points = answerable(attention), to_check(attention)
        peaks = report.get("peaks") or []
        rings = report.get("rings") or []
        if report.get("stopped"):
            done = len(report.get("steps") or [])
            self.layout_.addWidget(badge(tr(
                "Stopped by you after {done} steps: what was found so far is below "
                "({peaks} peaks, {rings} rings analysed).").format(done=done, peaks=len(peaks), rings=len(rings)),
                NOTE, self,
            ))
        elif report.get("ok"):
            text = sentence(technique_line(report))
            if questions:
                gap = "" if current_language() == "zh" else " "  # Chinese sentences follow each other without a space
                text += gap + tr("{n} question(s) in the Results step.").format(n=len(questions))
            self.layout_.addWidget(badge(text, GOOD, self))
            if points:
                self.layout_.addWidget(label(tr("{n} point(s) to check — see below").format(n=len(points)), self, role="muted"))
            for item in points:
                self.layout_.addWidget(badge(_point(item), NOTE, self))
                if item.get("hint"):
                    self.layout_.addWidget(label(words(item["hint"]), self, role="muted"))
        else:
            self.layout_.addWidget(badge(tr(
                "The analysis needs your answers in the Results step before it can give results." if questions else
                "No results: what stopped the analysis is listed below."), WARN, self,
            ))
            for item in attention:
                self.layout_.addWidget(label("• " + _point(item), self, role="muted"))

    def _peaks(self, report: dict) -> None:
        peaks = report.get("peaks") or []
        self.layout_.addWidget(_heading("Peaks", self))
        self.peak_table = peak_table(peaks, self)  # compact: fits the narrow Results tab; tooltips say the rest
        self.peak_table.setMinimumHeight(min(360, 60 + 30 * len(peaks)))
        self.layout_.addWidget(self.peak_table)
        self.layout_.addWidget(label(tr(PEAK_LEGEND), self, role="muted"))
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
        self.layout_.addWidget(label(tr(
            "Sizes are Scherrer lower bounds. Phases are not assigned automatically: ask the AI or compare "
            "q with the lines of your material."), self, role="muted",
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
        target.setText(tr("No q map for this frame."))
        return False

    def _checks(self, report: dict) -> None:
        self.layout_.addWidget(_heading("Checks", self))
        found = 0
        for peak in report.get("peaks") or ():
            if peak.get("caveat"):
                self.layout_.addWidget(badge(trf("q = {q} Å⁻¹: {caveat}", q=f"{peak['q']:.4g}", caveat=tr(peak["caveat"])),
                                             WARN, self))
                found += 1
        for text, detail in coverage_notes(report.get("rings") or ()):
            note = badge(text, NOTE, self)
            note.setToolTip(detail)
            self.layout_.addWidget(note)
            found += 1
        if any("shadowed" in str(peak.get("orientation") or "") for peak in report.get("peaks") or ()):
            self.layout_.addWidget(badge(tr(
                "One of the two standard sectors lies in a shadow at some peaks: those peaks are compared "
                "with the other sector only."), NOTE, self,
            ))
            found += 1
        if not found:
            self.layout_.addWidget(badge(tr("No artefacts, shadows or unmeasured ranges that change the results."), GOOD, self))
        self.q_map = ScaledPixmapLabel(self)  # shrinks with the panel instead of clipping the text beside it
        self.q_map.setObjectName("guidedQMap")
        self.q_map.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        if self._picture(self.q_map, ring_overlays(report), Q_MAP_PX):
            self.layout_.addWidget(self.q_map)
            self.layout_.addWidget(label(tr(
                "The analysed rings on the q map: solid white = measured, orange = in a shadow, red dashed = "
                "not measured (detector gaps, the missing wedge next to qz)."), self, role="muted",
            ))
        else:  # outside the layout it would be drawn over the outcome line at the top
            self.q_map.hide()

    def _geometry(self, report: dict) -> None:
        self.layout_.addWidget(_heading("Geometry", self))
        geometry = report.get("geometry") or {}
        quality = report.get("calibration_quality") or {}
        if not geometry.get("distance_mm"):
            self.layout_.addWidget(badge(tr("No geometry yet: nothing can be put in q."), WARN, self))
        else:
            assessment = str(quality.get("assessment") or "")
            color = GOOD if assessment.startswith(("good", "a saved")) or not assessment else NOTE if assessment.startswith("usable") else WARN
            self.layout_.addWidget(badge(words(assessment) or tr("Instrument profile of this detector"), color, self))
            centre = geometry.get("beam_center_px") or [None, None]
            form = QFormLayout()
            for name, value in (
                ("From", source_text(geometry.get("source"))),
                ("Distance", f"{geometry.get('distance_mm')} mm"),
                ("Beam centre", f"({centre[0]}, {centre[1]}) px"),
                ("Wavelength", f"{geometry.get('wavelength_A')} Å"),
                ("Incidence angle αi", f"{geometry.get('incidence_deg')}°"),
            ):
                form.addRow(tr(name), label(str(value), self))
            self.layout_.addLayout(form)
            confirmed = assessment.startswith("good") and quality.get("lines_checked")
            for warning in quality.get("warnings") or ():
                template = ("Note: {warning} — the line check above already confirms the calibration." if confirmed
                            else "Note: {warning}")
                self.layout_.addWidget(label(trf(template, warning=tr(str(warning))), self, role="muted"))
        for item in report.get("decisions") or ():
            if item["what"] in GEOMETRY_DECISIONS:
                self.layout_.addWidget(label("• " + trf("{what}: {decision} — {why}", what=tr(GEOMETRY_DECISIONS[item["what"]]),
                                                        decision=decision_text(item["decision"]), why=words(item["why"])),
                                             self, role="muted"))

    def _series(self, report: dict) -> None:
        self.layout_.addWidget(_heading("Series", self))
        frames = report.get("frames") or {}
        first, summed = int(frames.get("first") or 1), int(frames.get("summed") or 1)
        self.layout_.addWidget(label(trf(
            "A series of {total} frames: these results are frames {first}–{last} (the final state).",
            total=frames.get("total"), first=first, last=first + summed - 1), self,
        ))
        compare = QPushButton(tr("Compare with the Start of the Series"), self)
        compare.setObjectName("guidedCompareButton")
        compare.clicked.connect(self.compareRequested)
        self.layout_.addWidget(compare, 0, Qt.AlignLeft)
        if self.start_report is None or not self.start_report.get("ok"):
            return
        changes = sorted(series_changes(self.start_report, report), key=lambda item: item["change"] == "present at both")
        rows = [[f"{item['q']:.4g}", tr(item["start"]), tr(item["end"]), change_text(item["change"])]
                for item in changes]
        series_table = table(SERIES_HEADERS, rows, self, "guidedSeriesTable", numbers=(0,), single=False)
        series_table.setMinimumHeight(min(300, 60 + 30 * len(rows)))
        self.layout_.addWidget(series_table)

    def _report(self, report: dict) -> None:
        self.layout_.addWidget(_heading("Report", self))
        save = QPushButton(tr("Save Report…"), self)
        save.setObjectName("guidedSaveReport")
        save.setToolTip(tr("A web page with the pictures (opens in any browser), or the text as Markdown."))
        save.clicked.connect(self.saveRequested)
        self.layout_.addWidget(save, 0, Qt.AlignLeft)
        self.report_view = QTextBrowser(self)
        self.report_view.setObjectName("guidedReport")
        self.report_view.setMarkdown(report_markdown(report, self.start_report))
        self.report_view.setMinimumHeight(240)
        self.report_view.hide()
        toggle = QPushButton(tr("Show the Full Text"), self)
        toggle.setObjectName("guidedFullTextToggle")
        toggle.setCheckable(True)
        toggle.toggled.connect(self.report_view.setVisible)
        toggle.toggled.connect(lambda on: toggle.setText(tr("Hide the Full Text") if on else tr("Show the Full Text")))
        self.layout_.addWidget(toggle, 0, Qt.AlignLeft)
        self.layout_.addWidget(self.report_view)

    # -- the interface language -----------------------------------------------------------

    def refresh_language(self) -> None:
        """The report again in the interface language, with the same peak or solution selected and the full
        text as open as it was."""
        if self.report is None:
            return
        peak, solution = selected_row(self.peak_table), selected_row(self.gisaxs.fit_table if self.gisaxs else None)
        toggle = self.findChild(QPushButton, "guidedFullTextToggle")
        full_text = toggle is not None and toggle.isChecked()
        self.show_report(self.report, self.start_report)
        if peak >= 0 and self.peak_table is not None:
            self.peak_table.selectRow(peak)
        if solution >= 0 and self.gisaxs is not None and self.gisaxs.fit_table is not None:
            self.gisaxs.fit_table.selectRow(solution)
        toggle = self.findChild(QPushButton, "guidedFullTextToggle")
        if full_text and toggle is not None:
            toggle.setChecked(True)


__all__ = ["GuidedResultsPanel", "automatic_outcome"]
