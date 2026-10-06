"""Words and small widgets of the guided page: steps, questions, plain-language labels, saving.

The tables and the words in them (peaks, rings, coverage) are in ``guided_tables`` and are named here too.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QSize, Qt, QUrl
from PyQt5.QtGui import QDesktopServices, QFont, QPixmap
from PyQt5.QtWidgets import QFileDialog, QLabel, QLayout, QSizePolicy, QWidget

from src.gimap.app.presentation.components import show_toast
from src.gimap.app.presentation.i18n import current_language, tr

from ..application import energy_from_notes, incidence_from_notes, pixel_size_from_notes
from ..application.operations import HALVES_NAMES, halves_label
from .guided_tables import (  # named here as before (``__all__``)
    PEAK_COLUMNS,
    coverage_notes,
    listed,
    peak_rows,
    ring_summary,
    short_orientation,
    table,
    trust,
)

STEPS = (("data", "① Data"), ("geometry", "② Geometry"), ("check", "③ Check"), ("results", "④ Results"), ("report", "⑤ Report"))
GOOD, NOTE, WARN = "success", "info", "warning"
"""Badge roles (``gimapRole``; the theme gives each its colour in light and dark)."""
BADGE_ROLES = ("success", "info", "warning", "error")
OPTION_FIELDS = {
    "incidence_deg": ("Incidence angle αi (°)", "e.g. 0.2"),
    "energy_kev": ("X-ray energy (keV)", "e.g. 12.4"),
    "pixel_size_um": ("Pixel size (µm)", "Pilatus 172, Eiger 75, Lambda 55"),
    "calibration": ("Calibration file", "an image of a standard, or a .poni file"),
    "standard": ("Standard in that image", "agbh, lab6, ceo2 or lab6_ceo2"),
}
HALVES_DONE = "halves: "
"""How ``set_halves`` reports a success (``AnalyzeAutomation.set_halves``): ``halves: <side>``."""
ATTENTION_TITLES = {"model": "Model choice", "fit": "Fit", "beam centre": "Beam centre"}
"""Readable titles of the points a run asks a person to check (the report's item keys)."""
RUN_TEXT, RUN_ANYWAY_TEXT = "Run Automatic Analysis", "Run as GIWAXS Anyway"
GISAXS_NOTE = (
    "Analyze reads this frame as small-angle scattering (GISAXS). This guided analysis is for GIWAXS "
    "(crystal peaks and their orientation). For sizes and spacings of nanostructures use Expert View "
    "(horizontal and vertical cuts in Analyze), then Fitting."
)
EXPORT_FOLDER = "gimap_analysis"
_SAVE_FOLDERS: dict[str, str] = {}
"""The folder chosen last for the data of a folder, for this session."""
IDLE_TEXT = "No AI needed: the standard procedure, each decision with its reason."
RUN_MESSAGE = "Working… (finding a calibration can take a minute)"
GEOMETRY_MESSAGE = "Looking for a calibration near the data…"
START_MESSAGE = "Analysing the start of the series…"
OTHER_FRAMES_TEXT = "Results are for frames {a}–{b}; run again for this frame"
"""The automatic analysis's status while Analyze shows other frames of the series its results are for."""
CHANGE_TEXTS = {
    "appeared": "{n} appeared", "disappeared": "{n} disappeared", "shifted": "{n} shifted",
    "grew": "{n} grew", "faded": "{n} faded", "moved?": "{n} moved?",
}
"""How many lines changed how from the start to the end of a series (the first word of ``series_changes``)."""
NOTES_FOUND = (("αi = {value}°", incidence_from_notes), ("energy = {value} keV", energy_from_notes),
               ("pixel = {value} µm", pixel_size_from_notes))
TECHNIQUES = ("gisaxs", "giwaxs")


def label(text: str, parent: QWidget, *, role: str = "", bold: bool = False) -> QLabel:
    """A wrapped line of plain text that can be selected and copied with the mouse (a value, a sentence)."""
    label = QLabel(text, parent)
    label.setWordWrap(True)
    label.setTextFormat(Qt.PlainText)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    if role:
        label.setProperty("gimapRole", role)
    if bold:
        font = label.font()
        font.setBold(True)
        label.setFont(font)
    return label


def clear_layout(layout: QLayout) -> None:
    """Remove and delete everything in ``layout``, nested layouts included."""
    while layout.count():
        item = layout.takeAt(0)
        widget, inner = item.widget(), item.layout()
        if widget is not None:
            widget.setParent(None)
            widget.deleteLater()
        elif inner is not None:
            clear_layout(inner)
            inner.deleteLater()


def badge(text: str, role: str, parent: QWidget) -> QLabel:
    """A short verdict in semi-bold, coloured by its role: ``success``, ``info`` or ``warning`` (``GOOD``, ``NOTE``, ``WARN``)."""
    label = QLabel(text, parent)
    label.setProperty("gimapRole", role if role in BADGE_ROLES else NOTE)
    label.setTextFormat(Qt.PlainText)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)  # a verdict with its numbers can be copied
    font = label.font()
    font.setWeight(QFont.DemiBold)
    label.setFont(font)
    label.setWordWrap(True)
    return label


def attention_title(item: dict) -> str:
    """The title of a point a run asks about, in words (not the report's key)."""
    key = str(item.get("item") or "")
    title = ATTENTION_TITLES.get(key, key[:1].upper() + key[1:])
    return tr(title)


def step_summary(tool: str, arguments, summary: str, language: str) -> str:
    """A tool's one-line summary with the halves in words ("halves: both_abs" → "Both halves on |qy|").

    Only a success is rewritten: a failed ``set_halves`` keeps its error ("failed: …", "invalid input: …").
    """
    summary = str(summary or "")
    if tool == "set_halves" and summary.startswith(HALVES_DONE):
        return halves_label(summary[len(HALVES_DONE):].strip(), language)
    if tool == "choose_halves":
        side, sep, reason = summary.partition(": ")
        if sep and side in HALVES_NAMES:
            return f"{halves_label(side, language)}: {reason}"
    return summary


def notes_found(notes: str) -> str:
    """What the beamtime notes give (αi, energy, pixel size), in one line; empty when nothing."""
    found = []
    for template, read in NOTES_FOUND:
        value = read(notes)
        if value is not None:
            found.append(tr(template).format(value=f"{value:g}"))
    return tr("Found in the notes: {found}").format(found=listed(found)) if found else ""


def changes_summary(rows) -> str:
    """“2 appeared, 1 shifted”: how the lines changed from the start to the end of a series."""
    kinds: dict[str, int] = {}
    for row in rows:
        if row["change"] != "present at both":
            kind = row["change"].split(" ")[0]
            kinds[kind] = kinds.get(kind, 0) + 1
    return listed(
        tr(CHANGE_TEXTS[kind]).format(n=count) if kind in CHANGE_TEXTS else f"{count} {kind}"
        for kind, count in kinds.items()
    ) or tr("no line changed")


def detected_technique(status: dict, calibration_given: bool) -> Optional[str]:
    """The technique Analyze on Auto detected for the frame it shows (``status()``), for the procedure.

    So a run analyses what Analyze shows without switching the mode (a switch would become the saved
    mode too). ``None`` when a mode is chosen in Analyze (the procedure follows it), the frame is not
    classified yet (no geometry), or a calibration given will replace the geometry it was classified with.
    """
    kind = status.get("measurement")
    return kind if not calibration_given and status.get("mode") == "auto" and kind in TECHNIQUES else None


def other_frames(report: Optional[dict], shown: Optional[tuple]) -> Optional[tuple[int, int]]:
    """(first, last) frame of the series ``report`` is for, when Analyze shows ``shown`` = (first frame,
    frames summed) of that file and they differ; ``None`` when they agree, it is no series, or unknown."""
    frames = (report or {}).get("frames") or {}
    try:
        total = int(frames.get("total") or 1)
        first, summed = int(frames.get("first") or 1), int(frames.get("summed") or 1)
    except (TypeError, ValueError):
        return None
    if total <= 1 or shown is None or tuple(shown) == (first, summed):
        return None
    return first, first + summed - 1


def answerable(attention) -> list[dict]:
    """The questions a person answers in a field of the Results step: one per option the run reads.

    Points that share an option share its one field (GISAXS without αi asks about the incidence angle
    and the Yoneda band, both answered by αi), so every count matches the fields shown.
    """
    questions, seen = [], set()
    for item in attention or ():
        option = item.get("option")
        if option in OPTION_FIELDS and option not in seen:
            seen.add(option)
            questions.append(item)
    return questions


def to_check(attention) -> list[dict]:
    """The other points: advice to look at (the model, the fit, the beam centre …), no field to fill in."""
    return [item for item in attention or () if item.get("option") not in OPTION_FIELDS]


def run_status(report: dict, questions: int) -> str:
    """The line under Run: what a run found and where to look (``questions``: the answer fields shown below)."""
    points = to_check(report.get("needs_attention"))
    if report.get("failed"):  # ended early: the frame changed during the run
        return tr("The analysis stopped: {message}").format(message=report["failed"])
    if report.get("stopped"):
        return tr("Stopped by you — what was found so far is in the Results tab.")
    if report.get("ok"):
        text = tr("Done — see the Results tab.")
        gap = "" if current_language() == "zh" else " "  # Chinese sentences follow each other without a space
        if questions:
            return text + gap + tr("{n} question(s) below.").format(n=questions)
        if points:
            return text + gap + tr("{n} point(s) to check there.").format(n=len(points))
        return text
    if questions:
        return tr("Needs your answers below before it can give results.")
    return tr("No results: the Results tab says why.")


class ScaledPixmapLabel(QLabel):
    """A picture that shrinks with its column and never grows beyond its own size.

    A plain ``QLabel`` with a pixmap asks for the pixmap's width as its minimum, which pushes a narrow
    panel off the right edge. This one keeps the source picture, asks for nothing as a minimum and draws
    the source scaled to ``min(source width, width())``; ``heightForWidth`` keeps the aspect ratio. The
    hints depend only on the source, so rescaling in ``resizeEvent`` never feeds back into the layout.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._source: Optional[QPixmap] = None
        policy = QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

    def setPixmap(self, pixmap: QPixmap) -> None:  # noqa: N802 - Qt API
        self._source = QPixmap(pixmap) if pixmap is not None and not pixmap.isNull() else None
        if self._source is None:
            super().setPixmap(QPixmap())
        else:
            self._rescale(force=True)
        self.updateGeometry()

    def setText(self, text: str) -> None:  # noqa: N802 - Qt API
        self._source = None
        super().setText(text)
        self.updateGeometry()

    def source_pixmap(self) -> Optional[QPixmap]:
        return self._source

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API
        return QSize(0, 0)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        if self._source is None:
            return super().sizeHint()
        return QSize(self._source.width(), self._source.height())

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt API
        return self._source is not None or super().hasHeightForWidth()

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt API
        if self._source is None:
            return super().heightForWidth(width)
        shown = min(self._source.width(), max(1, int(width)))
        return int(round(self._source.height() * shown / max(1, self._source.width())))

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self, *, force: bool = False) -> None:
        if self._source is None:
            return
        width = min(self._source.width(), max(1, self.width()))
        current = self.pixmap()
        if not force and current is not None and not current.isNull() and current.width() == width:
            return
        super().setPixmap(self._source.scaledToWidth(width, Qt.SmoothTransformation))


# -- saving next to the data -------------------------------------------------------------------


def _folder_key(folder: Path) -> str:
    return os.path.normcase(os.path.abspath(str(folder)))


def proposed_save_path(frame, suffix: str) -> str:
    """Where to save ``<stem>_<suffix>`` of ``frame``: next to the data, in its ``gimap_analysis`` folder when it exists.

    The folder chosen last for data of the same folder (this session) comes first.
    """
    if not frame:
        return f"frame_{suffix}"
    source = Path(str(frame))
    name = f"{source.stem or 'frame'}_{suffix}"
    remembered = _SAVE_FOLDERS.get(_folder_key(source.parent))
    if remembered and Path(remembered).is_dir():
        return str(Path(remembered) / name)
    exports = source.parent / EXPORT_FOLDER
    return str((exports if exports.is_dir() else source.parent) / name)


def remember_save_folder(frame, path: str) -> None:
    """The next save for data of ``frame``'s folder starts where ``path`` was saved."""
    if frame and path:
        _SAVE_FOLDERS[_folder_key(Path(str(frame)).parent)] = str(Path(path).parent)


def ask_save_path(parent: QWidget, title: str, frame, suffix: str, filters: str) -> tuple[str, str]:
    """``(path, chosen filter)`` from a save dialog that starts next to the data; ``("", "")`` when cancelled."""
    path, chosen = QFileDialog.getSaveFileName(parent, tr(title), proposed_save_path(frame, suffix), filters)
    if path:
        remember_save_folder(frame, path)
    return path or "", chosen or ""


def _toast_parent(widget: QWidget) -> QWidget:
    window = widget.window() if widget is not None else None
    return window if window is not None else widget


def saved_toast(widget: QWidget, path) -> None:
    """“Saved <name>” with Open Folder (the folder of ``path``)."""
    folder = Path(str(path)).parent
    show_toast(
        _toast_parent(widget), tr("Saved {name}").format(name=Path(str(path)).name), level="ok",
        action=(tr("Open Folder"), lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))),
    )


def save_failed_toast(widget: QWidget, path, error) -> None:
    """A save that did not work, with the reason (never an error dialog)."""
    show_toast(
        _toast_parent(widget), tr("{name} could not be saved: {error}").format(name=Path(str(path)).name, error=error),
        level="error", timeout_ms=9000,
    )


__all__ = [
    "ATTENTION_TITLES", "BADGE_ROLES", "CHANGE_TEXTS", "EXPORT_FOLDER", "GEOMETRY_MESSAGE", "GISAXS_NOTE", "GOOD",
    "IDLE_TEXT", "NOTE", "OPTION_FIELDS", "OTHER_FRAMES_TEXT", "PEAK_COLUMNS", "RUN_ANYWAY_TEXT", "RUN_MESSAGE",
    "RUN_TEXT", "START_MESSAGE", "STEPS", "WARN", "ScaledPixmapLabel", "answerable", "ask_save_path",
    "attention_title", "badge", "changes_summary", "clear_layout", "coverage_notes", "detected_technique", "label",
    "listed", "notes_found", "other_frames", "peak_rows", "proposed_save_path", "remember_save_folder", "ring_summary",
    "run_status", "save_failed_toast", "saved_toast", "short_orientation", "step_summary", "table", "to_check", "trust",
]
