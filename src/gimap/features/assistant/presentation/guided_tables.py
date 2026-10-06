"""The tables of the Results tab and the words in them: peaks, orientation and trust verdicts, rings, coverage.

Every table is read-only, copies its rows (Ctrl+C, and Copy Rows / Copy Table on the right button), right-aligns
its numbers and gives its headers and their tooltips in the interface language (numbers and units never change).

The peak table fits the Results tab of a 1280-px window (about 330 px for the table): short headers whose
tooltips give the full name and how the value is obtained, a short verdict per cell (its tooltip is the whole
sentence), no row numbers. ``peak_rows`` keeps the full words of every cell; the saved report has the full table.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import QEvent, QObject, Qt
from PyQt5.QtWidgets import QHeaderView, QSizePolicy, QTableWidget, QTableWidgetItem, QWidget

from src.gimap.app.presentation.components.table_copy import enable_table_copy
from src.gimap.app.presentation.i18n import current_language, tr, trf
from src.gimap.app.presentation.theme import theme_manager

from .guided_words import words

PEAK_COLUMNS = (
    ("q (Å⁻¹)", "Where the peak is: the centre of a Gaussian fitted on a local linear background to the "
                "radial I(q) of the whole detector (all χ)."),
    ("d (Å)", "The lattice spacing of this reflection: d = 2π / q."),
    ("FWHM (Å⁻¹)", "The width of the fitted peak (full width at half maximum). It includes the instrument's "
                   "broadening."),
    ("trust", "Whether this is a crystal peak: spikes (one hot pixel, a streak or a flat-topped box from one detector "
              "row), broad halos (amorphous "
              "order) and weak peaks are flagged."),
    ("size (nm)", "Scherrer: L = 2π·0.9 / FWHM. The instrument's broadening is not removed, so it is a lower "
                  "bound (≥) of the crystallite size."),
    ("in-/out-of-plane", "The net intensity per pixel at this q in the out-of-plane sector (χ ≈ 0°, along the "
                         "surface normal) compared with the in-plane sector (χ ≈ ±90°)."),
)
"""The full name of every column of the peak table and how its value is obtained (the header tooltips).
The verdict first, the orientation last (the column that stretches)."""
PEAK_HEADERS = ("q", "d", "FWHM", "trust", "L", "OOP/IP")
"""The short headers the peak table shows: ``PEAK_COLUMNS`` gives each one's full name and unit in its tooltip,
``PEAK_LEGEND`` the units under the table (with units in the headers the table is wider than the Results tab)."""
PEAK_NUMBERS = (0, 1, 2, 4)
"""Columns of the peak table that hold numbers (right-aligned)."""
PEAK_LEGEND = ("q and FWHM in Å⁻¹, d in Å, L (Scherrer size, a lower bound) in nm. OOP: out-of-plane (χ ≈ 0°), "
               "IP: in-plane (χ ≈ ±90°), *: one sector only. Hover a cell for the whole sentence; the saved report "
               "has the full table.")
EDGE_VERDICT = "at the end of the data: check"
EDGE_NOTE = ("The peak lies within 1.5 widths of the end of the measured q range: its shape and position may be "
             "cut off. Look at I(q) before using it.")
TRUST_WORDS = (("a spike", "artefact (spike)", "spike"), ("a broad halo", "halo, not a crystal peak", "halo"),
               ("weak", "weak, tentative", "weak"), ("its shape could not", "fit failed: check the image", "no fit"))
"""(start of the caveat, the verdict, the short verdict of the compact table)."""
ORIENTATION_WORDS = (
    ("mainly out-of-plane", "OOP"), ("mainly in-plane", "IP"), ("both sectors", "both"),
    ("out-of-plane only", "OOP only"), ("in-plane only", "IP only"), ("not detected", "neither"),
)
"""(start of the sector comparison, its short form in the compact table)."""
ROLE = Qt.UserRole + 7
"""A cell's colour role (``success``, ``warning``): its text colour follows the theme."""


def listed(parts, separator: str = ", ") -> str:
    """``parts`` joined in the interface language (Chinese uses its own comma or semicolon)."""
    if current_language() == "zh":
        separator = "；" if separator.strip() == ";" else "，"
    return separator.join(parts)


# -- the words of the peak table -----------------------------------------------------------------------


def short_orientation(text: str) -> str:
    """The sector comparison in a few words (the full sentence is the tooltip)."""
    if not text:
        return "—"
    if text.startswith("only the") and "shadowed" in text:
        side = "out-of-plane" if "out-of-plane sector is usable" in text else "in-plane"
        return tr(f"{side} only (the other sector is in a shadow)")
    if text.startswith("not comparable"):
        return tr("cannot tell here (shadow / not measured)")
    if text.startswith("not measured"):
        return tr("not measured at this q")
    if text.startswith("only the"):
        return tr(text.split(" sector")[0].replace("only the ", "") + " only measured")
    return tr(text)


def compact_orientation(text: str) -> str:
    """The sector comparison in one or two words for a narrow table: OOP, IP, both …; ``*``: one sector only."""
    if not text:
        return "—"
    if text.startswith("only the"):
        return tr("OOP" if text.startswith("only the out-of-plane") else "IP") + "*"
    if text.startswith(("not comparable", "not measured")):
        return tr("n/a")
    return next((tr(short) for start, short in ORIENTATION_WORDS if text.startswith(start)), tr("check"))


def trust(caveat: str) -> str:
    if not caveat:
        return tr("reliable")
    return next((tr(verdict) for start, verdict, _short in TRUST_WORDS if caveat.startswith(start)), tr("check"))


def compact_trust(caveat: str, at_edge: bool = False) -> str:
    """✓ for a reliable peak, else ⚠ and one word (spike, halo, weak, no fit, edge, check)."""
    if not caveat:
        return "⚠ " + tr("edge") if at_edge else "✓"
    return "⚠ " + next((tr(short) for start, _verdict, short in TRUST_WORDS if caveat.startswith(start)), tr("check"))


def peak_rows(peaks: list[dict]) -> list[list]:
    """The full words of every cell of the peak table (``PEAK_COLUMNS``), each with its whole sentence."""
    rows = []
    for peak in peaks:
        size = peak.get("size_nm")
        orientation = str(peak.get("orientation") or "")
        caveat = str(peak.get("caveat") or "")
        size_text = "—" if size is None else ("≥ " if peak.get("size_is_lower_bound") else "") + f"{size:.3g}"
        at_edge = not caveat and "at_edge" in (peak.get("flags") or ())
        rows.append([
            f"{peak.get('q'):.4g}", f"{peak.get('d_A'):.4g}", f"{peak.get('fwhm'):.3g}",
            (tr(EDGE_VERDICT), tr(EDGE_NOTE)) if at_edge else (trust(caveat), tr(caveat) if caveat else ""),
            (size_text, tr(str(peak.get("size_note") or ""))),
            (short_orientation(orientation), tr(orientation)),
        ])
    return rows


def peak_cells(peaks: list[dict]) -> list[list]:
    """The compact peak table: numbers, ✓ or ⚠ and a word (coloured), OOP / IP …; every cell's tooltip is its
    full words and sentence (``peak_rows``)."""
    cells = []
    for peak, full in zip(peaks, peak_rows(peaks)):
        caveat = str(peak.get("caveat") or "")
        at_edge = not caveat and "at_edge" in (peak.get("flags") or ())
        verdict, why = full[3]
        size, size_note = full[4]
        words, sentence = full[5]
        cells.append([
            full[0], full[1], full[2],
            (compact_trust(caveat, at_edge), _joined(verdict, why), "warning" if caveat or at_edge else "success"),
            (size, size_note),
            (compact_orientation(str(peak.get("orientation") or "")), _joined(words, sentence)),
        ])
    return cells


def _joined(words: str, sentence: str) -> str:
    return words if not sentence or sentence == words else f"{words}\n{sentence}"


def peak_headers() -> list[tuple[str, str]]:
    """(short header, tooltip: the full name and how it is obtained) of the peak table, in the interface language."""
    return [(short, f"{tr(full)}\n{tr(tip)}") for short, (full, tip) in zip(PEAK_HEADERS, PEAK_COLUMNS)]


# -- the rings and what was not measured -------------------------------------------------------------------


def _ranges(rings: list[dict], pick) -> str:
    return listed(trf("{ranges} at q {q}", ranges=pick(ring), q=f"{ring['q']:.3g}") for ring in rings)


def coverage_notes(rings) -> list[tuple[str, str]]:
    """(one line, the per-ring sentences) for the shadow and the missing wedge of the analysed rings."""
    rings = [ring for ring in rings if ring.get("q")]
    notes = []
    shadowed = [ring for ring in rings if ring.get("shadowed")]
    if shadowed:
        where = _ranges(shadowed, lambda ring: " and ".join(f"{a:.0f}–{b:.0f}°" for a, b in ring["shadowed"]))
        notes.append((
            trf("Shadow (orange on the map): |χ| {where} Å⁻¹. The intensity there is far below the diffuse "
                "background, so these pixels count as unmeasured.", where=where),
            "\n".join(words(note) for ring in shadowed for note in ring.get("notes") or () if "shadowed" in note),
        ))
    wedge = [ring for ring in rings if (ring.get("missing") or [[None]])[0][0] == 0.0]
    if wedge:
        where = _ranges(wedge, lambda ring: f"< {ring['missing'][0][1]:.0f}°")
        notes.append((
            trf("Missing wedge (red dashed next to qz): |χ| {where} Å⁻¹ is not measured, so orientations closest "
                "to the surface normal are missing and Herman's f is biased low.", where=where),
            "\n".join(words(note) for ring in wedge for note in ring.get("notes") or () if "missing wedge" in note),
        ))
    return notes


def ring_summary(ring: dict) -> tuple[str, str]:
    """(one plain line, the full sentences) for an analysed ring's orientation."""
    q = f"{ring.get('q'):.4g}"
    detail = "\n".join(words(text) for text in (ring.get("reason"), ring.get("texture"), *(ring.get("notes") or ())) if text)
    texture = words(ring.get("texture"))
    if ring.get("herman") is not None:
        random, f = ring.get("herman_isotropic"), f"{ring['herman']:.2f}"
        if random is not None and abs(random) >= 0.005:
            return trf("Ring q ≈ {q} Å⁻¹: f = {f} (a random ring would give {random} here) — {texture}",
                       q=q, f=f, random=f"{random:.2f}", texture=texture), detail
        return trf("Ring q ≈ {q} Å⁻¹: f = {f} — {texture}", q=q, f=f, texture=texture), detail
    covered = ring.get("weighted_coverage")
    if covered is not None and "is measured at this q" in str(ring.get("reason") or ""):
        gaps = []
        if ring.get("shadowed"):
            gaps.append(trf("in a shadow at |χ| {ranges}", ranges=", ".join(f"{a:.0f}–{b:.0f}°" for a, b in ring["shadowed"])))
        missing = ring.get("missing") or ()
        if missing and missing[0][0] == 0.0:
            gaps.append(trf("missing wedge below {angle}°", angle=f"{missing[0][1]:.0f}"))
        values = {"q": q, "covered": f"{covered:.0%}"}
        if gaps:
            return trf("Ring q ≈ {q} Å⁻¹: orientation not determined — only {covered} of the range Herman's f needs "
                       "is measured ({gaps}). Hover for details.", gaps=listed(gaps, "; "), **values), detail
        return trf("Ring q ≈ {q} Å⁻¹: orientation not determined — only {covered} of the range Herman's f needs "
                   "is measured. Hover for details.", **values), detail
    return trf("Ring q ≈ {q} Å⁻¹: {reason}", q=q, reason=words(ring.get("reason")) or texture), detail


# -- the table widget -----------------------------------------------------------------------------------------


class LastColumnFill(QObject):
    """The last column of a table is as wide as its title and its cells, and fills the room the others leave.

    A stretched last section (``setStretchLastSection``) shrinks below its contents once the other columns
    fill a narrow panel, which clipped the title on both sides (“n-/out-of-pl”) and the words of every cell.
    Here a narrow panel scrolls the table sideways instead, and a wide one still has no empty strip.
    """

    def __init__(self, view: QTableWidget):
        super().__init__(view)
        self._view = view
        header = view.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(view.columnCount() - 1, QHeaderView.Interactive)
        view.viewport().installEventFilter(self)
        self.fit()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API
        if event.type() in (QEvent.Resize, QEvent.Show, QEvent.LayoutRequest):
            self.fit()
        return False

    def fit(self) -> None:
        view = self._view
        header, last = view.horizontalHeader(), view.columnCount() - 1
        others = sum(header.sectionSize(column) for column in range(last))
        wanted = max(header.sectionSizeHint(last), view.sizeHintForColumn(last), view.viewport().width() - others)
        if header.sectionSize(last) != wanted:
            header.resizeSection(last, wanted)


class _RoleColours(QObject):
    """The text colour of the cells with a role (``ROLE``), again after every switch of the theme."""

    def __init__(self, view: QTableWidget):
        super().__init__(view)
        self._view = view
        theme_manager().changed.connect(self.paint)
        self.paint()

    def paint(self, _mode: str = "") -> None:
        try:
            view = self._view
            manager = theme_manager()
            for row in range(view.rowCount()):
                for column in range(view.columnCount()):
                    item = view.item(row, column)
                    role = item.data(ROLE) if item is not None else None
                    if role:
                        item.setForeground(manager.color(role))
        except RuntimeError:  # the table is gone
            return


def table(headers, rows, parent: QWidget, name: str, *, numbers=(), row_numbers: bool = True,
          single: bool = True, copy_headers=()) -> QTableWidget:
    """A read-only table that copies its rows; ``rows`` are lists of text, ``(text, tooltip)`` or
    ``(text, tooltip, role)`` (``success`` / ``warning``: the text in that colour).

    ``headers`` are ``(title, tooltip)``, both shown in the interface language; ``numbers``: the columns whose
    cells are numbers (right-aligned). ``single``: one row at a time (a details pane follows the current row).
    ``copy_headers``: the full column names with units that copied rows carry (short headers fit the panel).
    """
    widget = QTableWidget(len(rows), len(headers), parent)
    widget.setObjectName(name)
    widget.setHorizontalHeaderLabels([tr(header) for header, _tip in headers])
    numeric = Qt.AlignRight | Qt.AlignVCenter
    for column, (_header, tip) in enumerate(headers):
        if tip:
            widget.horizontalHeaderItem(column).setToolTip(tr(tip))
        if column in numbers:
            widget.horizontalHeaderItem(column).setTextAlignment(numeric)
        if column < len(copy_headers) and copy_headers[column]:
            widget.horizontalHeaderItem(column).setData(Qt.UserRole, tr(copy_headers[column]))
    coloured = False
    for row, values in enumerate(rows):
        for column, value in enumerate(values):
            text, tip, role = (tuple(value) + ("", ""))[:3] if isinstance(value, tuple) else (value, "", "")
            item = QTableWidgetItem(text)
            if tip:
                item.setToolTip(tip)
            if column in numbers:
                item.setTextAlignment(numeric)
            if role:
                item.setData(ROLE, role)
                coloured = True
            widget.setItem(row, column, item)
    if coloured:
        _RoleColours(widget)
    widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    widget.verticalHeader().setVisible(row_numbers)
    if headers:
        LastColumnFill(widget)
    widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)  # a narrow panel scrolls the table, not the page
    widget.setMinimumWidth(120)
    widget.setHorizontalScrollMode(QTableWidget.ScrollPerPixel)  # sideways in a narrow panel, smoothly
    widget.setEditTriggers(QTableWidget.NoEditTriggers)
    widget.setSelectionBehavior(QTableWidget.SelectRows)
    widget.setSelectionMode(QTableWidget.SingleSelection if single else QTableWidget.ExtendedSelection)
    enable_table_copy(widget)
    return widget


def peak_table(peaks: list[dict], parent: QWidget, name: str = "guidedPeakTable") -> QTableWidget:
    """The compact peak table (``peak_cells``): one row at a time, since Fit details follow the selected peak."""
    widget = table(peak_headers(), peak_cells(peaks), parent, name, numbers=PEAK_NUMBERS, row_numbers=False,
                   copy_headers=[full for full, _tip in PEAK_COLUMNS])
    widget.setProperty("gimapCompactTable", True)
    return widget


def selected_row(view: Optional[QTableWidget]) -> int:
    """The current row of ``view`` (−1 when there is none or it is gone)."""
    try:
        return view.currentRow() if view is not None else -1
    except RuntimeError:
        return -1


__all__ = [
    "EDGE_NOTE", "EDGE_VERDICT", "LastColumnFill", "ORIENTATION_WORDS", "PEAK_COLUMNS", "PEAK_HEADERS", "PEAK_LEGEND",
    "PEAK_NUMBERS", "ROLE", "TRUST_WORDS", "compact_orientation", "compact_trust", "coverage_notes", "listed",
    "peak_cells", "peak_headers", "peak_rows", "peak_table", "ring_summary", "selected_row", "short_orientation",
    "table", "trust",
]
