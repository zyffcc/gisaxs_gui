"""Words and small widgets of the guided page: steps, questions, plain-language labels, tables."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHeaderView, QLabel, QLayout, QSizePolicy, QTableWidget, QTableWidgetItem, QWidget

STEPS = (("data", "① Data"), ("geometry", "② Geometry"), ("check", "③ Check"), ("results", "④ Results"), ("report", "⑤ Report"))
GOOD, NOTE, WARN = "#2e7d32", "#1e88e5", "#ef6c00"
OPTION_FIELDS = {
    "incidence_deg": ("Incidence angle αi (°)", "e.g. 0.2"),
    "energy_kev": ("X-ray energy (keV)", "e.g. 12.4"),
    "pixel_size_um": ("Pixel size (µm)", "Pilatus 172, Eiger 75, Lambda 55"),
    "calibration": ("Calibration file", "path of an image of a standard or a .poni file"),
    "standard": ("Standard in that image", "agbh, lab6, ceo2 or lab6_ceo2"),
}
RUN_TEXT, RUN_ANYWAY_TEXT = "Run Automatic Analysis", "Run as GIWAXS Anyway"
GISAXS_NOTE = (
    "Analyze reads this frame as small-angle scattering (GISAXS). This guided analysis is for GIWAXS "
    "(crystal peaks and their orientation). For sizes and spacings of nanostructures use Expert View "
    "(horizontal and vertical cuts in Analyze), then Fitting."
)
PEAK_COLUMNS = (
    ("q (Å⁻¹)", "Where the peak is: the centre of a Gaussian fitted on a local linear background to the "
                "radial I(q) of the whole detector (all χ)."),
    ("d (Å)", "The lattice spacing of this reflection: d = 2π / q."),
    ("FWHM (Å⁻¹)", "The width of the fitted peak (full width at half maximum). It includes the instrument's "
                   "broadening."),
    ("in-/out-of-plane", "The net intensity per pixel at this q in the out-of-plane sector (χ ≈ 0°, along the "
                         "surface normal) compared with the in-plane sector (χ ≈ ±90°)."),
    ("size (nm)", "Scherrer: L = 2π·0.9 / FWHM. The instrument's broadening is not removed, so it is a lower "
                  "bound (≥) of the crystallite size."),
    ("trust", "Whether this is a crystal peak: spikes (one hot pixel or a streak), broad halos (amorphous "
              "order) and weak peaks are flagged."),
)


def label(text: str, parent: QWidget, *, role: str = "", bold: bool = False) -> QLabel:
    label = QLabel(text, parent)
    label.setWordWrap(True)
    label.setTextFormat(Qt.PlainText)
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


def badge(text: str, color: str, parent: QWidget) -> QLabel:
    label = QLabel(text, parent)
    label.setStyleSheet(f"color: {color}; font-weight: 600;")
    label.setWordWrap(True)
    return label


def short_orientation(text: str) -> str:
    """The sector comparison in a few words (the full sentence is the tooltip)."""
    if not text:
        return "—"
    if text.startswith("only the") and "shadowed" in text:
        side = "out-of-plane" if "out-of-plane sector is usable" in text else "in-plane"
        return f"{side} only (the other sector is in a shadow)"
    if text.startswith("not comparable"):
        return "cannot tell here (shadow / not measured)"
    if text.startswith("not measured"):
        return "not measured at this q"
    if text.startswith("only the"):
        return text.split(" sector")[0].replace("only the ", "") + " only measured"
    return text


def trust(caveat: str) -> str:
    if not caveat:
        return "reliable"
    for start, label in (("a spike", "artefact (spike)"), ("a broad halo", "halo, not a crystal peak"),
                         ("weak", "weak, tentative"), ("its shape could not", "fit failed: check the image")):
        if caveat.startswith(start):
            return label
    return "check"


def _ranges(rings: list[dict], pick) -> str:
    return ", ".join(f"{pick(ring)} at q {ring['q']:.3g}" for ring in rings)


def coverage_notes(rings) -> list[tuple[str, str]]:
    """(one line, the per-ring sentences) for the shadow and the missing wedge of the analysed rings."""
    rings = [ring for ring in rings if ring.get("q")]
    notes = []
    shadowed = [ring for ring in rings if ring.get("shadowed")]
    if shadowed:
        where = _ranges(shadowed, lambda ring: " and ".join(f"{a:.0f}–{b:.0f}°" for a, b in ring["shadowed"]))
        notes.append((
            f"Shadow (orange on the map): |χ| {where} Å⁻¹. The intensity there is far below the diffuse "
            "background, so these pixels count as unmeasured.",
            "\n".join(note for ring in shadowed for note in ring.get("notes") or () if "shadowed" in note),
        ))
    wedge = [ring for ring in rings if (ring.get("missing") or [[None]])[0][0] == 0.0]
    if wedge:
        where = _ranges(wedge, lambda ring: f"< {ring['missing'][0][1]:.0f}°")
        notes.append((
            f"Missing wedge (red dashed next to qz): |χ| {where} Å⁻¹ is not measured, so orientations closest "
            "to the surface normal are missing and Herman's f is biased low.",
            "\n".join(note for ring in wedge for note in ring.get("notes") or () if "missing wedge" in note),
        ))
    return notes


def ring_summary(ring: dict) -> tuple[str, str]:
    """(one plain line, the full sentences) for an analysed ring's orientation."""
    head = f"Ring q ≈ {ring.get('q'):.4g} Å⁻¹"
    detail = "\n".join(str(text) for text in (ring.get("reason"), ring.get("texture"), *(ring.get("notes") or ())) if text)
    if ring.get("herman") is not None:
        random = ring.get("herman_isotropic")
        reference = f" (a random ring would give {random:.2f} here)" if random is not None and abs(random) >= 0.005 else ""
        return f"{head}: f = {ring['herman']:.2f}{reference} — {ring.get('texture')}", detail
    covered = ring.get("weighted_coverage")
    if covered is not None and "is measured at this q" in str(ring.get("reason") or ""):
        gaps = []
        if ring.get("shadowed"):
            gaps.append("in a shadow at |χ| " + ", ".join(f"{a:.0f}–{b:.0f}°" for a, b in ring["shadowed"]))
        missing = ring.get("missing") or ()
        if missing and missing[0][0] == 0.0:
            gaps.append(f"missing wedge below {missing[0][1]:.0f}°")
        where = f" ({'; '.join(gaps)})" if gaps else ""
        return (
            f"{head}: orientation not determined — only {covered:.0%} of the range Herman's f needs is "
            f"measured{where}. Hover for details.", detail,
        )
    return f"{head}: {ring.get('reason') or ring.get('texture')}", detail


def table(headers, rows, parent: QWidget, name: str) -> QTableWidget:
    """A read-only table; ``rows`` are lists of text or ``(text, tooltip)``."""
    widget = QTableWidget(len(rows), len(headers), parent)
    widget.setObjectName(name)
    widget.setHorizontalHeaderLabels([header for header, _tip in headers])
    for column, (_header, tip) in enumerate(headers):
        if tip:
            widget.horizontalHeaderItem(column).setToolTip(tip)
    for row, values in enumerate(rows):
        for column, value in enumerate(values):
            text, tip = value if isinstance(value, tuple) else (value, "")
            item = QTableWidgetItem(text)
            if tip:
                item.setToolTip(tip)
            widget.setItem(row, column, item)
    widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    widget.horizontalHeader().setStretchLastSection(True)
    widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)  # a narrow panel scrolls the table, not the page
    widget.setMinimumWidth(120)
    widget.setEditTriggers(QTableWidget.NoEditTriggers)
    widget.setSelectionBehavior(QTableWidget.SelectRows)
    widget.setSelectionMode(QTableWidget.SingleSelection)
    return widget


def peak_rows(peaks: list[dict]) -> list[list]:
    """The peak table's cells: plain words, with the full sentence and how it was obtained as tooltips."""
    rows = []
    for peak in peaks:
        size = peak.get("size_nm")
        orientation = str(peak.get("orientation") or "")
        caveat = str(peak.get("caveat") or "")
        size_text = "—" if size is None else ("≥ " if peak.get("size_is_lower_bound") else "") + f"{size:.3g}"
        rows.append([
            f"{peak.get('q'):.4g}", f"{peak.get('d_A'):.4g}", f"{peak.get('fwhm'):.3g}",
            (short_orientation(orientation), orientation),
            (size_text, str(peak.get("size_note") or "")),
            (trust(caveat), caveat) if caveat or "at_edge" not in (peak.get("flags") or ()) else (
                "at the end of the data: check",
                "The peak lies within 1.5 widths of the end of the measured q range: its shape and position "
                "may be cut off. Look at I(q) before using it.",
            ),
        ])
    return rows


__all__ = [
    "GISAXS_NOTE", "GOOD", "NOTE", "OPTION_FIELDS", "PEAK_COLUMNS", "RUN_ANYWAY_TEXT", "RUN_TEXT", "STEPS", "WARN", "badge", "clear_layout", "coverage_notes", "label",
    "peak_rows", "ring_summary", "short_orientation", "table", "trust",
]
