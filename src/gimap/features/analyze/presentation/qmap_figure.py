"""Collect the displayed q-map layers and the user's publication dimensions."""

from PyQt5.QtWidgets import (
    QDialog, QDialogButtonBox, QDoubleSpinBox, QFontComboBox, QFormLayout, QSpinBox,
)
from PyQt5.QtGui import QFont

from src.gimap.app.presentation.i18n import tr
from src.gimap.shared.figures import COLUMN_WIDTH_CM, RASTER_DPI


def figure_options(parent, previous):
    dialog = QDialog(parent)
    dialog.setWindowTitle(tr("q-map figure"))
    form = QFormLayout(dialog)
    fields = {}
    for key, label, default, minimum, maximum in (
        ("width_cm", "Width (cm)", COLUMN_WIDTH_CM, 3, 40),
        ("height_cm", "Height (cm)", COLUMN_WIDTH_CM * 0.85, 3, 40),
        ("font_pt", "Font size (pt)", 8, 5, 30),
        ("dpi", "Image resolution (dpi)", RASTER_DPI, 72, 1200),
    ):
        field = QSpinBox(dialog) if key == "dpi" else QDoubleSpinBox(dialog)
        field.setRange(minimum, maximum)
        field.setValue(previous.get(key, default))
        fields[key] = field
        form.addRow(tr(label), field)
    font = QFontComboBox(dialog)
    font.setCurrentFont(QFont(previous.get("font_family", "Arial")))
    form.addRow(tr("Font family"), font)
    buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel, parent=dialog)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    form.addRow(buttons)
    try:
        if dialog.exec_() == QDialog.Accepted:
            return {**{key: field.value() for key, field in fields.items()}, "font_family": font.currentFont().family()}
    finally:
        dialog.deleteLater()
    return None


def displayed_layers(page):
    view = page.detector_view
    layer = page._image_reflections
    lines, legend = [], []
    if view.marks.is_visible("shapes"):
        outlines, rectangles = page.region_shapes(1)
        lines.extend(outlines)
        for _key, x0, y0, x1, y1, color, _editable in rectangles:
            lines.append(([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color))
        colors = {color for _x, _y, color in lines}
        rows = page._visible_cut_rows() if page.view_model.state.analysis.kind == "gisaxs" else page._visible_rows()
        legend = [(tr(row.name), row.color) for row in rows if row.color in colors]
    if view.box_roi.isVisible():
        x, y = view.box_roi.pos()
        w, h = view.box_roi.size()
        lines.append(([x, x + w, x + w, x, x], [y, y, y + h, y + h, y], "#facc15"))
        legend.append((tr("q box"), "#facc15"))
    return {
        "region_lines": lines, "regions": legend, "arcs": layer.arcs,
        "spots": [(s.color, s.x, s.y, s.sizes, s.hollow) for s in layer.spots if s.x],
        "labels": layer.labels, "phases": layer.legend,
        "beam": (view.beam_target.pos().x(), view.beam_target.pos().y()) if view.beam_target.isVisible() else None,
        "horizon": view.horizon_line.pos().y() if view.horizon_line.isVisible() else None,
        "limits": view.view_box.viewRange(), "colorbar": view.marks.is_visible("colorbar"),
    }
