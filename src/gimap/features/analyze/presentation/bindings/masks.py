"""Behaviour of the preprocessing tools of the Mask step: drawn masks, mask files, mirror filling.

Rectangles and polygons are drawn on the detector image (the view switches to it); each shape is
listed, outlined in red on the image and left out of every curve at once. Masks can be saved as
JSON and loaded back, or a mask image of the same size can be loaded (non-zero = masked).
"""

from __future__ import annotations

import json
from pathlib import Path

from PyQt5.QtCore import QSignalBlocker
from PyQt5.QtWidgets import QFileDialog

from src.gimap.app.presentation.components import MASK_PURPOSE
from src.gimap.app.presentation.i18n import tr, trf

from ...application import POLYGON, RECTANGLE, MaskShape
from ..texts import mask_text, message_text
from .display import VIEW_DETECTOR

MASK_FILTER = "Masks (*.json *.edf *.tif *.tiff *.npy);;GIMaP masks (*.json);;Mask images (*.edf *.tif *.tiff)"


class MaskToolsMixin:
    """Needs the Mask-step widgets, ``shape_layer``, ``view_model``, ``run_analysis`` and ``set_view``."""

    def _connect_masks(self) -> None:
        self.mirror_fill_check.setChecked(self.view_model.state.corrections.mirror_fill)
        self.mirror_fill_check.toggled.connect(self._mirror_fill_changed)
        self.draw_rect_button.toggled.connect(lambda on: self._draw_toggled(RECTANGLE, on))
        self.draw_polygon_button.toggled.connect(lambda on: self._draw_toggled(POLYGON, on))
        self.shape_layer.shapeDrawn.connect(self._shape_drawn)
        self.shape_layer.drawingChanged.connect(self._drawing_changed)
        self.mask_remove_button.clicked.connect(self._remove_mask)
        self.mask_clear_button.clicked.connect(self._clear_masks)
        self.mask_save_button.clicked.connect(lambda: self.save_masks())
        self.mask_load_button.clicked.connect(lambda: self.load_masks())
        self._refresh_mask_list()

    def _mirror_fill_changed(self, enabled: bool) -> None:
        self.view_model.set_mirror_fill(enabled)
        self._options_changed()

    # -- drawing -------------------------------------------------------------------------

    def _draw_toggled(self, kind: str, on: bool) -> None:
        if not on:
            if self.shape_layer.kind == kind:
                self.shape_layer.cancel_drawing()
            return
        if self.view_model.state.analysis is None:
            self._set_draw_buttons("")
            self._status(tr("Open a frame first."), "warning")
            return
        if self.view_combo.currentIndex() != VIEW_DETECTOR:
            self.set_view(VIEW_DETECTOR)
        self.shape_layer.start_drawing(kind)
        hint = (
            "Click two opposite corners on the image (Esc cancels)." if kind == RECTANGLE else
            "Click the corners on the image; double-click or Enter closes the polygon, Esc cancels."
        )
        self._status(tr(hint))

    def _drawing_changed(self, kind: str) -> None:
        self._set_draw_buttons(kind if self.shape_layer.purpose == MASK_PURPOSE else "")

    def _set_draw_buttons(self, kind: str) -> None:
        for button, button_kind in ((self.draw_rect_button, RECTANGLE), (self.draw_polygon_button, POLYGON)):
            with QSignalBlocker(button):
                button.setChecked(kind == button_kind)

    def _shape_drawn(self, kind: str, points: list) -> None:
        try:
            shape = MaskShape(kind, tuple(points))
        except ValueError as exc:
            self._status(message_text(exc), "warning")
            return
        self.view_model.add_mask_shape(shape)
        self._refresh_mask_list()
        self._status(trf("Mask added: {shape}", shape=mask_text(shape)), "ok")
        self.run_analysis()

    # -- the list ------------------------------------------------------------------------

    def _refresh_mask_list(self) -> None:
        corrections = self.view_model.state.corrections
        self.mask_list.clear()
        for shape in corrections.mask_shapes:
            self.mask_list.addItem(mask_text(shape))
        if corrections.mask_path:
            self.mask_list.addItem(trf("Mask file: {name}", name=Path(corrections.mask_path).name))
        empty = not corrections.mask_shapes and not corrections.mask_path
        self.mask_list.setVisible(not empty)
        # Only what can be used: Remove / Clear / Save appear once there is a mask.
        self.mask_remove_button.setVisible(not empty)
        self.mask_clear_button.setVisible(not empty)
        self.mask_save_button.setVisible(bool(corrections.mask_shapes))

    def _remove_mask(self) -> None:
        row = self.mask_list.currentRow()
        corrections = self.view_model.state.corrections
        shapes = list(corrections.mask_shapes)
        if 0 <= row < len(shapes):
            del shapes[row]
            self.view_model.set_mask_shapes(shapes)
        elif corrections.mask_path and row == len(shapes):
            self.view_model.set_mask_path(None)
        elif shapes:
            shapes.pop()
            self.view_model.set_mask_shapes(shapes)
        self._refresh_mask_list()
        self.run_analysis()

    def _clear_masks(self) -> None:
        self.view_model.set_mask_shapes(())
        self.view_model.set_mask_path(None)
        self._refresh_mask_list()
        self.run_analysis()

    # -- files ---------------------------------------------------------------------------

    def save_masks(self, path: str | Path | None = None):
        shapes = self.view_model.state.corrections.mask_shapes
        if not shapes:
            return None
        analysis = self.view_model.state.analysis
        if path is None:
            folder = self.view_model.default_export_dir() or Path(self._last_folder or ".")
            path, _ = QFileDialog.getSaveFileName(self, tr("Save Masks"), str(folder / "masks.json"), "GIMaP masks (*.json)")
            if not path:
                return None
        record = {
            "software": "GIMaP Analyze", "convention": "canonical pixels: x right, y down, pixel (0, 0) from 0 to 1",
            "frame_shape": list(analysis.shape) if analysis is not None else None,
            "masks": [{"kind": shape.kind, "points": [list(point) for point in shape.points]} for shape in shapes],
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        self.notify_written(trf("Saved {name}", name=path.name), path.parent)
        return path

    def load_masks(self, path: str | Path | None = None) -> bool:
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, tr("Load Masks"), self._last_folder, MASK_FILTER)
            if not path:
                return False
        path = Path(path)
        try:
            if path.suffix.lower() == ".json":
                record = json.loads(path.read_text(encoding="utf-8"))
                shapes = [MaskShape(item["kind"], tuple(tuple(point) for point in item["points"])) for item in record["masks"]]
                self.view_model.set_mask_shapes((*self.view_model.state.corrections.mask_shapes, *shapes))
            else:
                self.view_model.set_mask_path(path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self._status(trf("Could not read the masks: {error}", error=exc), "error")
            return False
        self._refresh_mask_list()
        self._status(trf("Masks loaded from {name}", name=path.name), "ok")
        self.run_analysis()
        return True


__all__ = ["MaskToolsMixin"]
