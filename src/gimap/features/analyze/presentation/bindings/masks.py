"""Detector drawing, mask rule previews and ordered edits in Analyze."""

from dataclasses import replace
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QSignalBlocker

from src.gimap.app.presentation.components import MASK_PURPOSE
from src.gimap.app.presentation.i18n import tr, trf

from ...application import MaskRule, MaskShape
from ..texts import mask_text, message_text
from ..views.mask_tools_view import SHAPES
from .display import VIEW_DETECTOR
from .mask_files import MaskFilesMixin


HINTS = {
    "rectangle": "Click two opposite corners on the image (Esc cancels).",
    "polygon": "Click the corners on the image; double-click or Enter closes the polygon, Esc cancels.",
    "ellipse": "Click opposite corners of the box enclosing the ellipse.",
    "circle": "Click the centre and then a point on the rim.",
    "annulus": "Click the centre, then the inner rim, then the outer rim.",
    "brush": "Drag across the pixels to paint. Release to preview.",
    "eraser": "Drag across the pixels to restore. Release to preview.",
}


class MaskToolsMixin(MaskFilesMixin):
    def _connect_masks(self) -> None:
        self._mask_candidate = None
        self._mask_count = None
        self.mirror_fill_check.setChecked(self.view_model.state.corrections.mirror_fill)
        self.mirror_fill_check.toggled.connect(self._mirror_fill_changed)
        self.mask_tool.currentIndexChanged.connect(self._mask_tool_changed)
        self.mask_draw.toggled.connect(self._draw_toggled)
        self.shape_layer.shapeDrawn.connect(self._shape_drawn)
        self.shape_layer.drawingChanged.connect(self._drawing_changed)
        self.mask_preview.clicked.connect(self._preview_rule)
        self.mask_apply.clicked.connect(self._apply_mask_edit)
        self.mask_cancel.clicked.connect(self._cancel_mask_preview)
        self.view_combo.activated.connect(self._mask_inputs_changed)
        for _name, spin, _kinds in self.mask_fields:
            spin.valueChanged.connect(self._mask_inputs_changed)
        self.mask_relative.toggled.connect(self._mask_inputs_changed)
        self.mask_mode.currentIndexChanged.connect(self._mask_inputs_changed)
        self.mask_remove_button.clicked.connect(self._remove_mask)
        self.mask_clear_button.clicked.connect(self._clear_masks)
        self.mask_save_button.clicked.connect(lambda: self.save_masks())
        self.mask_load_button.clicked.connect(lambda: self.load_masks())
        self._mask_tool_changed()
        self._refresh_mask_list()

    def _mirror_fill_changed(self, enabled: bool) -> None:
        self.view_model.set_mirror_fill(enabled)
        self._options_changed()

    def _mask_inputs_changed(self) -> None:
        if self.shape_layer.purpose == MASK_PURPOSE:
            self.shape_layer.cancel_drawing()
        self._cancel_mask_preview()

    def _mask_tool_changed(self) -> None:
        self._mask_inputs_changed()
        kind = self.mask_tool.currentData()
        self.mask_draw.setVisible(kind in HINTS)
        self.mask_draw.setToolTip(tr(HINTS.get(kind, "")))
        self.mask_mode.setVisible(kind in SHAPES and kind != "eraser")
        self.mask_relative.setVisible(kind in ("above", "below"))
        self.mask_preview.setVisible(kind not in HINTS)
        self.mask_parameters.setVisible(any(kind in kinds for _, _, kinds in self.mask_fields))
        for _name, spin, kinds in self.mask_fields:
            spin.setVisible(kind in kinds)
            self.mask_parameters.layout().labelForField(spin).setVisible(kind in kinds)

    def _draw_toggled(self, on: bool) -> None:
        if not on:
            if self.shape_layer.purpose == MASK_PURPOSE:
                self.shape_layer.cancel_drawing()
            return
        if self.view_model.state.analysis is None:
            self._drawing_changed("")
            self._status(tr("Open a frame first."), "warning")
            return
        self._cancel_mask_preview()
        if self.view_combo.currentIndex() != VIEW_DETECTOR:
            self.set_view(VIEW_DETECTOR)
        kind = self.mask_tool.currentData()
        self.shape_layer.start_drawing(kind, radius=self.mask_radius.value())
        self._status(tr(HINTS[kind]))

    def _drawing_changed(self, kind: str) -> None:
        with QSignalBlocker(self.mask_draw):
            self.mask_draw.setChecked(bool(kind) and self.shape_layer.purpose == MASK_PURPOSE)

    def _shape_drawn(self, kind: str, points: list) -> None:
        try:
            mode = self.mask_mode.currentData()
            shape = MaskShape(kind, tuple(points), mode=mode, radius=self.mask_radius.value())
        except ValueError as exc:
            self._status(message_text(exc), "warning")
            return
        if kind in ("rectangle", "polygon") and mode == "exclude":
            self._append_edit(shape)
        else:
            self._preview_edit(shape)

    def _preview_rule(self) -> None:
        kind = self.mask_tool.currentData()
        try:
            if kind == "q_annulus":
                limits = (self.mask_q_min.value(), self.mask_q_max.value())
                edit = MaskShape(kind, (limits,), mode=self.mask_mode.currentData())
            else:
                value, relative = None, False
                if kind in ("grow", "shrink"):
                    value = self.mask_amount.value()
                elif kind in ("above", "below", "saturated"):
                    value = self.mask_threshold.value()
                    relative = kind != "saturated" and self.mask_relative.isChecked()
                edit = MaskRule(kind, value, relative)
            self._preview_edit(edit)
        except ValueError as exc:
            self._status(message_text(exc), "warning")

    def _preview_edit(self, edit) -> None:
        self._cancel_mask_preview()
        state = self.view_model.state
        analysis = state.analysis
        if analysis is None:
            self._status(tr("Open a frame first."), "warning")
            return
        if self._center_analysis_pending:
            self._status(tr("Wait for the current frame analysis."), "warning")
            return
        if self.view_combo.currentIndex() != VIEW_DETECTOR:
            self.set_view(VIEW_DETECTOR)
        current = state.corrections
        candidate = replace(current, mask_shapes=(*current.mask_shapes, edit))
        self.mask_preview_label.setText(tr("Calculating mask preview…"))
        self.mask_preview_label.show()
        self.mask_cancel.show()

        def ready(mask):
            if state.analysis is not analysis or state.corrections != current:
                self._cancel_mask_preview()
                return
            self._mask_candidate = (analysis, current, edit)
            self._mask_count = int(np.count_nonzero(mask))
            self._show_mask_count()
            self.mask_apply.show()
            self.mask_preview.hide()
            self.step_pages["mask"].ensureWidgetVisible(self.mask_cancel)
            if self.view_combo.currentIndex() == VIEW_DETECTOR:
                self.shape_layer.set_mask_image(mask)

        def failed(error, _details):
            self._cancel_mask_preview()
            self._status(message_text(error), "warning")

        self.tasks.submit("mask_preview", lambda: self.view_model.preview_user_mask(analysis, candidate),
                          on_done=ready, on_error=failed)

    def _show_mask_count(self) -> None:
        if self._mask_count is not None:
            self.mask_preview_label.setText(trf("After this edit: {count} pixels masked", count=self._mask_count))
            self.mask_preview_label.show()

    def _cancel_mask_preview(self) -> None:
        self.tasks.cancel("mask_preview")
        redraw = self._mask_candidate is not None
        self._mask_candidate = self._mask_count = None
        self.mask_apply.hide()
        self.mask_cancel.hide()
        self.mask_preview.setVisible(self.mask_tool.currentData() not in HINTS)
        self.mask_preview_label.hide()
        if redraw:
            self._refresh_shapes()

    def _apply_mask_edit(self) -> None:
        if self._mask_candidate is not None:
            analysis, corrections, edit = self._mask_candidate
            state = self.view_model.state
            self._cancel_mask_preview()
            if analysis is state.analysis and corrections == state.corrections:
                self._append_edit(edit)

    def _append_edit(self, edit) -> None:
        self.view_model.add_mask_shape(edit)
        self._refresh_mask_list()
        self.run_analysis()

    def _refresh_mask_list(self) -> None:
        corrections = self.view_model.state.corrections
        self.mask_list.clear()
        if corrections.mask_path:
            self.mask_list.addItem(trf("Mask file: {name}", name=Path(corrections.mask_path).name))
            self.mask_list.item(0).setToolTip(corrections.mask_path)
        for edit in corrections.mask_shapes:
            self.mask_list.addItem(mask_text(edit))
            details = [mask_text(edit), *edit.sources]
            if edit.kind == "image":
                details.append(edit.path)
            if edit.kind in ("circle", "annulus"):
                x, y = edit.points[0]
                details.insert(1, f"x = {x:g} px, y = {y:g} px")
            self.mask_list.item(self.mask_list.count() - 1).setToolTip("\n".join(details))
        if self.mask_list.count():
            self.mask_list.setCurrentRow(self.mask_list.count() - 1)
        for widget in (self.mask_list, self.mask_remove_button, self.mask_clear_button, self.mask_save_button):
            widget.setVisible(bool(self.mask_list.count()))
        self._show_mask_count()
        self._show_flat_field()

    def _remove_mask(self) -> None:
        corrections = self.view_model.state.corrections
        row = self.mask_list.currentRow()
        if row < 0:
            row = self.mask_list.count() - 1
        if corrections.mask_path and row == 0:
            self.view_model.set_mask_path(None)
        else:
            index = row - int(bool(corrections.mask_path))
            edits = list(corrections.mask_shapes)
            if not 0 <= index < len(edits):
                return
            del edits[index]
            self.view_model.set_mask_shapes(edits)
        self._refresh_mask_list()
        self.run_analysis()

    def _clear_masks(self) -> None:
        self.view_model.set_mask_shapes(())
        self.view_model.set_mask_path(None)
        self._refresh_mask_list()
        self.run_analysis()


__all__ = ["MaskToolsMixin"]
