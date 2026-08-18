"""Workflow behavior for Format Converter."""

from __future__ import annotations


from PyQt5.QtWidgets import (
    QMessageBox,
)


from ..display_formatting import _human_bytes


class WorkflowMixin:
    """Own workflow presentation behavior."""

    def _back(self) -> None:
        self.stack.setCurrentIndex(max(0, self.stack.currentIndex() - 1))
        self._update_step_header()

    def _next(self) -> None:
        index = self.stack.currentIndex()
        if index == 0:
            if not self.sources:
                QMessageBox.information(self, "Format Converter", "Add at least one input file.")
                return
            self._refresh_selection_table()
            self.stack.setCurrentIndex(1)
        elif index == 1:
            if not any(source.included and source.selected_frames for source in self.sources):
                QMessageBox.information(
                    self, "Format Converter", "Select at least one image or frame to convert."
                )
                return
            self._configure_output_formats()
            self.stack.setCurrentIndex(2)
        else:
            self._review_and_convert()
            return
        self._update_step_header()

    def _update_step_header(self) -> None:
        current = self.stack.currentIndex()
        for index, label in enumerate(self.step_labels):
            if index == current:
                label.setStyleSheet(
                    "background: #2563eb; color: white; border-radius: 5px; font-weight: 600;"
                )
            elif index < current:
                label.setStyleSheet("background: #dbeafe; color: #1d4ed8; border-radius: 5px;")
            else:
                label.setStyleSheet("background: #f1f5f9; color: #475569; border-radius: 5px;")
        self.back_button.setEnabled(current > 0)
        self.next_button.setText("Review & Convert" if current == 2 else "Next")

    def _review_and_convert(self) -> None:
        options = self._options()
        if not options.destination:
            QMessageBox.warning(self, "Output settings", "Choose an output destination.")
            return
        try:
            review = self.view_model.conversion_review(options)
        except Exception as exc:
            QMessageBox.warning(self, "Output settings", f"Invalid output settings:\n{exc}")
            return
        text = (
            f"Input:\n{review.input_summary}\n"
            f"{review.image_count:,} selected image(s) / frame(s)\n\n"
            f"Output:\n{options.output_format}\n"
            f"Destination: {review.destination}\nNaming: {review.naming}\n\n"
            f"Estimated output:\n{review.output_files:,} file(s), approximately "
            f"{_human_bytes(review.estimated_bytes)}"
        )
        if review.is_large_output:
            text += "\n\n⚠ Large output: conversion can take considerable time and disk space."
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Confirm conversion")
        dialog.setIcon(QMessageBox.Information)
        dialog.setText(text)
        back = dialog.addButton("Back", QMessageBox.RejectRole)
        convert = dialog.addButton("Convert", QMessageBox.AcceptRole)
        dialog.addButton("Cancel", QMessageBox.DestructiveRole)
        dialog.setDefaultButton(convert)
        dialog.exec_()
        if dialog.clickedButton() == convert:
            self._start_conversion(options)
        elif dialog.clickedButton() == back:
            return
