"""Workflow behavior for Format Converter."""

from __future__ import annotations


from PyQt5.QtWidgets import (
    QMessageBox,
)


from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.app.presentation.theme import set_role
from ..display_formatting import _human_bytes


def review_text(review, output_format: str) -> str:
    """What "Review && Convert" asks to confirm, in the interface language.

    Numbers, names, the format and the folder stay as they are; every sentence is a template of
    the table.
    """
    inputs = ", ".join(
        trf("{count} {kind} file(s)", count=count, kind=kind) for kind, count in review.input_counts
    )
    lines = [
        tr("Input:"),
        inputs or review.input_summary,
        trf("{images} selected image(s) / frame(s)", images=f"{review.image_count:,}"),
        "",
        tr("Output:"),
        output_format,
        trf("Destination: {destination}", destination=review.destination),
        trf("Naming: {naming}", naming=review.naming),
        "",
        tr("Estimated output:"),
        trf(
            "{files} file(s), approximately {size}",
            files=f"{review.output_files:,}",
            size=_human_bytes(review.estimated_bytes),
        ),
    ]
    if review.is_large_output:
        lines += ["", tr("⚠ Large output: conversion can take considerable time and disk space.")]
    return "\n".join(lines)


class WorkflowMixin:
    """Own workflow presentation behavior."""

    def _back(self) -> None:
        self.stack.setCurrentIndex(max(0, self.stack.currentIndex() - 1))
        self._update_step_header()

    def _next(self) -> None:
        index = self.stack.currentIndex()
        if index == 0:
            if not self.sources:
                QMessageBox.information(
                    self, tr("Format Converter"), tr("Add at least one input file.")
                )
                return
            self._refresh_selection_table()
            self.stack.setCurrentIndex(1)
        elif index == 1:
            if not any(source.included and source.selected_frames for source in self.sources):
                QMessageBox.information(
                    self,
                    tr("Format Converter"),
                    tr("Select at least one image or frame to convert."),
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
                set_role(label, "step-current")
            elif index < current:
                set_role(label, "step-done")
            else:
                set_role(label, "step")
        self.back_button.setEnabled(current > 0)
        # "&&" is a literal ampersand; a single "&" would become an Alt mnemonic ("Review _Convert").
        self.next_button.setText(tr("Review && Convert") if current == 2 else tr("Next"))
        # The one primary action of every step. No setDefault(): Enter in the filter or the
        # custom-frames field must not jump to the next step.
        set_role(self.next_button, "primary")

    def _review_and_convert(self) -> None:
        options = self._options()
        if not options.destination:
            QMessageBox.warning(self, tr("Output settings"), tr("Choose an output destination."))
            return
        try:
            review = self.view_model.conversion_review(options)
        except Exception as exc:
            QMessageBox.warning(
                self, tr("Output settings"), trf("Invalid output settings:\n{error}", error=tr(str(exc)))
            )
            return
        dialog = QMessageBox(self)
        dialog.setWindowTitle(tr("Confirm conversion"))
        dialog.setIcon(QMessageBox.Information)
        dialog.setText(review_text(review, options.output_format))
        back = dialog.addButton(tr("Back"), QMessageBox.RejectRole)
        convert = dialog.addButton(tr("Convert"), QMessageBox.AcceptRole)
        dialog.addButton(tr("Cancel"), QMessageBox.DestructiveRole)
        dialog.setDefaultButton(convert)
        dialog.exec_()
        if dialog.clickedButton() == convert:
            self._start_conversion(options)
        elif dialog.clickedButton() == back:
            return
