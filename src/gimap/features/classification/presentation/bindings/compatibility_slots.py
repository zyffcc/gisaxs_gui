"""Compatibility Slots coordination for Classification."""

from __future__ import annotations


class CompatibilitySlotsMixin:
    """Own compatibility slots presentation behavior."""

    def _on_import_clicked(self):
        self._start_import()

    def _on_clf_start_clicked(self):
        self._start_training()

    def _on_clf_save_clicked(self):
        self._save_active_model()

    def _on_clf_load_clicked(self):
        self._load_model()

    def _on_import_classify_clicked(self):
        self._predict_new_data_menu()
