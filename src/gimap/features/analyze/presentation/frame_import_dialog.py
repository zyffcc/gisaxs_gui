"""Choose source-scoped frame references before reading detector images."""
from PyQt5.QtCore import QSignalBlocker
from PyQt5.QtWidgets import QDialog, QDialogButtonBox
from src.gimap.app.presentation.i18n import language_changed, trf
from .frame_selection import FrameSelection
from .views.frame_import_dialog_view import FrameImportDialogView


class FrameImportDialog(QDialog, FrameImportDialogView):
    def __init__(self, sources, *, selections=None, parent=None):
        super().__init__(parent)
        self.sources = list(sources)
        self._choices = {key: FrameSelection.from_record(value) if isinstance(value, dict) else value
                         for key, value in dict(selections or {}).items()}
        self._current = -1
        self.setup_import_view()
        self._source_names()
        self.source_combo.currentIndexChanged.connect(self._source_changed)
        self.import_mode_combo.currentIndexChanged.connect(lambda _index: self._edited())
        for spin in (self.import_first_spin, self.import_last_spin, self.import_step_spin):
            spin.valueChanged.connect(lambda _value: self._edited())
        self.import_numbers_edit.textChanged.connect(lambda _text: self._edited())
        self.import_buttons.accepted.connect(self.accept)
        self.import_buttons.rejected.connect(self.reject)
        language_changed().connect(self.refresh_language)
        self._source_changed(0)

    def _source_names(self):
        with QSignalBlocker(self.source_combo):
            selected = max(0, self.source_combo.currentIndex())
            self.source_combo.clear()
            for source in self.sources:
                self.source_combo.addItem(trf("{name} · {n} frames", name=source.path.name, n=source.frames))
            if self.sources:
                self.source_combo.setCurrentIndex(min(selected, len(self.sources) - 1))

    def _source_changed(self, index):
        if not 0 <= index < len(self.sources):
            return
        self._current = index
        source = self.sources[index]
        choice = self._choices.get(str(source.path))
        if isinstance(choice, dict):
            choice = FrameSelection.from_record(choice)
        indices = choice.indices(source.frames) if choice is not None else range(source.frames)
        widgets = (self.import_mode_combo, self.import_first_spin, self.import_last_spin,
                   self.import_step_spin, self.import_numbers_edit)
        blockers = [QSignalBlocker(widget) for widget in widgets]
        for spin in widgets[1:4]:
            spin.setRange(1, max(1, source.frames))
        if choice is None:
            mode, first, last, step, numbers = "all", 1, source.frames, 1, ""
        elif isinstance(indices, range) and len(indices):
            mode, first, last, step, numbers = "range", indices.start + 1, indices[-1] + 1, indices.step, ""
        elif len(indices) > 1000:
            mode, first, last, step, numbers = "current", 1, source.frames, 1, ""
        else:
            mode, first, last, step, numbers = "specific", 1, source.frames, 1, ", ".join(str(i + 1) for i in indices)
        self.import_mode_combo.setCurrentIndex(self.import_mode_combo.findData(mode))
        self.import_first_spin.setValue(first)
        self.import_last_spin.setValue(last)
        self.import_step_spin.setValue(step)
        self.import_numbers_edit.setText(numbers)
        del blockers
        self._edited()

    def _edited(self):
        if not 0 <= self._current < len(self.sources):
            return
        source = self.sources[self._current]
        mode = self.import_mode_combo.currentData()
        for spin in (self.import_first_spin, self.import_last_spin, self.import_step_spin):
            spin.setEnabled(mode == "range")
        self.import_numbers_edit.setEnabled(mode == "specific")
        error = ""
        try:
            if mode == "all":
                choice = None
            elif mode == "current":
                choice = self._choices.get(str(source.path))
            elif mode == "range":
                first, last = self.import_first_spin.value(), self.import_last_spin.value()
                if first > last:
                    raise ValueError("The first frame must not be after the last frame.")
                choice = FrameSelection.range(first - 1, last - 1, self.import_step_spin.value())
            else:
                tokens = self.import_numbers_edit.text().replace("，", ",").split(",")
                if not all(token.strip().isdigit() for token in tokens):
                    raise ValueError("Enter frame numbers separated by commas, for example 1, 5, 20.")
                indices = [int(token.strip()) - 1 for token in tokens]
                if any(i < 0 or i >= source.frames for i in indices):
                    raise ValueError("Every frame number must be within this source.")
                choice = FrameSelection.explicit(indices)
            self._choices[str(source.path)] = choice
        except ValueError as exc:
            error = str(exc)
        self.import_error_label.setText(trf(error) if error else "")
        self.import_buttons.button(QDialogButtonBox.Ok).setEnabled(not error)
        if not error:
            total = sum(len(self._choices[str(item.path)].indices(item.frames))
                        if self._choices.get(str(item.path)) is not None else item.frames for item in self.sources)
            self.import_summary_label.setText(trf("{n} frames will be imported from {sources} sources.",
                                                 n=total, sources=len(self.sources)))

    def selections(self):
        return dict(self._choices)

    def refresh_language(self):
        self._source_names()
        self._edited()
