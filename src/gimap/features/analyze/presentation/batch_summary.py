"""What the Batch Export dialog says before anything is written (a part of ``BatchExportDialog``).

* **Where** — the folder the files go to, shortened from the left so that its last folders, the part that
  differs between two data sets, stay in view. A folder that will not take files is said here, with Export
  off, not after the first frame was reduced. A folder remembered from a batch of other data is named as such,
  and *Next to the Data* brings back the one beside the files.
* **What is there already** — an earlier export of the same data, or curves for Fitting that Fitting would
  list together with the new ones: the dialog says so, and Export asks *Replace*, *Keep Both in a New
  Subfolder* or *Cancel* (``_ask_earlier``; nothing is removed or mixed without that answer). Enter answers
  *Replace* only when Replace takes nothing this export does not write again; otherwise *Keep Both*, and the
  question names what Replace would take. Send Series to Fitting writes into a folder of its own,
  ``<name>_fitting``, beside a Batch Export of the same data — not over it.
* **Summary** — how many frames, which curves, which cut and which halves, about how big and how long.
* **Cuts** — where the automatic cut of every frame is taken (the view model's sentence), and the box that has
  it found again in every frame. While the frame on screen is read again (the automatic analysis' sum was
  switched off), no cut is named and Export waits (``frame_shown``): the cut said is the cut every frame gets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from src.gimap.app.presentation import theme
from src.gimap.app.presentation.i18n import tr, trf

from ..application import (
    NO_PERMISSION,
    NO_SUCH_PLACE,
    NOT_A_FOLDER,
    BatchChoices,
    EarlierExport,
    earlier_export,
    folder_name,
    may_be_written,
    new_subfolder,
)
from .batch_texts import halves_text

CANNOT_WRITE = "This folder cannot be written ({reason}): choose another one."
EARLIER_EXPORT = ("This folder already holds an export of these data ({when}, {frames} frames): Export asks whether "
                  "to replace it or to keep both.")
EARLIER_CURVES = ("This folder already holds {n} curves for Fitting: Export asks whether to replace them or to keep "
                  "both.")
CARRIED = "This is the folder of the last batch, which was made of other data."
READING = "the frame on screen is read again without the sum …"
"""The Cuts row while the frame the series is cut like is being read (Export waits for it)."""
REPLACE_TAKES = "Replace removes the {n} files of the earlier export first."
REPLACE_TAKES_OTHERS = ("Replace removes the {n} files of the earlier export first, among them {m} files of kinds this "
                        "export does not write again: {kinds}.")
KIND_WORDS = {
    "tables": "tables of all frames", "per_frame": "curves of single frames", "fit_input": "curves for Fitting",
    "q_map": "q maps", "cake": "cakes", "detector_image": "pictures", "q_map_image": "pictures",
    "frames": "converted frames", "fit": "fit tables", "fit_plot": "fit tables", "fit_curves": "fitted curves",
    "other": "other files",
}
"""How the question names the kinds of file Replace would take for good (in this order)."""
LISTED = ", "
"""Between the kinds named in a row (Chinese: 、)."""
CURVES_SHOWN = 3
"""The summary names at most this many curves (then “+ n more”)."""
Cost = Callable[[BatchChoices, int], tuple[int, int, float]]


def size_text(size: float) -> str:
    """Bytes in the unit that gives two or three digits (``28 MB``, ``1.3 GB``)."""
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if size >= scale:
            value = size / scale
            return f"{value:.1f} {unit}" if value < 10 else f"{value:.0f} {unit}"
    return f"{max(1, int(size))} B"


def duration_text(seconds: float) -> str:
    if seconds < 90:
        return trf("{n} s", n=max(1, int(round(seconds))))
    if seconds < 5400:
        return trf("{n} min", n=int(round(seconds / 60.0)))
    return trf("{n} h", n=f"{seconds / 3600.0:.1f}")


class BatchFolderMixin:
    """Needs the widgets of ``BatchExportView``, ``choices()``, ``chosen_curves()`` and ``frame_count()``."""

    def _setup_folder(self, *, destination: Path, subfolder: bool, beside: Optional[Path], carried: bool,
                      cost: Optional[Cost], cut_text: Optional[Callable[[], str]], pin_cuts: Optional[bool],
                      set_pin_cuts: Optional[Callable[[bool], None]], notes: Sequence[str], sum_size: int,
                      reading: bool = False) -> None:
        self._beside = Path(beside) if beside is not None else None
        self._carried = str(destination) if carried else ""
        self._cost, self._cut_text, self._set_pin_cuts = cost, cut_text, set_pin_cuts
        self._sum_size = max(1, int(sum_size))
        self._earlier_seen: Optional[tuple] = None
        self._earlier_answer = ""
        self._new_target: Optional[Path] = None
        self._reading = bool(reading)
        self._folder_name = folder_name(self._stem, for_fitting=getattr(self, "_for_fitting", False))
        self.notes_label.setText("\n".join(str(note) for note in notes if note))
        self.notes_label.setVisible(bool(self.notes_label.text()))
        shown = cut_text is not None
        for widget in (self.cut_caption, self.cut_label):
            widget.setVisible(shown)
        self.pin_check.setVisible(shown and pin_cuts is not None and set_pin_cuts is not None)
        self.pin_check.setChecked(pin_cuts is False)
        self.pin_check.toggled.connect(self._pin_toggled)
        self._show_cut()
        self.destination_edit.setText(str(destination))
        self.destination_edit.textChanged.connect(self._sync)
        self.subfolder_check.setText(tr("In a subfolder named “{name}”").format(name=self._folder_name))
        self.subfolder_check.setChecked(subfolder)
        self.subfolder_check.toggled.connect(self._sync)
        self.browse_button.clicked.connect(self._browse)
        self.beside_button.setVisible(self._beside is not None)
        self.beside_button.clicked.connect(lambda: self.destination_edit.setText(str(self._beside)))

    # -- the cut of every frame ----------------------------------------------------------

    def _show_cut(self) -> None:
        if self._reading:
            text = tr(READING)
        else:
            text = self._cut_text() if self._cut_text is not None else ""
        self.cut_label.setText(text)
        self.cut_label.setToolTip(text)

    def frame_shown(self, *, reading: bool = False) -> None:
        """The frame on screen changed while the dialog is open: the cut of every frame is said again from it.
        ``reading``: it is still being read again (without the automatic analysis' sum): no cut is named, and
        Export waits for it."""
        self._reading = bool(reading)
        self._show_cut()
        self._sync()

    def reading(self) -> bool:
        return bool(getattr(self, "_reading", False))

    def _pin_toggled(self, again: bool) -> None:
        """*Find the cut again in every frame*: the view model holds the choice (the page gives it back when the
        dialog is cancelled); the sentence beside the box and the summary follow."""
        if self._set_pin_cuts is not None:
            self._set_pin_cuts(not again)
        self._show_cut()
        self._sync()

    def finds_cut_again(self) -> bool:
        return self.pin_check.isChecked()

    # -- the folder ----------------------------------------------------------------------

    def destination(self) -> Path:
        return Path(self.destination_edit.text().strip())

    def subfolder(self) -> bool:
        return self.subfolder_check.isChecked()

    def target(self) -> Path:
        """Where the files go: the destination, or its subfolder named after the data — or, after *Keep Both*,
        the new subfolder beside an earlier export."""
        if self._new_target is not None:
            return self._new_target
        return self.destination() / self._folder_name if self.subfolder() else self.destination()

    def replaces_earlier(self) -> bool:
        """*Replace* was answered: the earlier export in the folder is taken away before this one is written."""
        return self._earlier_answer == "replace"

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, tr("Save the Batch to"), self.destination_edit.text())
        if folder:
            self.destination_edit.setText(folder)

    def _earlier(self, choices: Optional[BatchChoices] = None) -> Optional[EarlierExport]:
        """What the folder holds of an earlier export (looked up again only when the folder changes)."""
        choices = choices or self.choices()
        folder = self.destination() / self._folder_name if self.subfolder() else self.destination()
        key = (str(folder), bool(choices.fit_input))
        if self._earlier_seen is None or self._earlier_seen[0] != key:
            found = earlier_export(folder, self._stem, fit_inputs=choices.fit_input) if self.destination_edit.text().strip() else None
            self._earlier_seen = (key, found)
        return self._earlier_seen[1]

    def _sync_folder(self, choices: BatchChoices, curves: Sequence[str]) -> bool:
        """The target line and what is to be known about the folder; ``True`` when it can be exported to."""
        has_folder = bool(self.destination_edit.text().strip())
        needs_curves = choices.tables or choices.per_frame
        self.beside_button.setEnabled(self._beside is not None and self.destination() != self._beside)
        notes, role, ready = [], "muted", False
        if not has_folder:
            text = tr("Choose a folder.")
        elif not choices.writes_anything:
            text = tr("Tick at least one thing to save.")
        elif needs_curves and not curves:
            text = tr("Tick at least one curve.")
        else:
            target = self.target()
            reason = may_be_written(target)
            if reason is not None:
                reason = tr(reason) if reason in (NOT_A_FOLDER, NO_SUCH_PLACE, NO_PERMISSION) else reason
                text, role = trf(CANNOT_WRITE, reason=reason), "warning"
            else:
                ready = True
                room = max(240, self.width() - 180)  # the last folders stay in view: they are what differs
                text = tr("Files go to {folder}").format(
                    folder=self.target_label.fontMetrics().elidedText(str(target), Qt.ElideLeft, room))
                earlier = self._earlier(choices)
                if earlier is not None and earlier.record is not None:
                    notes.append(trf(EARLIER_EXPORT, when=earlier.when, frames=earlier.frames))
                elif earlier is not None:
                    notes.append(trf(EARLIER_CURVES, n=len(earlier.fit_inputs)))
                if self._carried and self.destination_edit.text().strip() == self._carried:
                    notes.append(tr(CARRIED))
            self.target_label.setToolTip(str(target))
        self.target_label.setText(text)
        theme.set_role(self.target_label, role)
        self.folder_note.setText("\n".join(notes))
        self.folder_note.setVisible(bool(notes))
        return ready

    # -- the summary ---------------------------------------------------------------------

    def _frames_part(self) -> str:
        count, total, every = self.frame_count(), self._frames, max(1, self.every_spin.value())
        if self._sum_size > 1:
            if every > 1:
                return trf("{count} of {total} sums of {size} frames (Every {n})", count=count, total=total,
                           size=self._sum_size, n=every)
            return trf("{count} sums of {size} frames", count=count, size=self._sum_size)
        if every > 1:
            return trf("{count} of {total} frames (Every {n})", count=count, total=total, n=every)
        return tr("1 frame") if count == 1 else trf("{count} frames", count=count)

    def _sync_summary(self, choices: BatchChoices, curves: Sequence[str]) -> None:
        parts = [self._frames_part()]
        if (choices.tables or choices.per_frame) and curves:
            titles = [self._curve_titles.get(key, key) for key in curves]
            more = len(titles) - CURVES_SHOWN
            names = ", ".join(titles[:CURVES_SHOWN]) + (" " + trf("+ {n} more", n=more) if more > 0 else "")
            parts.append(trf("curves: {names}", names=names))
        if self._cut_text is not None and self.cut_label.text():
            parts.append(self.cut_label.text())
        if self._gisaxs and (self._for_fitting or choices.fit_input or choices.fit != "none"):
            parts.append(trf("a fit takes: {halves}", halves=halves_text(choices.fit_halves)))
        lines = [" · ".join(parts)]
        if self._cost is not None:
            per_frame, total, seconds = self._cost(choices, self.frame_count())
            cost = [trf("about {time}", time=duration_text(seconds))]
            if per_frame > 0:
                cost.insert(0, trf("about {size} per frame, {total} in all", size=size_text(per_frame), total=size_text(total)))
            elif total > 0:  # the tables of every frame alone
                cost.insert(0, trf("about {total} in all", total=size_text(total)))
            lines.append(" · ".join(cost) + " · " + tr("README.txt in the folder says what each file is"))
        else:
            lines.append(tr("README.txt in the folder says what each file is"))
        self.summary_label.setText("\n".join(lines))

    # -- Export ----------------------------------------------------------------------------

    def accept(self) -> None:
        """Export. A folder that holds an earlier export first asks what is to become of it; *Cancel* there
        leaves the dialog open."""
        if self._reading:  # (Enter) the cut every frame gets is not known yet
            return
        self._new_target, self._earlier_answer = None, ""
        earlier = self._earlier()
        if earlier is not None:
            answer = self._ask_earlier(earlier)
            if answer not in ("replace", "new"):
                return
            self._earlier_answer = answer
            if answer == "new":
                self._new_target = new_subfolder(self.destination(), self._folder_name)
        super().accept()

    def _ask_earlier(self, earlier: EarlierExport) -> str:
        """``replace``, ``new`` or ``cancel``: what is to become of the earlier export in the folder."""
        box, replace, keep = self._earlier_box(earlier)
        box.exec_()
        clicked = box.clickedButton()
        return "replace" if clicked is replace else "new" if clicked is keep else "cancel"

    def _earlier_box(self, earlier: EarlierExport):
        """The question about an earlier export, with its *Replace* and *Keep Both* buttons (and *Cancel*). Enter
        presses *Replace* only for an earlier export of these data that this one writes again kind for kind (the
        same batch once more); else *Keep Both*, and the question says what Replace would take for good."""
        new = new_subfolder(self.destination(), self._folder_name)
        curves = len(earlier.fit_inputs)
        if earlier.record is not None:
            text = trf("“{folder}” already holds an export of these data, made {when} ({frames} frames).",
                       folder=earlier.folder.name, when=earlier.when, frames=earlier.frames)
        else:
            text = trf("“{folder}” already holds {n} curves for Fitting.", folder=earlier.folder.name, n=curves)
        if curves:
            text += " " + tr("Fitting ▸ In-situ series lists every curve of that folder: the old and the new ones "
                             "would be fitted as one series.")
        choices = self.choices()
        lost = earlier.not_written_again(choices)
        if lost:
            kinds = list(dict.fromkeys(tr(KIND_WORDS[kind]) for kind in earlier.kinds_not_written_again(choices)))
            said = [trf(REPLACE_TAKES_OTHERS, n=len(earlier.to_remove), m=len(lost), kinds=tr(LISTED).join(kinds))]
        else:
            said = [trf(REPLACE_TAKES, n=len(earlier.to_remove))]
        if earlier.record is not None and not earlier.complete:
            said.append(tr("It was written by an earlier version of GIMaP, whose record lists only its tables: the "
                           "files of single frames stay."))
        said.append(trf("Keep Both writes this export into the new subfolder “{name}”.", name=new.name))
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle(self.windowTitle())
        box.setText(text)
        box.setInformativeText(" ".join(said))
        replace = box.addButton(tr("Replace"), QMessageBox.DestructiveRole)
        keep = box.addButton(tr("Keep Both in a New Subfolder"), QMessageBox.AcceptRole)
        box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(replace if earlier.record is not None and not lost else keep)
        return box, replace, keep


__all__ = [
    "BatchFolderMixin", "CANNOT_WRITE", "CARRIED", "EARLIER_CURVES", "EARLIER_EXPORT", "KIND_WORDS", "LISTED", "READING",
    "REPLACE_TAKES", "REPLACE_TAKES_OTHERS", "duration_text", "size_text",
]
