"""Save Geometry as Preset…: a name and the dates of the beamtime the geometry belongs to.

A preset is found again for a scan with the same detector serial number and flat-field file whose date lies
in the range (``application.scan.matching_preset``); the range starts a week either side of the scan.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import QDate
from PyQt5.QtWidgets import QDateEdit, QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QVBoxLayout

from src.gimap.app.presentation.i18n import apply_to, current_language, tr

DAYS_EITHER_SIDE = 7


def ask_preset(parent, *, serial: str, flat_field: str, date: QDate) -> Optional[tuple[str, str, str]]:
    """(name, first day, last day) as ``YYYY-MM-DD``, or None when cancelled."""
    dialog = QDialog(parent)
    dialog.setWindowTitle("Save Geometry as Preset")
    layout = QVBoxLayout(dialog)
    note = QLabel("The geometry on the page (X-rays, distance, incidence, beam centre, sample roll) is used for every "
                  "scan of this detector and flat field taken in these days.", dialog)
    note.setWordWrap(True)
    layout.addWidget(note)
    form = QFormLayout()
    name = QLineEdit(dialog)
    name.setPlaceholderText(tr("for example the beamtime"))
    first, last = QDateEdit(dialog), QDateEdit(dialog)
    for edit, day in ((first, date.addDays(-DAYS_EITHER_SIDE)), (last, date.addDays(DAYS_EITHER_SIDE))):
        edit.setCalendarPopup(True)
        edit.setDisplayFormat("yyyy-MM-dd")
        edit.setDate(day)
    form.addRow("Name", name)
    form.addRow("Detector serial number", QLabel(serial, dialog))
    form.addRow("Flat field", QLabel(flat_field or "–", dialog))
    form.addRow("First day", first)
    form.addRow("Last day", last)
    layout.addLayout(form)
    buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel, dialog)
    save = buttons.button(QDialogButtonBox.Save)
    save.setEnabled(False)  # a preset is found again by its name: Save waits for one
    name.textChanged.connect(lambda words: save.setEnabled(bool(words.strip())))
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    apply_to(dialog, current_language())
    if dialog.exec_() != QDialog.Accepted or not name.text().strip():
        return None
    low, high = sorted((first.date(), last.date()))
    return name.text().strip(), low.toString("yyyy-MM-dd"), high.toString("yyyy-MM-dd")


__all__ = ["ask_preset"]
