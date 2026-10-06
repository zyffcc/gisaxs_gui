"""Combo boxes whose items are configuration values (a detector preset, a particle shape, an activation).

The interface language may translate what an item shows ("Custom" → "自定义"); the value written to the
configuration is the item's data, which never changes. A selection is followed with ``currentIndexChanged``
(a translated item text also emits ``currentTextChanged``, which is not an edit).
"""

from __future__ import annotations

from typing import Iterable, Optional

from PyQt5.QtWidgets import QComboBox


def fill_values(combo: QComboBox, values: Iterable[str], current: Optional[str] = None) -> QComboBox:
    """Add ``values`` as items whose data is the value itself; select ``current`` when it is one of them."""
    for value in values:
        combo.addItem(str(value), str(value))
    if current is not None:
        index = combo.findData(str(current))
        if index >= 0:
            combo.setCurrentIndex(index)
    return combo


def combo_value(combo: QComboBox) -> str:
    """The configuration value of the selected item (its data; the text for an item without data)."""
    data = combo.currentData()
    return str(data) if data is not None else combo.currentText()


def set_combo_value(combo: QComboBox, value) -> None:
    """Select the item holding ``value`` (added when the list does not have it, as a loaded project may)."""
    text = str(value)
    index = combo.findData(text)
    if index < 0:
        index = combo.findText(text)
    if index < 0:
        combo.addItem(text, text)
        index = combo.count() - 1
    combo.setCurrentIndex(index)


__all__ = ["combo_value", "fill_values", "set_combo_value"]
