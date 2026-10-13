"""Texts the Trainset page composes at run time, shown in the interface language and made again after a switch.

``set(widget, template, **values)`` shows ``trf(template, **values)`` (or ``tr(template)`` without values)
and remembers the English template with its values, so ``refresh()`` composes it in the new language
(the widget walker only knows exact table keys, not texts with numbers or names in them). A text that
is not in the table (an error message, a file name) is shown as it is, in either language.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from src.gimap.app.presentation.i18n import tr, trf


def compose(template: str, values: Dict[str, Any]) -> str:
    return trf(template, **values) if values else tr(template)


class ComposedTexts:
    def __init__(self) -> None:
        self._texts: Dict[Tuple[int, str], Tuple[Any, str, str, Dict[str, Any]]] = {}

    def set(self, widget, template: str, *, setter: str = "setText", **values) -> str:
        """Show ``template`` (filled with ``values``) on ``widget`` through ``setter``; returns the shown text."""
        text = compose(str(template), values)
        self._texts[(id(widget), setter)] = (widget, setter, str(template), dict(values))
        getattr(widget, setter)(text)
        return text

    def template(self, widget, setter: str = "setText") -> str:
        """The English template last shown on ``widget`` ("" when none)."""
        entry = self._texts.get((id(widget), setter))
        return entry[2] if entry is not None else ""

    def refresh(self) -> None:
        """Compose every remembered text again in the current language (widgets already gone are dropped)."""
        for key, (widget, setter, template, values) in list(self._texts.items()):
            try:
                getattr(widget, setter)(compose(template, values))
            except RuntimeError:  # the C++ widget is gone (a closed dialog)
                self._texts.pop(key, None)


__all__ = ["ComposedTexts", "compose"]
