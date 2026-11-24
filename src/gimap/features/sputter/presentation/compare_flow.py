"""Compare… (the input bar, beside Export…): opens Compare Scans (``compare_dialog.py``) — a mixin of ``SputterPage``.

The dialog lists the result shown on the main screen first and the results analysed under the results root (the one
named in More options and the default one), newest first. The result shown and the newest other one start ticked, so
that after analysing two scans one click puts them side by side. One dialog at a time: Compare… again brings it to the
front and ticks the result now shown.

Needs ``compare_button``, ``service`` (its ``compare``), ``results_picker`` and ``client_result``.
"""

from __future__ import annotations

from typing import Optional

from .compare_dialog import CompareDialog


class CompareMixin:
    _compare_dialog: Optional[CompareDialog] = None

    def _connect_compare(self) -> None:
        self.compare_button.clicked.connect(lambda: self.open_compare())
        self.compare_button.setEnabled(getattr(self.service, "compare", None) is not None)

    def compare_roots(self) -> list:
        chosen = self.results_picker.path()
        return [path for path in (chosen or None, self.service.results_root()) if path]

    def open_compare(self) -> Optional[CompareDialog]:
        compare = getattr(self.service, "compare", None)
        if compare is None:
            return None
        shown = self.client_result.folder if self.client_result is not None else None
        dialog = self._compare_dialog
        if dialog is not None and dialog.isVisible():
            if shown is not None:
                dialog.add_folder(shown)
            dialog.raise_()
            dialog.activateWindow()
            return dialog
        folders = compare.analysed(self.compare_roots())
        others = [folder for folder in folders if shown is None or folder.resolve() != shown.resolve()]
        ticked = others[:1] if shown is not None else others[:2]
        dialog = CompareDialog(compare, self, folders=folders, shown=shown, ticked=ticked)
        self._compare_dialog = dialog
        dialog.refresh()
        dialog.show()
        return dialog

    def close_compare(self) -> None:
        if self._compare_dialog is not None:
            self._compare_dialog.close()
            self._compare_dialog = None


__all__ = ["CompareMixin"]
