"""The marks over Analyze's images and plots: each kind shown or hidden, the ones you made removable.

Every image (detector, q map, cake, the Series map) and every curve plot has a **Marks** menu
(``components/marks.py``). The choices are remembered between sessions (settings
``analyze.hidden_marks``, per view). On the image, the marks you made can be removed from the same
menu — the drawn masks, the cut regions, the q box — and Undo (Ctrl+Z) brings them back. Pressing
**Sources** or **Show Masked Pixels** shows the pixel overlay again even if it was hidden.
"""

from __future__ import annotations

HIDDEN_KEY = "hidden_marks"
SECTION = "analyze"


class MarksMixin:
    """Needs the views (``detector_view``, the curve plots, ``series_map_view``), ``view_model`` and the
    mask / region / box controls."""

    def _mark_views(self) -> dict:
        return {
            "image": self.detector_view, "top": self.top_plot, "bottom": self.bottom_plot,
            "series": self.series_map_view, "series_profile": self.series_profile_plot,
            "series_trace": self.series_trace_plot,
        }

    def _connect_marks(self) -> None:
        stored = self._stored_hidden_marks()
        for name, view in self._mark_views().items():
            view.marks.set_hidden(stored.get(name, ()))
            view.marks.changed.connect(lambda _key, _visible: self._remember_marks())
        self.series_map_view.marks.set_title("bands", "Frame and q bands")
        marks = self.detector_view.marks
        state = self.view_model.state
        marks.add_removal(
            "Remove the Drawn Masks", self._clear_masks,
            lambda: bool(state.corrections.mask_shapes or state.corrections.mask_path),
            "The masks drawn or loaded are removed from every curve; Undo (Ctrl+Z) brings them back",
        )
        marks.add_removal(
            "Remove All Cut Regions", self._remove_all_regions, lambda: bool(state.giwaxs.regions),
            "Your cut regions and their curves; Undo (Ctrl+Z) brings them back",
        )
        marks.add_removal(
            "Remove the q Box", lambda: self.box_check.setChecked(False), lambda: state.giwaxs.box is not None,
            "The box and its two curves; Undo (Ctrl+Z) brings it back",
        )
        for button in (self.sources_button, self.show_mask_button):
            button.toggled.connect(lambda on: on and self.detector_view.marks.set_visible("pixels", True, notify=True))

    def _remove_all_regions(self) -> None:
        self.view_model.set_regions(())
        self._hidden_regions.clear()
        self.run_analysis()

    def _stored_hidden_marks(self) -> dict:
        settings = self.view_model.settings
        if settings is None:
            return {}
        try:
            stored = settings.get(SECTION, HIDDEN_KEY, {}) or {}
        except Exception:  # noqa: BLE001 - a preference only
            return {}
        return {str(name): [str(key) for key in keys] for name, keys in stored.items() if isinstance(keys, list)}

    def _remember_marks(self) -> None:
        settings = self.view_model.settings
        if settings is None:
            return
        hidden = {name: view.marks.hidden() for name, view in self._mark_views().items() if view.marks.hidden()}
        try:
            settings.set(SECTION, HIDDEN_KEY, hidden)
            settings.save()
        except Exception:  # noqa: BLE001 - a preference only
            pass


__all__ = ["MarksMixin"]
