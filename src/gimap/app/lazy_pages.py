"""Workspaces that are built when they are first needed: Compare and Sputter.

Building every page before the window appeared cost seconds of a start-up that showed nothing; Compare and
Sputter are not needed before the user goes there (or Analyze sends a series to Compare, or a project with
their part is opened). Until then an empty placeholder holds their place in the page stack, so the stack
keeps its order and size. ``compare_page`` / ``sputter_page`` build the page on first use; ``built_page``
answers without building (what the quit question, a language switch, a project save and the change watch
use: a page that was never built has nothing to say, stop or save).
"""

from __future__ import annotations

from PyQt5.QtWidgets import QWidget

from . import startup_progress

LAZY_PAGES = ("compare", "sputter")
BUILDING = {"compare": "Opening Compare…", "sputter": "Opening Sputter…"}
"""What the splash says while a page is built during the start-up (a project that has its part)."""


class LazyPagesMixin:
    """Needs ``ui`` (with ``mainWindowWidget`` and ``comparePageHost``), ``pages``, ``task_runner``,
    ``preferences`` and ``analyze_page``."""

    def _prepare_lazy_pages(self) -> None:
        self._built_pages: dict[str, QWidget] = {}
        self._page_built_callbacks: list = []
        self._lazy_placeholders = {"compare": self.ui.comparePageHost}
        sputter = QWidget()
        sputter.setObjectName("sputterPageHost")
        self.ui.mainWindowWidget.addWidget(sputter)
        self._lazy_placeholders["sputter"] = sputter
        # Analyze ▸ Send to Compare builds Compare when it is used.
        self.analyze_page.set_compare_target(self._send_series_to_compare)

    def built_page(self, key: str):
        """The page of ``key`` when it is built (every other page is), else ``None``; never builds it."""
        if key in LAZY_PAGES:
            return getattr(self, "_built_pages", {}).get(key)
        page = getattr(self, "pages", {}).get(key)
        return page

    def page_widget(self, key: str) -> QWidget:
        """The page of ``key``, built now if it was not."""
        if key in LAZY_PAGES and key not in self._built_pages:
            self._build_lazy_page(key)
        return self.pages[key]

    def on_page_built(self, callback) -> None:
        """``callback(key, page)`` after a lazy page was built (the splitter memory restores its layout)."""
        self._page_built_callbacks.append(callback)

    @property
    def compare_page(self):
        return self.page_widget("compare")

    @property
    def sputter_page(self):
        return self.page_widget("sputter")

    def _build_lazy_page(self, key: str) -> None:
        startup_progress.report(BUILDING[key])
        placeholder = self._lazy_placeholders.pop(key)
        page = self._create_compare_page() if key == "compare" else self._create_sputter_page()
        stack = self.ui.mainWindowWidget
        index = stack.indexOf(placeholder)
        stack.insertWidget(index if index >= 0 else stack.count(), page)
        if self.ui.mainWindowWidget.currentWidget() is placeholder:
            stack.setCurrentWidget(page)
        stack.removeWidget(placeholder)
        placeholder.setParent(None)
        placeholder.deleteLater()
        if getattr(self.ui, "comparePageHost", None) is placeholder:
            delattr(self.ui, "comparePageHost")
        self._built_pages[key] = page
        self.pages[key] = page
        for callback in list(self._page_built_callbacks):
            callback(key, page)

    def _create_compare_page(self):
        """Compare: Analyze's Series maps (Send to Compare) and curve files side by side."""
        from src.gimap.features.compare.bootstrap import create_compare_service
        from src.gimap.features.compare.presentation.page import ComparePage

        page = ComparePage(create_compare_service(), task_runner=self.task_runner, preferences=self.preferences,
                           analyze_map=self.analyze_page.current_series)
        self.ui.comparePage = page
        return page

    def _create_sputter_page(self):
        """Sputter: the grains of an in-situ sputter series, window by window (the bundled analysis package)."""
        from src.gimap.features.sputter.bootstrap import create_sputter_service
        from src.gimap.features.sputter.presentation.page import SputterPage

        page = SputterPage(create_sputter_service(), task_runner=self.task_runner, preferences=self.preferences)
        self.ui.sputterPage = page
        return page


__all__ = ["LAZY_PAGES", "LazyPagesMixin"]
