"""Adaptive behavior for the application-owned workspace stack."""

from __future__ import annotations

from PyQt5.QtWidgets import QSizePolicy, QStackedWidget


def configure_adaptive_stack(stack: QStackedWidget) -> None:
    """Preserve the legacy per-page scroll-area sizing behavior."""

    stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    stack.currentChanged.connect(lambda index: _on_page_changed(stack, index))
    print("StackedWidget configured for adaptive mode (each page has its own ScrollArea)")
    print("ℹ️  Each page has its own ScrollArea; no need to adjust the main scroll area")
    print("ℹ️  Each page has its own ScrollArea; no need to compress the main scroll area")


def _on_page_changed(stack: QStackedWidget, index: int) -> None:
    try:
        current_page = stack.widget(index)
        if current_page is None:
            return
        current_page.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        current_page.updateGeometry()
        page_names = {
            0: "Cut Fitting",
            1: "GIMaP Predict",
            2: "Trainset Build",
            3: "Classification",
        }
        page_name = page_names.get(index, f"Page {index}")
        print(f"Switched to {page_name} page (index: {index})")
    except Exception as exc:
        print(f"Page switch handling failed: {exc}")


class LayoutUtils:
    """Compatibility facade for external callers of the former utility class."""

    setup_adaptive_stacked_widget = staticmethod(configure_adaptive_stack)
    _on_page_changed = staticmethod(_on_page_changed)

    @staticmethod
    def _restore_full_scroll_area(_stack: QStackedWidget) -> None:
        return None

    @staticmethod
    def _compress_scroll_area_for_predict(_stack: QStackedWidget) -> None:
        return None


__all__ = ["LayoutUtils", "configure_adaptive_stack"]
