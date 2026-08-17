"""Generic plot shell with toolbar and empty presentation。"""

from __future__ import annotations

from PyQt5.QtWidgets import QHBoxLayout, QStackedLayout, QVBoxLayout, QWidget

from ..styles import apply_design_system
from .feedback import EmptyState
from .sections import ParameterSection


class PlotPanel(ParameterSection):
    """Presentation shell for an externally supplied plot/canvas widget。"""

    def __init__(
        self,
        title: str = "Preview",
        description: str = "",
        parent: QWidget | None = None,
        *,
        empty_title: str = "No preview yet",
        empty_message: str = "Choose an input to begin.",
    ) -> None:
        super().__init__(title, description, parent)
        self.toolbar_widget = QWidget(self.content)
        self.toolbar_layout = QHBoxLayout(self.toolbar_widget)
        self.toolbar_layout.setContentsMargins(0, 0, 0, 0)
        self.toolbar_layout.setSpacing(6)
        self.add_widget(self.toolbar_widget)

        self.plot_host = QWidget(self.content)
        self.plot_stack = QStackedLayout(self.plot_host)
        self.plot_stack.setContentsMargins(0, 0, 0, 0)
        self.empty_state = EmptyState(empty_title, empty_message, self.plot_host)
        self.plot_stack.addWidget(self.empty_state)
        self.add_widget(self.plot_host, 1)
        self._plot_widget: QWidget | None = None
        apply_design_system(self)

    def add_toolbar_widget(self, widget: QWidget) -> None:
        self.toolbar_layout.addWidget(widget)

    def add_toolbar_stretch(self) -> None:
        self.toolbar_layout.addStretch(1)

    def set_plot_widget(self, widget: QWidget) -> None:
        if self._plot_widget is widget:
            self.show_plot()
            return
        if self._plot_widget is not None:
            self.plot_stack.removeWidget(self._plot_widget)
            self._plot_widget.setParent(None)
        self._plot_widget = widget
        widget.setParent(self.plot_host)
        self.plot_stack.addWidget(widget)
        self.show_plot()

    def show_empty(self, title: str | None = None, message: str | None = None) -> None:
        if title is not None or message is not None:
            self.empty_state.set_content(
                title if title is not None else self.empty_state.title_label.text(),
                message if message is not None else self.empty_state.message_label.text(),
            )
        self.plot_stack.setCurrentWidget(self.empty_state)

    def show_plot(self) -> None:
        if self._plot_widget is None:
            self.plot_stack.setCurrentWidget(self.empty_state)
        else:
            self.plot_stack.setCurrentWidget(self._plot_widget)
