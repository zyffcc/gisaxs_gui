"""Legacy spin-box signal behavior retained by the Fitting trigger manager."""

from __future__ import annotations

from PyQt5.QtCore import QTimer


class LegacyParameterTriggerMixin:
    """Compatibility API for callers registered before meta-driven triggers."""

    def _setup_widget_signals(self, widget, widget_id: str, handler_info: dict):
        widget.editingFinished.connect(
            lambda: self._on_immediate_trigger(widget_id, widget.value())
        )
        wheel_timer = QTimer()
        wheel_timer.setSingleShot(True)
        wheel_timer.timeout.connect(
            lambda: self._on_delayed_trigger(widget_id, widget.value())
        )
        self._wheel_timers[widget_id] = wheel_timer
        widget.valueChanged.connect(
            lambda value: self._on_value_changed_with_delay(
                widget_id, value, handler_info
            )
        )

    def _on_immediate_trigger(self, widget_id: str, value):
        handler_info = self._parameter_handlers.get(widget_id)
        if handler_info:
            try:
                handler_info["immediate_handler"](value)
                self._trigger_immediate_save(handler_info["category"])
            except Exception as exc:
                print(f"Error in immediate trigger for {widget_id}: {exc}")

    def _on_value_changed_with_delay(self, widget_id: str, _value, handler_info: dict):
        try:
            if widget_id in self._wheel_timers:
                timer = self._wheel_timers[widget_id]
                timer.stop()
                timer.start(handler_info["wheel_delay"])
        except Exception as exc:
            print(f"Error in delayed trigger setup for {widget_id}: {exc}")

    def _on_delayed_trigger(self, widget_id: str, value):
        handler_info = self._parameter_handlers.get(widget_id)
        if handler_info:
            try:
                handler_info["delayed_handler"](value)
                self._trigger_delayed_save(
                    handler_info["category"], handler_info["save_delay"]
                )
            except Exception as exc:
                print(f"Error in delayed trigger for {widget_id}: {exc}")

    def _trigger_immediate_save(self, _category: str):
        return None

    def _trigger_delayed_save(self, category: str, delay: int):
        if category not in self._save_timers:
            timer = QTimer()
            timer.setSingleShot(True)
            timer.timeout.connect(lambda: self._execute_delayed_save(category))
            self._save_timers[category] = timer
        timer = self._save_timers[category]
        timer.stop()
        timer.start(delay)

    def _execute_delayed_save(self, _category: str):
        return None

    def set_global_delays(self, wheel_delay: int, save_delay: int):
        self.wheel_delay = wheel_delay
        self.save_delay = save_delay

    def is_widget_registered(self, widget_id: str) -> bool:
        return widget_id in self._parameter_handlers


__all__ = ["LegacyParameterTriggerMixin"]
