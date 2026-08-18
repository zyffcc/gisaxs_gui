"""Adaptive per-window monitor and responsive-profile controller."""

from __future__ import annotations

from PyQt5.QtCore import QObject, QEvent, QTimer, pyqtSignal
from PyQt5.QtGui import QScreen
from PyQt5.QtWidgets import QWidget

from src.gimap.app.ports import UserPreferencesRepository

from .responsive_profiles import _preferences_for
from .screen_geometry import (
    apply_window_profile,
    effective_ui_scale,
    profile_for_screen,
    screen_for_window,
)
class AdaptiveWindowProfileController(QObject):
    """Debounced per-window monitor/profile watcher."""

    profileChanged = pyqtSignal(object, object)

    def __init__(
        self,
        window: QWidget,
        callback=None,
        debounce_ms: int = 200,
        apply_window_minimum: bool = True,
        preferences: UserPreferencesRepository | None = None,
    ):
        super().__init__(window)
        self.window = window
        self.preferences = _preferences_for(window, preferences)
        self.callback = callback
        self.apply_window_minimum = apply_window_minimum
        self._screen = None
        self._profile_signature = None
        self._connected_screens = set()
        self._screen_signal_handlers = []
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(debounce_ms)
        self._timer.timeout.connect(self.refresh)
        window.installEventFilter(self)
        QTimer.singleShot(0, self._connect_window_handle)
        QTimer.singleShot(0, self.refresh)

    def eventFilter(self, watched, event):
        # Qt can deliver a final event while the Python wrapper is being torn
        # down and its instance dictionary is already partially cleared.
        if watched is getattr(self, "window", None) and event.type() in (
            QEvent.Move,
            QEvent.Resize,
            QEvent.Show,
            QEvent.MouseButtonRelease,
            QEvent.WindowStateChange,
        ):
            self.schedule()
        return super().eventFilter(watched, event)

    def schedule(self) -> None:
        self._timer.start()

    def _connect_window_handle(self) -> None:
        handle = self.window.windowHandle()
        if handle is None:
            QTimer.singleShot(50, self._connect_window_handle)
            return
        try:
            handle.screenChanged.connect(self._on_screen_changed)
        except TypeError:
            pass
        self._connect_screen(handle.screen())

    def _connect_screen(self, screen: QScreen | None) -> None:
        if screen is None or id(screen) in self._connected_screens:
            return
        self._connected_screens.add(id(screen))
        for signal in (
            screen.geometryChanged,
            screen.availableGeometryChanged,
            screen.logicalDotsPerInchChanged,
        ):
            try:
                handler = lambda *args: self.schedule()
                self._screen_signal_handlers.append(handler)
                signal.connect(handler)
            except TypeError:
                pass

    def _on_screen_changed(self, screen: QScreen) -> None:
        self._connect_screen(screen)
        self.refresh(screen)
        QTimer.singleShot(0, self._force_layout_refresh)
        self.schedule()

    def _force_layout_refresh(self) -> None:
        widget = self.window
        while widget is not None:
            layout = widget.layout()
            if layout is not None:
                layout.invalidate()
                layout.activate()
            widget.updateGeometry()
            widget = widget.parentWidget()

    def refresh(self, forced_screen: QScreen | None = None) -> None:
        screen = forced_screen or screen_for_window(self.window)
        self._connect_screen(screen)
        profile = profile_for_screen(screen, preferences=self.preferences)
        screen_changed = screen is not self._screen
        signature = (
            profile.key,
            profile.min_window.width(),
            profile.min_window.height(),
            profile.content_min,
            profile.workspace_min,
            profile.preview_min,
            round(profile.density_scale, 3),
            effective_ui_scale(self.window, preferences=self.preferences),
        )
        profile_changed = signature != self._profile_signature
        if not screen_changed and not profile_changed:
            return
        self._screen = screen
        self._profile_signature = signature
        if self.apply_window_minimum:
            should_resize = screen_changed and not self.window.isMaximized() and not self.window.isFullScreen()
            apply_window_profile(
                self.window,
                profile,
                resize_window=should_resize,
                screen=screen,
                preferences=self.preferences,
            )
        if self.callback is not None:
            self.callback(profile, screen)
        self.profileChanged.emit(profile, screen)
        QTimer.singleShot(0, self._force_layout_refresh)

def install_adaptive_window_profile(
    window: QWidget,
    callback=None,
    debounce_ms: int = 200,
    apply_window_minimum: bool = True,
    preferences: UserPreferencesRepository | None = None,
):
    controller = AdaptiveWindowProfileController(
        window,
        callback,
        debounce_ms,
        apply_window_minimum,
        preferences,
    )
    window._adaptive_profile_controller = controller
    return controller
