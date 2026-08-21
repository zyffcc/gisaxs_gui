"""Bind Fitting numeric increments to persistent UI preferences."""

from __future__ import annotations

import math

from PyQt5.QtWidgets import QDoubleSpinBox

from src.gimap.app.ports import UserPreferencesRepository


class ParameterStepPreferences:
    """Own the UI-only defaults behind Fitting step spin boxes."""

    KEY_PREFIX = "fitting.parameter_step"

    def __init__(self, repository: UserPreferencesRepository):
        self.repository = repository

    def bind(
        self,
        step_spinbox: QDoubleSpinBox,
        value_spinbox: QDoubleSpinBox,
        preference_name: str,
        built_in_default: float,
    ) -> None:
        preference_key = f"{self.KEY_PREFIX}.{preference_name}"
        saved_value = self._valid_value(
            self.repository.get(preference_key, built_in_default),
            built_in_default,
        )
        step_spinbox.setDecimals(8)
        step_spinbox.setRange(1e-9, 1e9)
        step_spinbox.setSingleStep(built_in_default)
        step_spinbox.setProperty("defaultStepValue", built_in_default)
        step_spinbox.setProperty("preferenceKey", preference_key)
        step_spinbox.setValue(saved_value)
        value_spinbox.setSingleStep(saved_value)
        step_spinbox.valueChanged.connect(
            lambda value, target=value_spinbox: target.setSingleStep(float(value))
        )
        step_spinbox.editingFinished.connect(
            lambda spinbox=step_spinbox: self.save(spinbox)
        )

    def save(self, step_spinbox: QDoubleSpinBox) -> None:
        preference_key = step_spinbox.property("preferenceKey")
        if not preference_key:
            return
        self.repository.set(str(preference_key), float(step_spinbox.value()))
        self.repository.save()

    def reset(self, step_spinbox: QDoubleSpinBox) -> None:
        default_value = step_spinbox.property("defaultStepValue")
        if default_value is None:
            return
        step_spinbox.setValue(float(default_value))
        self.save(step_spinbox)

    @staticmethod
    def _valid_value(value, fallback: float) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return fallback
        return value if math.isfinite(value) and value > 0 else fallback


__all__ = ["ParameterStepPreferences"]
