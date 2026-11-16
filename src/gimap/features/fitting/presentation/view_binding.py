"""What the application keeps of Fitting outside its pages: 1D Predict, the project's parameters and the
session saved on close.

The pages themselves (Single analysis, ``single/page.py``; In-situ series, ``single/series_page.py``) are
built by ``FittingWorkspace`` and run their own fits. This binding opens the 1D Predict dialog (Fitting ▸
1D Predict, and the page's button) on the curve of Single analysis: its points as the page shows them —
the halves chosen, the fitting range and the left-out points — read again from the file, so that a curve
Analyze wrote keeps the pixel counts of its detector columns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QWidget

from ..application import LoadCurveRequest
from ..application.single_fit import left_out_points
from ..application.workflow_v5 import default_options
from .workflow_v5_dialog import WorkflowV5Dialog

__all__ = ["FittingViewBinding", "native_curve_input"]


def _create_default_fitting_view_model():
    """The standalone composition, only when no view model was injected (tests, scripts)."""
    from src.gimap.app.bootstrap import create_standalone_legacy_context
    from src.gimap.features.fitting.bootstrap import create_fitting_view_model

    return create_fitting_view_model(create_standalone_legacy_context())


def native_curve_input(data: dict, options: dict):
    """Native detector columns written by Analyze: the counting contract of the CBF path.

    Analyze averages each detector column of the horizontal cut band (mean counts per pixel, Poisson σ,
    pixel count), so V5 gets that observation's metadata and the same working tolerance on σ. Returns
    ``(data with σ, metadata)``, or ``None`` for other curves."""
    observation = data.get("observation") or {}
    pixels, counting = data.get("pixels"), data.get("err")
    if observation.get("source") != "native_detector_columns" or pixels is None or counting is None:
        return None
    q = np.asarray(data.get("q", []), float).reshape(-1)
    y = np.asarray(data.get("I", []), float).reshape(-1)
    pixels = np.asarray(pixels, float).reshape(-1)
    counting = np.asarray(counting, float).reshape(-1)
    if not (q.shape == y.shape == pixels.shape == counting.shape) or not q.size:
        return None
    cbf = observation.get("file_format") == "cbf"
    metadata = dict(
        source="native_cbf_columns" if cbf else "native_detector_columns",
        q_source="analyze_native_columns",
        measured_columns=int(q.size),
        valid_pixel_counts=pixels.tolist(),
        intensity_unit=observation.get("intensity_unit", "counts_per_pixel"),
        gap_margin_px=int(observation.get("gap_guard_px", 0)),
        threshold_enabled=bool(observation.get("threshold_enabled", False)),
        counting_model_valid=bool(observation.get("counting_model_valid", True)),
        mirror_replaced_pixels=0,
        stack_count=max(1, int(observation.get("summed_frames", 1))),
        uncertainty="Poisson sum/count approximation; additional relative noise is a working tolerance",
        sampling="Native measured columns; no interpolated points across detector gaps",
    )
    sigma = np.sqrt(counting**2 + (options["relative_noise"] * abs(y)) ** 2 + options["absolute_noise"] ** 2)
    return {**data, "err": sigma}, metadata


class FittingViewBinding(QObject):
    """1D Predict, parameters and session of Fitting."""

    status_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    parameters_changed = pyqtSignal(dict)

    def __init__(self, ui, parent=None, fitting_view_model=None):
        super().__init__(parent)
        self.ui = ui
        self.parent = parent
        parent_widget = parent if isinstance(parent, QWidget) else None
        if parent_widget is None and parent is not None:
            getter = getattr(parent, "parent", None)
            candidate = getter() if callable(getter) else getter
            parent_widget = candidate if isinstance(candidate, QWidget) else None
        self.main_window = parent_widget or (ui if isinstance(ui, QWidget) else None)
        self.fitting_view_model = fitting_view_model or _create_default_fitting_view_model()
        self.preferences = self.fitting_view_model.context.preferences
        self.model_params_manager = self.fitting_view_model.storage.model_parameters
        if self.model_params_manager is None:
            raise ValueError("FittingViewBinding requires model parameter storage")
        self.current_parameters = {"fitting_params": {}}
        self._workflow_v5_dialog = None
        self._workflow_observation_metadata = {}
        self._workflow_sigma_estimated = False
        self._initialized = False

    def initialize(self) -> None:
        """Connect the 1D Predict button of Single analysis (once the workspace is built)."""
        if self._initialized:
            return
        page = self._fit_page()
        if page is not None:
            page.batchRequested.connect(self.open_ai_fitting_workspace)
        self._initialized = True

    def _fit_page(self):
        workspace = getattr(self.ui, "fittingWorkspace", None)
        return getattr(workspace, "fit_page", None)

    # -- parameters of a project -----------------------------------------------------------------

    def get_parameters(self) -> dict:
        return self.current_parameters.copy()

    def set_parameters(self, parameters) -> None:
        self.current_parameters.update(parameters)
        self.parameters_changed.emit(self.current_parameters)

    def reload_particle_parameters(self) -> bool:
        """Read the stored model parameters again (after a project replaced them)."""
        return bool(self.model_params_manager.load_parameters())

    def reset_to_defaults(self) -> None:
        self.model_params_manager.reset_to_defaults()
        self.current_parameters = {"fitting_params": {}}
        self.parameters_changed.emit(self.current_parameters)

    # -- the session saved on close ------------------------------------------------------------------

    def get_imported_file(self) -> str:
        """The curve file of Single analysis (``""`` when none)."""
        curve = getattr(getattr(self._fit_page(), "session", None), "curve", None)
        return str(getattr(curve, "path", "") or "")

    def get_session_data(self) -> dict:
        session = {"ai_fitting": {"workflow_v5": self._workflow_options()}}
        path = self.get_imported_file()
        if path:
            session["last_1d_file"] = path
        return session

    def restore_session(self, session) -> None:
        """The 1D Predict settings of a saved session (the curve is the page's own to reopen)."""
        stored = session.get("ai_fitting") if isinstance(session, dict) else None
        options = stored.get("workflow_v5") if isinstance(stored, dict) else None
        if isinstance(options, dict):
            self._save_workflow_options(options)

    # -- 1D Predict ------------------------------------------------------------------------------

    def _ai_fitting_settings(self) -> dict:
        settings = self.preferences.get("ai_fitting", {})
        return settings if isinstance(settings, dict) else {}

    def _workflow_options(self) -> dict:
        return {**default_options(), **self._ai_fitting_settings().get("workflow_v5", {})}

    def _save_workflow_options(self, options) -> None:
        settings = self._ai_fitting_settings()
        settings["workflow_v5"] = dict(options)
        self.preferences.set("ai_fitting", settings)
        self.preferences.save()

    def open_ai_fitting_workspace(self) -> None:
        dialog = self._workflow_v5_dialog
        if dialog is None:
            dialog = WorkflowV5Dialog(
                self._current_ai_curve_arrays,
                self._workflow_options(),
                self.main_window,
                runner=self.fitting_view_model.context.jobs,
                stated_q_unit=self.fitting_view_model.stated_q_unit,  # the unit of q each file's header states
            )
            dialog.settings_changed.connect(self._save_workflow_options)
            dialog.sigma_estimated = lambda: self._workflow_sigma_estimated
            dialog.observation_metadata = lambda: self._workflow_observation_metadata
            self._workflow_v5_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _current_curve_data(self):
        """The curve of Single analysis as its file has it (``q`` in the file's unit), with the page's
        selection of its points: ``(data, side, q_range, excluded)``, or ``None`` without a curve."""
        session = getattr(self._fit_page(), "session", None)
        curve = getattr(session, "curve", None)
        if curve is None or not curve.path:
            return None
        outcome = self.fitting_view_model.load_curve(LoadCurveRequest(Path(curve.path), curve.source_unit))
        if outcome.error is not None:
            raise ValueError(outcome.error.message)
        loaded = outcome.value
        data = {"q": loaded.q, "I": loaded.intensity, "err": loaded.error, "pixels": loaded.pixels,
                "q_source_unit": loaded.q_source_unit, "observation": dict(loaded.observation or {})}
        return data, session.side, session.q_range, set(session.excluded)

    def _current_ai_curve_arrays(self):
        """``(q in nm⁻¹, I, σ)`` of the points of Single analysis that 1D Predict fits, or ``None``.

        The measured sign and intensity are kept (V5 folds the halves itself); the halves chosen on the
        page select the sign, its fitting range (|q|) and its left-out points select the columns."""
        self._workflow_observation_metadata = {}
        current = self._current_curve_data()
        if current is None:
            return None
        data, side, q_range, excluded = current
        options = self._workflow_options()
        native = native_curve_input(data, options)
        if native is not None:
            data, self._workflow_observation_metadata = native
        q_file = np.asarray(data["q"], float).reshape(-1)
        y = np.asarray(data["I"], float).reshape(-1)
        if q_file.shape != y.shape or not q_file.size:
            return None
        q = q_file * (10.0 if data.get("q_source_unit") == "angstrom" else 1.0)
        mask = np.isfinite(q) & np.isfinite(y) & (q != 0)
        if side == "positive":
            mask &= q > 0
        elif side == "negative":
            mask &= q < 0
        if q_range is not None:
            low, high = sorted(float(value) for value in q_range)
            mask &= (np.abs(q) >= low) & (np.abs(q) <= high)
        if excluded:
            mask &= ~left_out_points(np.abs(q), excluded)
        if not mask.any():
            return None
        metadata = self._workflow_observation_metadata
        if metadata:
            counts = np.asarray(metadata["valid_pixel_counts"], dtype=float)
            metadata.update(
                valid_pixel_counts=counts[mask].tolist(),
                selected_points=int(mask.sum()),
                selected_positive=int(np.count_nonzero(mask & (q > 0))),
                selected_negative=int(np.count_nonzero(mask & (q < 0))),
                input_selection=dict(side=side, q_range=None if q_range is None else list(q_range),
                                     excluded_q=sorted(excluded)),
            )
        sigma = data.get("err")
        self._workflow_sigma_estimated = sigma is None or bool(metadata)
        if sigma is None:
            floor = options["absolute_noise"] or max(float(np.max(abs(y[mask]))) * 0.001, 1e-12)
            sigma = np.hypot(options["relative_noise"] * abs(y), floor)
        sigma = np.asarray(sigma, float).reshape(-1)
        if sigma.shape != q.shape:
            raise ValueError("Input sigma length does not match q")
        return q[mask], y[mask], sigma[mask]
