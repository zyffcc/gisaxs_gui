"""Application composition, navigation and feature lifecycle coordination."""

from pathlib import Path

from PyQt5.QtCore import QObject, QTimer, pyqtSignal
from PyQt5.QtWidgets import QMainWindow

from .fitting_session import FittingSessionCoordinator
from .presentation.parameter_validation import show_parameter_validation
from .presentation.workspace_feedback import show_workspace_unavailable
from .workspace_parameters import WorkspaceParameterCoordinator


class ApplicationRuntime(QObject):
    """Connect the application shell to feature-owned presentation runtimes."""

    status_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(int)

    def __init__(self, ui, parent=None, *, simulation_port=None):
        super().__init__(parent)
        self.ui = ui
        self.parent = parent
        self.current_parameters = {}
        self.app_context = getattr(parent, "app_context", None) or getattr(
            ui, "app_context", None
        )
        if self.app_context is None:
            raise ValueError("ApplicationRuntime requires AppContext")
        if simulation_port is None:
            raise ValueError("ApplicationRuntime requires an injected simulation port")
        components = getattr(ui, "components", None)
        if components is None:
            raise ValueError("ApplicationRuntime requires composed feature pages")

        self.settings = self.app_context.settings
        self._compose_features(components, simulation_port)
        self.fitting_session = FittingSessionCoordinator(
            self.settings, self.fitting
        )
        self.workspace_parameters = WorkspaceParameterCoordinator(
            repository=self.app_context.project_parameters,
            trainset=self.trainset,
            fitting=self.fitting,
            classification=self.classification,
            prediction=self.prediction,
            status=self.status_updated.emit,
        )
        self._setup_connections()
        QTimer.singleShot(200, self._delayed_feature_initialization)

    def _compose_features(self, components, simulation_port) -> None:
        """Construct feature presentation runtimes with injected dependencies."""
        from src.gimap.features.classification.bootstrap import (
            create_classification_view_model,
        )
        from src.gimap.features.classification.presentation.view_binding import (
            ClassificationViewBinding,
        )
        from src.gimap.features.fitting.bootstrap import create_fitting_view_model
        from src.gimap.features.fitting.presentation.view_binding import (
            FittingViewBinding,
        )
        from src.gimap.features.prediction.bootstrap import create_prediction_view_model
        from src.gimap.features.prediction.presentation.view_binding import (
            PredictionViewBinding,
        )
        from src.gimap.features.trainset.bootstrap import create_trainset_view_model
        from src.gimap.features.trainset.presentation.view_binding import (
            TrainsetViewBinding,
        )

        self.trainset = TrainsetViewBinding(
            self.ui,
            self,
            simulation_port=simulation_port,
            trainset_view_model=create_trainset_view_model(
                self.app_context, simulation_port
            ),
            page=components.trainset_page,
            project_root=Path(__file__).resolve().parents[3],
        )
        self.fitting = FittingViewBinding(
            self.ui,
            self,
            fitting_view_model=getattr(
                components,
                "fitting_view_model",
                None,
            )
            or create_fitting_view_model(self.app_context),
        )
        self.classification = ClassificationViewBinding(
            self.ui,
            self,
            classification_view_model=create_classification_view_model(
                self.app_context
            ),
            page=components.classification_page,
        )
        self.prediction = PredictionViewBinding(
            self.ui,
            self,
            prediction_view_model=create_prediction_view_model(self.app_context),
        )

    def _delayed_feature_initialization(self) -> None:
        """Initialize feature presentation only after the shell is visible."""
        try:
            print("Application runtime: Starting delayed feature initialization...")
            self._initialize_ui()
            self.trainset.initialize()
            self.fitting.initialize()
            self.classification.initialize()
            self.prediction.initialize()
            QTimer.singleShot(1000, self.fitting_session.load_last_session)
            print("Application runtime: Feature initialization complete")
        except Exception as exc:
            print(f"Application runtime: Feature initialization failed: {exc}")

    def _setup_connections(self) -> None:
        self.ui.trainsetBuildButton.clicked.connect(self._switch_to_trainset_build)
        self.ui.gisaxsPredictButton.clicked.connect(self._switch_to_gisaxs_predict)
        self.ui.cutAndFittingButton.clicked.connect(self._switch_to_cut_fitting)
        self.ui.ClassficationButton.clicked.connect(self._switch_to_classification)
        try:
            self.ui.WAXSButton.clicked.connect(self._switch_to_waxs)
        except Exception:
            pass

        trainset = self.trainset
        trainset.parameters_changed.connect(self._on_parameters_changed)
        trainset.generation_started.connect(
            lambda: self.status_updated.emit("Trainset generation started...")
        )
        trainset.generation_finished.connect(
            lambda: self.status_updated.emit("Trainset generation completed!")
        )
        trainset.progress_updated.connect(self.progress_updated)
        trainset.prediction_module_registered.connect(
            self._open_registered_prediction_module
        )

        fitting = self.fitting
        fitting.parameters_changed.connect(
            lambda params: self._on_parameters_changed("Fitting parameters", params)
        )
        fitting.status_updated.connect(self.status_updated)
        fitting.progress_updated.connect(self.progress_updated)

        classification = self.classification
        classification.parameters_changed.connect(
            lambda params: self._on_parameters_changed(
                "Classification parameters", params
            )
        )
        classification.status_updated.connect(self.status_updated)
        classification.progress_updated.connect(self.progress_updated)
        classification.classification_completed.connect(
            self._on_classification_completed
        )

        prediction = self.prediction
        prediction.parameters_changed.connect(
            lambda params: self._on_parameters_changed(
                "GISAXS prediction parameters", params
            )
        )
        prediction.status_updated.connect(self.status_updated)
        prediction.progress_updated.connect(self.progress_updated)

    def _initialize_ui(self) -> None:
        self.ui.mainWindowWidget.setCurrentIndex(2)
        self._update_button_states(0)
        self.status_updated.emit("GIMaP ready")

    def _open_registered_prediction_module(self, module_name: str) -> None:
        prediction = self.prediction
        prediction._refresh_modules()
        combo = getattr(self.ui, "gisaxsPredictModuleSelectCombox", None)
        if combo is not None and combo.findText(module_name) >= 0:
            combo.setCurrentText(module_name)
        prediction._on_module_selected(module_name)
        self._switch_to_gisaxs_predict()

    def _switch_to_waxs(self) -> None:
        page_index = getattr(self.ui, "waxsPageIndex", None)
        if page_index is None:
            show_workspace_unavailable(
                self.parent if isinstance(self.parent, QMainWindow) else None,
                "WAXS",
                "Embedded WAXS page is not available.",
            )
            return
        self.ui.mainWindowWidget.setCurrentIndex(page_index)
        self.status_updated.emit("Switched to WAXS / in-situ processing mode")
        self._update_button_states(4)

    def _switch_to_cut_fitting(self) -> None:
        self.ui.mainWindowWidget.setCurrentIndex(2)
        self.status_updated.emit("Switched to Cut Fitting mode")
        self._update_button_states(0)
        if not self.fitting._initialized:
            self.fitting.initialize()

    def _switch_to_gisaxs_predict(self) -> None:
        self.ui.mainWindowWidget.setCurrentIndex(1)
        self.status_updated.emit("Switched to GISAXS prediction mode")
        self._update_button_states(1)
        if not self.prediction._initialized:
            self.prediction.initialize()

    def _switch_to_trainset_build(self) -> None:
        self.ui.mainWindowWidget.setCurrentIndex(0)
        self.status_updated.emit("Switched to Trainset Build mode")
        self._update_button_states(2)

    def _switch_to_classification(self) -> None:
        self.ui.mainWindowWidget.setCurrentIndex(3)
        self.status_updated.emit("Switched to Classification mode")
        self._update_button_states(3)
        if not self.classification._initialized:
            self.classification.initialize()

    def _on_parameters_changed(self, module_name, parameters) -> None:
        self.current_parameters[module_name] = parameters
        self.status_updated.emit(f"{module_name} parameters updated")

    def get_all_parameters(self) -> dict:
        return self.workspace_parameters.snapshot()

    def load_parameters_from_file(self, file_path) -> bool:
        return self.workspace_parameters.load(file_path)

    def save_parameters_to_file(self, file_path) -> bool:
        return self.workspace_parameters.save(file_path)

    def validate_all_parameters(self) -> list[tuple[str, bool, str]]:
        return self.workspace_parameters.validate()

    def show_validation_results(self) -> None:
        show_parameter_validation(self.parent, self.validate_all_parameters())

    def reset_all_parameters(self) -> None:
        self.workspace_parameters.reset()

    def save_current_session(self) -> None:
        self.fitting_session.save()

    def save_session_on_close(self) -> None:
        self.save_current_session()
        print("Application runtime: Session saved on close")

    def handle_window_close(self) -> None:
        self.save_session_on_close()

    def _on_classification_completed(self, results) -> None:
        self.status_updated.emit(
            f"Classification completed, processed {len(results)} items"
        )

    def _update_button_states(self, active_index: int) -> None:
        self.ui.components.sidebar.set_active_index(active_index)

    @property
    def trainset_controller(self):
        """Deprecated runtime attribute retained for third-party integrations."""
        return self.trainset

    @property
    def classification_controller(self):
        """Deprecated runtime attribute retained for third-party integrations."""
        return self.classification

    @property
    def gisaxs_predict_controller(self):
        """Deprecated runtime attribute retained for third-party integrations."""
        return self.prediction

    @property
    def fitting_controller(self):
        """Deprecated runtime attribute retained for third-party integrations."""
        return self.fitting


__all__ = ["ApplicationRuntime"]
