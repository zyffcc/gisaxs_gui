"""Cross-workspace parameter coordination at the application composition boundary."""

from __future__ import annotations

from collections.abc import Callable

from .project_parameters import LoadProjectParameters, SaveProjectParameters


class WorkspaceParameterCoordinator:
    """Coordinate public parameter APIs without knowing feature internals."""

    def __init__(
        self,
        *,
        repository,
        trainset,
        fitting,
        classification,
        prediction,
        status: Callable[[str], None],
    ):
        self.trainset = trainset
        self.fitting = fitting
        self.classification = classification
        self.prediction = prediction
        self.status = status
        self._load = LoadProjectParameters(repository) if repository is not None else None
        self._save = SaveProjectParameters(repository) if repository is not None else None

    def snapshot(self) -> dict:
        return {
            "trainset": self.trainset.get_parameters(),
            "fitting": self.fitting.get_parameters(),
            "fitting_model_parameters": {
                "fitting": self.fitting.model_params_manager.get_parameter(
                    "fitting", None, {}
                )
            }
            if hasattr(self.fitting, "model_params_manager")
            else {},
            "classification": self.classification.get_parameters(),
            "gisaxs_predict": self.prediction.get_parameters(),
        }

    def load(self, file_path) -> bool:
        try:
            if self._load is None:
                raise RuntimeError("Project parameter storage is unavailable")
            parameters = self._load.execute(file_path)
            if "trainset" in parameters:
                self.trainset.set_parameters(parameters["trainset"])
            if "fitting" in parameters:
                self.fitting.set_parameters(parameters["fitting"])
            self._restore_fitting_model_parameters(parameters)
            if "classification" in parameters:
                self.classification.set_parameters(parameters["classification"])
            if "gisaxs_predict" in parameters:
                self.prediction.set_parameters(parameters["gisaxs_predict"])
            self.status(f"Parameters loaded from {file_path} successfully")
            return True
        except Exception as exc:
            self.status(f"Failed to load parameters: {exc}")
            return False

    def save(self, file_path) -> bool:
        try:
            if self._save is None:
                raise RuntimeError("Project parameter storage is unavailable")
            saved_path = self._save.execute(file_path, self.snapshot())
            self.status(f"Parameters saved to {saved_path} successfully")
            return True
        except Exception as exc:
            self.status(f"Failed to save parameters: {exc}")
            return False

    def validate(self) -> list[tuple[str, bool, str]]:
        results = []
        for name, runtime in (
            ("Trainset parameters", self.trainset),
            ("Fitting parameters", self.fitting),
            ("Classification parameters", self.classification),
            ("GISAXS prediction parameters", self.prediction),
        ):
            if hasattr(runtime, "validate_parameters"):
                is_valid, message = runtime.validate_parameters()
                results.append((name, is_valid, message))
        return results

    def reset(self) -> None:
        for runtime in (
            self.trainset,
            self.fitting,
            self.classification,
            self.prediction,
        ):
            runtime.reset_to_defaults()
        self.status("All parameters have been reset to default values")

    def _restore_fitting_model_parameters(self, parameters: dict) -> None:
        model_parameters = parameters.get("fitting_model_parameters") or {}
        fitting_model = (
            model_parameters.get("fitting")
            if isinstance(model_parameters, dict)
            else None
        )
        if not isinstance(fitting_model, dict) or not hasattr(
            self.fitting, "model_params_manager"
        ):
            return
        self.fitting.model_params_manager.replace_section("fitting", fitting_model)
        self.fitting.model_params_manager.save_parameters()
        self.fitting.reload_particle_parameters()


__all__ = ["WorkspaceParameterCoordinator"]
