"""BornAgain SimulationPort process adapter。"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...app.jobs import JobRequest, JobRunner
from ...features.trainset.application.ports import SimulationPort
from ..jobs import LocalProcessJobRunner, decode_array
from .availability import BornAgainAvailability
from .errors import (
    BornAgainBrokenError,
    BornAgainError,
    BornAgainNotInstalledError,
    BornAgainUnsupportedVersionError,
)


class BornAgainSimulator(SimulationPort):
    def __init__(
        self,
        runner: JobRunner | None = None,
        simulation_timeout_seconds: float = 300.0,
    ):
        self.runner = runner or LocalProcessJobRunner()
        self.simulation_timeout_seconds = simulation_timeout_seconds
        self._availability: BornAgainAvailability | None = None

    def availability(self, refresh: bool = False) -> BornAgainAvailability:
        if self._availability is not None and not refresh:
            return self._availability
        result = self.runner.run(
            JobRequest(
                handler="src.gimap.integrations.bornagain.worker:probe_bornagain",
                timeout_seconds=15.0,
            )
        )
        if result.succeeded:
            self._availability = BornAgainAvailability.from_dict(result.value)
        else:
            message = result.error.message if result.error is not None else "Unknown probe failure."
            self._availability = BornAgainAvailability(
                state="broken",
                message=f"BornAgain probe worker failed: {message}",
            )
        return self._availability

    def is_available(self) -> bool:
        return self.availability().available

    def simulate(
        self,
        config: dict[str, Any],
        sampled: dict[str, Any],
    ) -> np.ndarray:
        availability = self.availability()
        self._raise_if_unavailable(availability)
        result = self.runner.run(
            JobRequest(
                handler="src.gimap.integrations.bornagain.worker:simulate_bornagain",
                payload={"config": config, "sampled": sampled},
                timeout_seconds=self.simulation_timeout_seconds,
            )
        )
        if not result.succeeded:
            message = result.error.message if result.error is not None else "Unknown simulation failure."
            error_type = result.error.exception_type if result.error is not None else ""
            mapping = {
                "BornAgainNotInstalledError": BornAgainNotInstalledError,
                "BornAgainUnsupportedVersionError": BornAgainUnsupportedVersionError,
                "BornAgainBrokenError": BornAgainBrokenError,
            }
            raise mapping.get(error_type, BornAgainBrokenError)(message)
        return np.asarray(decode_array(result.value), dtype=np.float32)

    @staticmethod
    def _raise_if_unavailable(availability: BornAgainAvailability) -> None:
        if availability.available:
            return
        errors = {
            "not_installed": BornAgainNotInstalledError,
            "unsupported": BornAgainUnsupportedVersionError,
            "broken": BornAgainBrokenError,
        }
        raise errors.get(availability.state, BornAgainError)(availability.message)
