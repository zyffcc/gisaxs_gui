import hashlib
import subprocess
import sys

import numpy as np
import pytest

from src.gimap.app.jobs import JobResult
from src.gimap.integrations.bornagain import (
    BornAgainNotInstalledError,
    BornAgainSimulator,
    BornAgainVersion,
)
from trainset.config import default_project_config, synchronize_parameter_specs
from trainset.generator import DatasetGenerator
from trainset.simulation import simulate_pattern


class UnavailableRunner:
    def run(self, request, on_progress=None):
        del on_progress
        return JobResult(
            job_id=request.job_id,
            status="succeeded",
            value={
                "state": "not_installed",
                "message": "BornAgain is not installed in this Python environment.",
            },
        )

    def cancel(self, _job_id):
        return False

    def shutdown(self):
        return None


def _small_config():
    config = synchronize_parameter_specs(default_project_config())
    config["roi"].update({"x": 0, "y": 0, "width": 32, "height": 32})
    config["detector"].update(
        {
            "nbins_x": 32,
            "nbins_y": 32,
            "beam_center_x": 16.0,
            "beam_center_y": 16.0,
        }
    )
    return config


def test_bornagain_version_accepts_24_1_and_rejects_other_minor() -> None:
    assert BornAgainVersion.parse("24.1").supported
    assert BornAgainVersion.parse("24.1.3").supported
    assert not BornAgainVersion.parse("24.2").supported


def test_importing_integration_does_not_import_bornagain_in_gui_process() -> None:
    code = (
        "import sys; "
        "import src.gimap.integrations.bornagain; "
        "assert 'bornagain' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_unavailable_runtime_has_specific_error() -> None:
    simulator = BornAgainSimulator(runner=UnavailableRunner())
    assert simulator.availability().state == "not_installed"
    with pytest.raises(BornAgainNotInstalledError, match="not installed"):
        simulator.simulate({}, {})


def test_bornagain_24_1_minimal_32x32_simulation_matches_baseline() -> None:
    simulator = BornAgainSimulator(simulation_timeout_seconds=60.0)
    availability = simulator.availability()
    if not availability.available:
        pytest.skip(availability.message)
    config = _small_config()
    generator = DatasetGenerator(config, simulation_port=simulator)
    sampled = generator.sample_parameters(1)[0]

    image = simulate_pattern(config, sampled, simulator=simulator)

    assert image.shape == (32, 32)
    assert image.dtype == np.float32
    assert np.all(np.isfinite(image))
    digest = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
    assert digest == "d46e15830e91b3eb63c1ac655918c2553548149cecb078071422337a07b7acc8"
