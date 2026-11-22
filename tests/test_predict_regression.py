"""Reproduce the fit with known composition for two cuts from the paper's first benchmark."""
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "tools/benchmarks/per_cut/posterior_recovery.json.gz"
# One sphere, and a mixture with two competing solutions.
CUTS = (184, 167)


@pytest.fixture(scope="module")
def fits():
    # Match the benchmark's one BLAS thread even if pytest has already imported NumPy.
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    process = subprocess.run([sys.executable, __file__], cwd=ROOT, env=env,
                             capture_output=True, text=True, check=True)
    return json.loads(process.stdout)


@pytest.mark.parametrize("index", CUTS)
def test_predict_reproduces_archived_fit(index, fits):
    with gzip.open(RECORD, "rt", encoding="utf-8") as stream:
        expected = next(row["fit"] for row in json.load(stream)["results"] if row["case"]["index"] == index)
    actual = fits[str(index)]
    assert actual["ok"], actual
    assert set(expected["flags"]) <= set(actual["flags"])
    for key in expected.keys() - {"flags", "seconds"}:
        assert actual[key] == expected[key], key


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    from tools.benchmarks import posterior_recovery as benchmark

    with gzip.open(RECORD, "rt", encoding="utf-8") as stream:
        summary = json.load(stream)["summary"]
    folder = ROOT / "src/gimap/features/fitting/domain/posterior/data"
    benchmark._init(str(folder / summary["network"]), summary["settings"])
    assert benchmark.NETWORK.meta["sha256"] == summary["network_sha256"]
    print(json.dumps({index: benchmark.run(index, fit_only=True)["fit"] for index in CUTS}))
