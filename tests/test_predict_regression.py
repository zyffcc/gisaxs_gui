"""Reproduce individual fit, choose and auto results from the paper's archived benchmarks."""
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PER_CUT = ROOT / "tools/benchmarks/per_cut"


@pytest.mark.parametrize("archive,stage,index", [
    # Known composition: a sphere, and a mixture with two competing solutions.
    pytest.param("posterior_recovery", "fit", 184, id="184"),
    pytest.param("posterior_recovery", "fit", 167, id="167"),
    ("posterior_recovery", "choose", 176),
    ("posterior_newseed_auto2", "auto", 89),
    ("posterior_newseed", "fit", 208),
])
def test_predict_reproduces_archived_fit(archive, stage, index):
    with gzip.open(PER_CUT / f"{archive}.json.gz", "rt", encoding="utf-8") as stream:
        record = json.load(stream)
    expected = next(row[stage] for row in record["results"] if row["case"]["index"] == index)
    # A fresh process matches the benchmark's BLAS thread count and isolates each archive's settings.
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    process = subprocess.run([sys.executable, __file__, stage, str(index)], cwd=ROOT, env=env,
                             input=json.dumps(record["summary"]), capture_output=True, text=True, check=True)
    actual = json.loads(process.stdout)
    assert actual["ok"], actual
    assert set(expected["flags"]) <= set(actual["flags"])
    for key in expected.keys() - {"flags", "seconds"}:
        assert actual[key] == expected[key], key


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    from tools.benchmarks import posterior_recovery as benchmark

    stage, index = sys.argv[1:]
    summary = json.load(sys.stdin)
    folder = ROOT / "src/gimap/features/fitting/domain/posterior/data"
    benchmark._init(str(folder / summary["network"]), summary["settings"])
    assert benchmark.NETWORK.meta["sha256"] == summary["network_sha256"]
    benchmark.OPTIONS["stages"] = [stage]
    print(json.dumps(benchmark.run(int(index))[stage]))
