"""Reproduce the paper's five archived fit, choose and auto results.

Windows with Python 3.10.18 produced the archive. There every archived numerical field must
match exactly, apart from elapsed time. Elsewhere compare each composition exactly, its
probability to 0.05 absolute, every population estimate to 1e-2 relative and its interval ends to
2e-2 relative: the least-squares optimum ends where its tolerances stop it and the ends come from
importance weights, and both amplify the differences of floating-point arithmetic between platforms
(Ubuntu CI: up to 1.6e-3 for an estimate, 1.25e-3 for a 95 % end; Windows, Python 3.11: 0.031 in the
probability of one of two competing solutions of cut 167). A change of the method moves them by more. The least-squares errors (keys ending in _err) are compared on the archive platform only: from a
Jacobian at a bound they differ by up to 45 % between platforms (s_nn_err of cut 208 on Ubuntu, Python 3.11).
The portable diagnostics are bic, low_ess, model_error, qzR_* and cyl_sphere. Refinement near
a bound can add or remove flags for boundaries, correlations and unconstrained parameters.
"""
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PER_CUT = ROOT / "tools/benchmarks/per_cut"
ARCHIVE_PLATFORM = sys.platform == "win32" and sys.version_info[:3] == (3, 10, 18)


def _portable_flags(flags):
    return {flag for flag in flags if flag.partition(":")[0] in {"bic", "low_ess", "model_error", "cyl_sphere"}
            or flag.startswith("qzR_")}


def _check_populations(actual, expected):
    for index, (population, reference) in enumerate(zip(actual, expected, strict=True), 1):
        assert population.keys() == reference.keys()
        for key, value in reference.items():
            if key.endswith("_err"):
                continue
            rel = 2e-2 if key.endswith(("_68", "_95")) else 1e-2
            assert population[key] == pytest.approx(value, rel=rel, abs=0), (index, key)


def _check_solution(actual, expected):
    assert actual["families"] == expected["families"]
    assert actual["probability"] == pytest.approx(expected["probability"], abs=0.05)
    _check_populations(actual["populations"], expected["populations"])


@pytest.mark.parametrize("archive,stage,index,added_flags", [
    # Known composition: a sphere, and a mixture with two competing solutions.
    # The first archive predates the qzR_fail warning for a sphere with q_z R > 2.
    pytest.param("posterior_recovery", "fit", 184, {"qzR_fail:R_1"}, id="184"),
    pytest.param("posterior_recovery", "fit", 167, {"qzR_fail:R_2"}, id="167"),
    pytest.param("posterior_recovery", "choose", 176, set(), id="posterior_recovery-choose-176"),
    pytest.param("posterior_newseed_auto2", "auto", 89, set(), id="posterior_newseed_auto2-auto-89"),
    pytest.param("posterior_newseed", "fit", 208, set(), id="posterior_newseed-fit-208"),
])
def test_predict_reproduces_archived_fit(archive, stage, index, added_flags):
    with gzip.open(PER_CUT / f"{archive}.json.gz", "rt", encoding="utf-8") as stream:
        record = json.load(stream)
    expected = next(row[stage] for row in record["results"] if row["case"]["index"] == index)
    # A fresh process matches the benchmark's BLAS thread count and isolates each archive's settings.
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    process = subprocess.run([sys.executable, __file__, stage, str(index)], cwd=ROOT, env=env,
                             input=json.dumps(record["summary"]), capture_output=True, text=True, check=True)
    actual = json.loads(process.stdout)
    assert actual["ok"], actual
    expected_flags = set(expected["flags"]) | added_flags
    assert _portable_flags(actual["flags"]) == _portable_flags(expected_flags)
    for key in ("composition", "composition_index", "in_set"):
        assert actual[key] == expected[key], key
    _check_solution(actual, expected)
    _check_populations(actual["posterior"], expected["posterior"])
    # The second of two competing solutions is a different local optimum on another platform (cut 167 on
    # Windows CI: R 3.02 against 3.18 nm), so across platforms only the most probable solution is compared;
    # on the archive platform every alternative is compared bit for bit below.
    assert len(actual["alternatives"]) == len(expected["alternatives"])

    if ARCHIVE_PLATFORM:
        assert set(actual["flags"]) == expected_flags
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
