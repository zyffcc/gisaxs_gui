"""Reproduce five Predict fits on the paper's archived synthetic inputs.

Windows with Python 3.10.18 checks every numerical field exactly, including alternatives.
Other platforms compare the leading composition and branch exactly, the weights to 0.05
absolute, the leading population estimates to 1% relative and their interval ends to 2%.
Where a platform compares the compositions by BIC and the archive by evidence (or the
reverse), the intervals are of another kind: only the estimates away from a bound are compared.
The benchmark README documents the observed platform differences behind these tolerances.
"""
import gzip
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PER_CUT = ROOT / "tools/benchmarks/per_cut"
REFERENCE = ROOT / "tests/data/ml1d/predict_regression.json"
ARCHIVE_PLATFORM = sys.platform == "win32" and sys.version_info[:3] == (3, 10, 18)
CASES = [
    ("posterior_r17_first", "fit", 184),
    ("posterior_r17_first", "fit", 167),
    ("posterior_r17_first", "choose", 176),
    ("posterior_r17_fresh", "auto", 89),
    ("posterior_r17_fresh", "fit", 208),
]


def _portable_flags(flags):
    return {flag for flag in flags if flag.partition(":")[0] in {"bic", "low_ess", "model_error", "cyl_sphere"}
            or flag.startswith("qzR_")}


_POPULATION_KEYS = {"R": "R", "sigma_R": "sR", "D": "D", "sigma_nn": "s_nn"}


def _at_bound(flags):
    """(population number, key) of every population value flagged at a bound (``at_bound:sigma_R_2``)."""
    found = set()
    for flag in flags:
        kind, _, name = flag.partition(":")
        stem, _, number = name.rpartition("_")
        if kind == "at_bound" and stem in _POPULATION_KEYS and number.isdigit():
            found.add((int(number), _POPULATION_KEYS[stem]))
    return found


def _check_populations(actual, expected, skip=None):
    """``skip``: None to compare every value and interval end, else the at-bound values to leave out with the
    interval ends (another comparison method on this platform)."""
    for index, (population, reference) in enumerate(zip(actual, expected, strict=True), 1):
        assert population.keys() == reference.keys()
        for key, value in reference.items():
            interval = key.endswith(("_68", "_95"))
            if key.endswith("_err") or (skip is not None and (interval or (index, key) in skip)):
                continue
            rel = 2e-2 if interval else 1e-2
            assert population[key] == pytest.approx(value, rel=rel, abs=0), (index, key)


def _check_solution(actual, expected, skip=None):
    assert actual["families"] == expected["families"]
    assert actual["probability"] == pytest.approx(expected["probability"], abs=0.05)
    _check_populations(actual["populations"], expected["populations"], skip)


def _run(archive, stage, index):
    with gzip.open(PER_CUT / f"{archive}.json.gz", "rt", encoding="utf-8") as stream:
        summary = json.load(stream)["summary"]
    # A fresh process matches the benchmark's BLAS thread count and isolates each archive's settings.
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    process = subprocess.run([sys.executable, __file__, stage, str(index)], cwd=ROOT, env=env,
                             input=json.dumps(summary), capture_output=True, text=True, check=True)
    actual = json.loads(process.stdout)
    assert actual["ok"], actual
    return actual


@pytest.mark.parametrize("archive,stage,index", CASES)
def test_predict_reproduces_archived_fit(archive, stage, index):
    record = json.loads(REFERENCE.read_text(encoding="utf-8"))
    expected = record["results"][f"{archive}/{stage}/{index}"]
    actual = _run(archive, stage, index)
    expected_flags = set(expected["flags"])
    # Compositions are compared by BIC when one of them drew fewer than 10 effective samples (ESS_FAILED). The
    # pooled ESS of the less probable compositions is not portable: for cut 89 it is 27 and 71 on the archive
    # platform, 4-6 and about 9.6 on most CI computers, 13 and 71 on one Windows CI computer. The flag follows.
    assert ("bic" in actual["flags"]) == (actual["compared_by"] == "bic")
    assert _portable_flags(actual["flags"]) - {"bic"} == _portable_flags(expected_flags) - {"bic"}
    for key in ("composition", "composition_index", "in_set"):
        assert actual[key] == expected[key], key
    assert actual["composition_weights"] == pytest.approx(expected["composition_weights"], abs=0.05)
    if actual["compared_by"] == expected["compared_by"]:
        _check_solution(actual, expected)
        _check_populations(actual["posterior"], expected["posterior"])
    else:
        # BIC intervals come from the least-squares errors, the evidence's from the sampled posterior.
        _check_solution(actual, expected, skip=_at_bound(expected_flags))
    # The second of two competing solutions is a different local optimum on another platform (cut 167 on
    # Windows CI: R 3.02 against 3.18 nm), so across platforms only the most probable solution is compared;
    # on the archive platform every alternative is compared bit for bit below.
    assert len(actual["alternatives"]) == len(expected["alternatives"])

    if ARCHIVE_PLATFORM:
        assert actual.keys() - {"seconds"} == expected.keys()
        assert set(actual["flags"]) == expected_flags
        for key in expected.keys() - {"flags"}:
            assert actual[key] == expected[key], key


def _refresh():
    """Replay only the five registered benchmark stages on the exact-reference platform."""
    if not ARCHIVE_PLATFORM:
        raise SystemExit("Refresh this reference on Windows with Python 3.10.18.")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    record = dict(commit=commit, runtime=dict(system=platform.system(), python=platform.python_version(),
                                              numpy=version("numpy"), scipy=version("scipy"), blas_threads=1),
                  results={})
    for archive, stage, index in CASES:
        result = _run(archive, stage, index)
        del result["seconds"]
        case = f"{archive}/{stage}/{index}"
        record["results"][case] = result
        print(case, result["compared_by"], result["probability"], flush=True)
    REFERENCE.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__" and sys.argv[1:] == ["--refresh"]:
    _refresh()
elif __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    from tools.benchmarks import posterior_recovery as benchmark

    stage, index = sys.argv[1:]
    summary = json.load(sys.stdin)
    folder = ROOT / "src/gimap/features/fitting/domain/posterior/data"
    benchmark._init(str(folder / summary["network"]), summary["settings"])
    assert benchmark.NETWORK.meta["sha256"] == summary["network_sha256"]
    benchmark.OPTIONS["stages"] = [stage]
    print(json.dumps(benchmark.run(int(index))[stage]))
