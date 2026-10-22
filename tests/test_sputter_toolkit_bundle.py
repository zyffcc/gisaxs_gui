"""The bundled Sputter toolkit (``utils/sputter_toolkit``): byte-exact, free of private names, and its fixes.

The Sputter workspace runs this research code as a subprocess, like ``utils/ML_Fitting_1D_GISAXS``; the
program never imports it. Its two network checkpoints are bound to the exact bytes of five of its source
files, and ``MANIFEST.json`` holds the SHA-256 of every file, so git must hand out the files unchanged
(``.gitattributes``: ``-text``). ``SOURCE.json`` says where the code came from and which local patches were
made here.

The parts below that run toolkit code need only numpy, scipy and fabio (all in requirements.txt); the
check of the network policies needs PyTorch and is skipped without it.
"""

from __future__ import annotations

import ast
import base64
import csv
import hashlib
import importlib
import json
import math
import random
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
KIT = "utils/sputter_toolkit"
KIT_DIR = ROOT / KIT
NOT_IN_MANIFEST = {"MANIFEST.json", "verify.py"}
BOUND_SOURCES = {  # recorded in both checkpoints; NRExpert refuses to load when one of them changes
    "sputtergisaxs/neural_refine_posterior.py": "7de7a1add7e1af67cd7dc20509f9a73e8b069a804973b02e8fb233c658e8c680",
    "sputtergisaxs/neural_refine_training.py": "52a442b5c9f449c2673f543677db1058517927e899359c186e122058fc5ab6ea",
    "sputtergisaxs/neural_refine_chart.py": "60cce3f97da529f19e4e3ad1f60150d152e7834cfd1b308e3ed0d3b7e1bd1176",
    "sputtergisaxs/v2_posterior.py": "2d0c4bf4e9749f09f55251d9feaae101ab98776208828ee3af8b696c699615a6",
    "sputtergisaxs/neural_refine_input.py": "f3ae63d165d16fdbc7c41207b0a3d432c2764fdd10bb46ba9caac33f13af5806",
}
POLICY_KEYS = {"module", "horizontal", "vertical", "direct_features", "normalization", "highq_only",
               "common_signal_indices", "D_window_ratio"}
PRIVATE_TOKENS = ROOT / "docs" / "agents" / "eval" / "local" / "private_tokens.txt"
"""The names of the private datasets, people and accounts the toolkit was developed on, and numbers of their
instrument geometry: one per line (``#`` starts a comment), in the git-ignored local folder, so that no committed
file repeats them (AGENTS.md). On a clone the file is absent and only the generic checks run."""
ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]{1,2}[A-Za-z_]|(?<![\w.])/(?:home|data|Users|scratch)/")
TEXT_SUFFIXES = {".py", ".json", ".md", ".txt", ".csv"}

needs_git = pytest.mark.skipif(
    not shutil.which("git") or not (ROOT / ".git").exists(),
    reason="not a git checkout (or git is not installed)",
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest() -> dict[str, str]:
    return json.loads((KIT_DIR / "MANIFEST.json").read_text(encoding="utf-8"))["files"]


def _git(*arguments: str, data: bytes | None = None) -> bytes:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, input=data, capture_output=True, check=True, timeout=120
    ).stdout


def _stored_by_git(folder: str) -> dict[str, bytes]:
    """The bytes git holds (in its index) for each tracked file of ``folder``: what a clone gets."""
    names = {}
    for record in _git("ls-files", "--stage", "-z", "--", folder).split(b"\0"):
        if record:
            entry, path = record.split(b"\t", 1)
            names[path.decode("utf-8")] = entry.split()[1]
    if not names:
        return {}
    output = _git("cat-file", "--batch", data=b"\n".join(names.values()) + b"\n")
    stored, position = {}, 0
    for path in names:
        end = output.index(b"\n", position)
        size = int(output[position:end].split()[2])
        stored[path] = output[end + 1:end + 1 + size]
        position = end + 1 + size + 1
    return stored


def _kit_files() -> list[Path]:
    return sorted(path for path in KIT_DIR.rglob("*")
                  if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"})


@pytest.fixture(scope="module")
def kit():
    """Import ``sputtergisaxs`` from the bundle; forget it again afterwards."""
    sys.path.insert(0, str(KIT_DIR))
    try:
        yield importlib.import_module("sputtergisaxs")
    finally:
        sys.path.remove(str(KIT_DIR))
        for name in [name for name in sys.modules if name == "sputtergisaxs" or name.startswith("sputtergisaxs.")]:
            del sys.modules[name]


# ------------------------------------------------------------------------------------- the bytes --
@needs_git
def test_git_does_not_convert_line_endings_in_the_toolkit():
    for name in (f"{KIT}/sputtergisaxs/neural_refine_input.py", f"{KIT}/experts/TS/policy.json", f"{KIT}/MANIFEST.json"):
        answer = _git("check-attr", "text", "--", name).decode("utf-8").strip()
        assert answer.endswith("text: unset"), answer
    ignored = subprocess.run(["git", "check-ignore", "-q", f"{KIT}/experts/TS/model/best.pt"], cwd=ROOT, timeout=60)
    assert ignored.returncode == 1, "git ignores the bundled Sputter toolkit; it must be tracked"


@needs_git
def test_git_stores_the_toolkit_files_the_manifest_was_made_from():
    stored = _stored_by_git(KIT)
    assert stored, "git tracks no file of the bundled Sputter toolkit"
    manifest = {f"{KIT}/{name}": digest for name, digest in _manifest().items()}
    missing = [path for path in manifest if path not in stored]
    assert not missing, "git does not hold these toolkit files (commit them as they are on disk): " + ", ".join(missing)
    different = [path for path, digest in manifest.items() if _sha(stored[path]) != digest]
    assert not different, "git stores these toolkit files with other bytes than MANIFEST.json: " + ", ".join(different)
    extra = sorted(set(stored) - set(manifest) - {f"{KIT}/{name}" for name in NOT_IN_MANIFEST})
    assert not extra, "tracked toolkit files missing from MANIFEST.json (run python verify.py --write): " + ", ".join(extra)


def test_the_toolkit_matches_its_manifest():
    manifest = _manifest()
    on_disk = {path.relative_to(KIT_DIR).as_posix(): path for path in _kit_files()}
    absent = [name for name in manifest if name not in on_disk]
    assert not absent, "toolkit files missing on disk: " + ", ".join(absent)
    changed = [name for name, digest in manifest.items() if _sha(on_disk[name].read_bytes()) != digest]
    assert not changed, "toolkit files differ from MANIFEST.json: " + ", ".join(changed)
    unlisted = sorted(set(on_disk) - set(manifest) - NOT_IN_MANIFEST)
    assert not unlisted, "toolkit files not in MANIFEST.json: " + ", ".join(unlisted)


def test_the_toolkit_text_files_end_their_lines_with_lf():
    crlf = [path.relative_to(KIT_DIR).as_posix() for path in _kit_files()
            if path.suffix in TEXT_SUFFIXES and b"\r" in path.read_bytes()]
    assert not crlf, "toolkit text files with CR line endings (verify.py --write writes LF): " + ", ".join(crlf)


def test_the_network_bound_sources_are_unchanged():
    """Both checkpoints record these SHA-256 values; the toolkit refuses to run with other bytes."""
    for name, digest in BOUND_SOURCES.items():
        assert _sha((KIT_DIR / name).read_bytes()) == digest, name


def test_the_source_record_names_its_origin_and_patches():
    source = json.loads((KIT_DIR / "SOURCE.json").read_text(encoding="utf-8"))
    assert source["source_commit"] and source["local_patches"]
    for patch in source["local_patches"]:
        assert patch["files"] and patch["change"] and patch["reason"]
        assert all((KIT_DIR / name).is_file() for name in patch["files"]), patch["files"]


# ------------------------------------------------------------------------------------ privacy --
def _private_tokens() -> set[str]:
    if not PRIVATE_TOKENS.is_file():
        return set()
    lines = PRIVATE_TOKENS.read_text(encoding="utf-8-sig").splitlines()
    return {line.split("#", 1)[0].strip().lower() for line in lines} - {""}


def test_no_absolute_paths_in_the_toolkit():
    found = []
    for path in _kit_files():
        if path.suffix in TEXT_SUFFIXES:
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if ABSOLUTE_PATH.search(line):
                    found.append(f"{path.relative_to(KIT_DIR).as_posix()}:{number}")
    assert not found, "absolute paths in the toolkit:\n" + "\n".join(found)


def test_no_private_names_in_the_toolkit_or_its_committed_companions():
    private = _private_tokens()
    if not private:
        pytest.skip("no local list of private names (docs/agents/eval/local/private_tokens.txt; absent on a clone)")
    companions = [ROOT / name for name in ("tests/test_sputter_toolkit_bundle.py", "requirements-sputter.txt",
                                           ".gitattributes", "AGENTS.md")]
    found = []
    for path in [p for p in _kit_files() if p.suffix in TEXT_SUFFIXES] + [p for p in companions if p.is_file()]:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if private & set(re.findall(r"[a-z0-9]+", line.lower())):
                found.append(f"{path.relative_to(ROOT).as_posix()}:{number}")  # the line, never the name
    assert not found, "a private dataset, person or geometry value is named here:\n" + "\n".join(found)


def test_the_policies_keep_only_the_input_transform():
    for module in ("polymer", "TS"):
        policy = json.loads((KIT_DIR / "experts" / module / "policy.json").read_text(encoding="utf-8"))
        assert set(policy) == POLICY_KEYS, module
        assert policy["D_window_ratio"] is None


def test_the_late_stages_are_routed_alike():
    """The physical path assigns the TS network exactly where the per-window rule would."""
    def late(name: str) -> set[str]:
        tree = ast.parse((KIT_DIR / "sputtergisaxs" / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "LATE" for t in node.targets):
                return set(ast.literal_eval(node.value))
        raise AssertionError(f"no LATE in {name}")

    assert late("polymer_review_infer.py") == late("sputter_sequence.py")
    assert "from .sputter_sequence import LATE" in (KIT_DIR / "sputtergisaxs" / "sputter_physical.py").read_text(encoding="utf-8")


# ----------------------------------------------------------------------- damaged frames --
def _byte_offset(image: np.ndarray) -> bytes:
    """CBF byte_offset encoding (1, 3 or 7 bytes per difference)."""
    flat = image.ravel().astype(np.int64)
    delta = np.diff(np.r_[0, flat])
    size = np.where(np.abs(delta) <= 127, 1, np.where(np.abs(delta) <= 32767, 3, 7))
    offset = np.r_[0, np.cumsum(size)[:-1]]
    out = np.zeros(int(size.sum()), np.uint8)
    one, three, seven = size == 1, size == 3, size == 7
    out[offset[one]] = delta[one].astype(np.int8).view(np.uint8)
    out[offset[three]] = 0x80
    pair = delta[three].astype("<i2").view(np.uint8).reshape(-1, 2)
    out[offset[three] + 1], out[offset[three] + 2] = pair[:, 0], pair[:, 1]
    out[offset[seven]] = 0x80
    out[offset[seven] + 1] = 0x00
    out[offset[seven] + 2] = 0x80
    quad = delta[seven].astype("<i4").view(np.uint8).reshape(-1, 4)
    for j in range(4):
        out[offset[seven] + 3 + j] = quad[:, j]
    return out.tobytes()


def _write_cbf(path: Path, image: np.ndarray, *, stamp: str, rows_in_header: int | None = None,
               note_lines: int = 0) -> None:
    payload = _byte_offset(image)
    notes = "".join(f"# Invented_note {index:05d} " + "x" * 64 + "\r\n" for index in range(note_lines))
    rows, cols = image.shape
    rows_in_header = rows_in_header or rows
    md5 = base64.b64encode(hashlib.md5(payload).digest()).decode()
    head = ("###CBF: VERSION 1.5, invented test frame\r\n\r\ndata_invented\r\n\r\n"
            '_array_data.header_convention "PILATUS_1.2"\r\n_array_data.header_contents\r\n;\r\n'
            f"# Detector: PILATUS 2M, invented\r\n# {stamp}\r\n{notes}# Pixel_size 172e-6 m x 172e-6 m\r\n"
            "# Exposure_time 0.0300000 s\r\n# Exposure_period 0.0700000 s\r\n# Count_cutoff 1048574 counts\r\n;\r\n\r\n"
            "_array_data.data\r\n;\r\n--CIF-BINARY-FORMAT-SECTION--\r\nContent-Type: application/octet-stream;\r\n"
            '     conversions="x-CBF_BYTE_OFFSET"\r\nContent-Transfer-Encoding: BINARY\r\n'
            f'X-Binary-Size: {len(payload)}\r\nX-Binary-ID: 1\r\nX-Binary-Element-Type: "signed 32-bit integer"\r\n'
            f"X-Binary-Element-Byte-Order: LITTLE_ENDIAN\r\nContent-MD5: {md5}\r\n"
            f"X-Binary-Number-of-Elements: {cols * rows_in_header}\r\nX-Binary-Size-Fastest-Dimension: {cols}\r\n"
            f"X-Binary-Size-Second-Dimension: {rows_in_header}\r\nX-Binary-Size-Padding: 4095\r\n\r\n")
    path.write_bytes(head.encode("ascii") + b"\x0c\x1a\x04\xd5" + payload + b"\x00" * 4095
                     + b"\r\n--CIF-BINARY-FORMAT-SECTION----\r\n;\r\n\r\n")


def _stamp(index: int) -> str:
    return (datetime(2025, 1, 1, 12) + timedelta(seconds=0.07 * index)).isoformat(timespec="milliseconds")


def test_a_damaged_cbf_raises_a_typed_error_and_an_intact_one_decodes_exactly(kit, tmp_path):
    cbfio = importlib.import_module("sputtergisaxs.cbfio")
    image = np.random.default_rng(7).poisson(3.0, (40, 33)).astype(np.int64)
    image[5, 7], image[9, 2], image[0, :4] = 40000, 900000, -1
    good = tmp_path / "invented_00001.cbf"
    _write_cbf(good, image, stamp=_stamp(0))
    read, meta = cbfio.read_cbf(good, count_limit=float("inf"))
    fallback, _ = cbfio._decode_byte_offset(good.read_bytes())
    assert np.array_equal(np.nan_to_num(read, nan=-1), image) and np.array_equal(fallback, image)
    assert meta["exposure_s"] == pytest.approx(0.03)

    raw = good.read_bytes()
    cut = tmp_path / "invented_00002.cbf"
    cut.write_bytes(raw[:len(raw) - 4095 - 600])
    rows = tmp_path / "invented_00003.cbf"
    _write_cbf(rows, image, stamp=_stamp(0), rows_in_header=41)
    for path in (cut, rows):
        with pytest.raises(cbfio.CBFDamagedError) as caught:
            cbfio.read_cbf(path)
        assert caught.value.reason.startswith("CBF payload")
    with pytest.raises(cbfio.CBFDamagedError):
        cbfio._decode_byte_offset(cut.read_bytes())

    # An intact frame with a header longer than the first read block is not mistaken for a damaged one.
    long = tmp_path / "invented_00004.cbf"
    _write_cbf(long, image, stamp=_stamp(0), note_lines=400)
    assert long.read_bytes().find(b"\x0c\x1a\x04\xd5") > 25000
    read, _ = cbfio.read_cbf(long, count_limit=float("inf"))
    fallback, _ = cbfio._decode_byte_offset(long.read_bytes())
    assert np.array_equal(np.nan_to_num(read, nan=-1), image) and np.array_equal(fallback, image)
    assert cbfio.payload_check(long)["nx"] == 33


def test_the_intake_leaves_a_damaged_frame_out_and_says_so(kit, tmp_path, capfd):
    intake = importlib.import_module("sputtergisaxs.polymer_dev_intake")
    data = importlib.import_module("sputtergisaxs.polymer_review_data")
    rng = np.random.default_rng(11)
    scan = tmp_path / "scan"
    scan.mkdir()
    for index in range(12):
        _write_cbf(scan / f"invented_{index + 1:05d}.cbf", rng.poisson(2.0, (1679, 1475)).astype(np.int64), stamp=_stamp(index))
    damaged = scan / "invented_00005.cbf"
    damaged.write_bytes(damaged.read_bytes()[:damaged.stat().st_size // 2])

    run = intake.prepare(scan, tmp_path / "intake", energy_ev=12000.0, sdd_mm=3000.0, alpha_i_deg=0.4, metal="Ag",
                         beam_col=780.0, beam_row=1380.0, workers=1, system_note="invented")
    said = capfd.readouterr().out
    assert "INTAKE_PLAN 1 12" in said
    assert "INTAKE_EXCLUDED invented_00005.cbf CBF payload truncated" in said
    assert run["selected_frames"] == 12 and run["analysed_frames"] == 11
    assert [entry["frame"] for entry in run["excluded_frames"]] == ["invented_00005.cbf"]
    assert run["frame_denominator"].startswith("11 analysed = 12 selected - 1 damaged")
    saved = json.loads((tmp_path / "intake" / "excluded_frames.json").read_text(encoding="utf-8"))
    assert [entry["frame"] for entry in saved["excluded"]] == ["invented_00005.cbf"]
    chunk = json.loads((tmp_path / "intake" / "chunk_00000.json").read_text(encoding="utf-8"))
    assert chunk["crop"] == [441, 1411, 224, 1334]  # the support window, relative to the beam centre

    record, rows = data.intake_rows(tmp_path / "intake")
    receipt = data.aggregate(tmp_path / "intake", tmp_path / "binned", record, rows,
                             data.bin_ids([row["time_s"] for row in rows], seconds=0.5), nominal_seconds=0.5)
    windows = json.loads((tmp_path / "binned" / "chunk_00000.json").read_text(encoding="utf-8"))["rows"]
    assert receipt["raw_frames"] == 11 and sum(window["raw_frames"] for window in windows) == 11
    assert "invented_00005.cbf" not in [member for window in windows for member in window["raw_members"]]
    assert [entry["frame"] for entry in receipt["excluded_raw_frames"]] == ["invented_00005.cbf"]


# ----------------------------------------------------------------------- the physical path --
def test_the_stage_path_never_goes_back_and_its_film_onset_disagrees_least(kit):
    physical = importlib.import_module("sputtergisaxs.sputter_physical")
    names = ["pre_nucleation_candidate", "separate_islands_candidate", "coalescence_candidate",
             "coalescence_or_film_candidate", "stage_unresolved", ""]
    chance = random.Random(3)
    for _ in range(3000):
        stages = chance.choices(names, weights=[chance.random() for _ in names], k=chance.randint(1, 25))
        path = physical.stage_path(stages)
        if all(physical.STAGE_ORDINAL.get(stage) is None for stage in stages):
            assert path == [None] * len(stages)
            continue
        assert all(a <= b for a, b in zip(path, path[1:]))
        onset = next((i for i, level in enumerate(path) if physical.is_late(level)), len(stages))
        counts = physical.onset_disagreements(stages)
        assert counts[onset] == counts.min()
    # Unresolved windows take the value of the previous resolved window (of the next one at the start).
    assert physical.stage_path(["stage_unresolved", "separate_islands_candidate", "stage_unresolved",
                                "coalescence_or_film_candidate", ""]) == [1, 1, 1, 3, 3]


def test_unresolved_windows_do_not_vote_in_the_time_consistent_sequence(kit):
    sequence = importlib.import_module("sputtergisaxs.sputter_sequence")
    assert sequence.rule_module("stage_unresolved") is None and sequence.rule_module("") is None
    assert sequence.rule_module("coalescence_candidate") == "TS_layer"
    assert sequence.rule_module("separate_islands_candidate") == "polymer_embed"
    assert "unresolved_neutral" in sequence.PROTOCOL and "not re-scored" in sequence.PROTOCOL


def _modulated(period: float | None, *, seconds: float = 240.0, seed: int = 5, every_other: float = 0.0):
    """Invented intensity: a slow rise, a modulation of the given period and noise.

    ``every_other`` adds a component of twice the period: the dips then alternate in depth (two unequal dips per
    repeat), and the autocorrelation is stronger at twice the period than at the period itself.
    """
    rng = np.random.default_rng(seed)
    times = np.r_[np.arange(0.05, 5.0, 0.1), np.arange(5.25, seconds, 0.5)]
    nominal = np.where(times < 5.0, 0.1, 0.5)
    level = 2000.0 * (1.0 + times / 100.0)
    if period:
        level = level * np.exp(0.25 * np.sin(2 * np.pi * times / period)
                               + every_other * np.sin(np.pi * times / period + 0.7))
    return list(times), list(level * np.exp(rng.normal(0.0, 0.05, times.size))), list(nominal)


def test_the_rotation_period_comes_from_the_intensity_modulation(kit):
    physical = importlib.import_module("sputtergisaxs.sputter_physical")
    for period in (7.0, 22.0):
        found, record = physical.rotation_period(*_modulated(period))
        assert found == pytest.approx(period, rel=0.03), record
        assert record["acf_peak"] >= 0.3 and record["cycles"] >= 3 and record["full_repeat_s"] is None
        assert record["psd_peak_period_s"] == pytest.approx(period, rel=0.06)
    # Two unequal dips per repeat: the period is the dip spacing, as for a single dip, and the exact repeat is named.
    found, record = physical.rotation_period(*_modulated(22.0, seconds=300.0, every_other=0.16))
    assert record["acf_local_maxima"][0][1] < 0.3 < record["double_lag_acf"]  # the first maximum alone is weak
    assert found == pytest.approx(22.0, rel=0.03), record
    assert record["full_repeat_s"] == pytest.approx(44.0, rel=0.05)
    found, record = physical.rotation_period(*_modulated(None))
    assert found is None and record["reason"]
    found, record = physical.rotation_period(*_modulated(55.0, seconds=120.0))
    assert found is None  # fewer than three cycles


def test_the_trend_follows_growth_even_when_the_span_is_longer_than_the_island_phase(kit):
    physical = importlib.import_module("sputtergisaxs.sputter_physical")
    rng = np.random.default_rng(13)
    times = list(np.arange(0.25, 17.0, 0.5))
    truth = 0.5 * np.exp(0.08 * np.array(times))
    values = list(truth * np.exp(rng.normal(0.0, 0.25, len(times))))
    values[5] = values[6] = None  # windows without a value stay without a trend
    trend = physical.rolling_log_trend(times, values, span=43.0)
    assert trend[5] is None and trend[6] is None
    shown = np.array([value for value in trend if value is not None])
    assert np.all(np.diff(shown) > 0)  # it grows, not one constant
    for index in (0, len(times) - 1):  # the ends are not pulled towards the middle
        assert trend[index] == pytest.approx(truth[index], rel=0.25)
    short = [value for value in physical.rolling_log_trend(times, values, span=2.5)]
    kept = [index for index, value in enumerate(short) if value is not None]
    assert np.corrcoef(np.log([short[i] for i in kept]), np.log(truth[kept]))[0, 1] > 0.8


def test_the_island_volume_equivalent_thickness():
    hemisphere = (2 / 3) * math.pi * 3.0 ** 3 / ((math.sqrt(3) / 2) * 12.0 ** 2)
    sys.path.insert(0, str(KIT_DIR))
    try:
        physical = importlib.import_module("sputtergisaxs.sputter_physical")
        assert physical.t_eq(3.0, 12.0) == pytest.approx(hemisphere)
        assert physical.t_eq(3.0, 12.0, 0.3) == pytest.approx(hemisphere * math.exp(0.27))
        assert physical.t_eq(None, 12.0) is None
    finally:
        sys.path.remove(str(KIT_DIR))


def _invented_series(folder: Path, *, alternate: bool = True) -> None:
    """binned/frames.csv (+ frames_alt.csv) of an invented series: an empty window, pre-nucleation, islands, a stray TS
    window, then a film."""
    stages = (["pre_nucleation_candidate"] * 3 + ["separate_islands_candidate", "stage_unresolved"] * 6
              + ["coalescence_candidate"] + ["separate_islands_candidate"] * 10
              + ["coalescence_or_film_candidate", "stage_unresolved"] * 4 + ["separate_islands_candidate"]
              + ["coalescence_or_film_candidate"] * 10)
    late = {"coalescence_candidate", "coalescence_or_film_candidate", "continuous_film_candidate"}
    rows, other = [], []
    for i, stage in enumerate(stages):
        time = 5.0 + 0.5 * i
        base = dict(frame=f"bin_{i:06d}", time_s=time, stage=stage, nominal_bin_seconds=0.5, intensity_rate=1000.0 + i,
                    selected_counts=0 if i == 0 else 50000 + i,  # the first window: shutter still closed
                    feature_status="bilateral" if stage == "separate_islands_candidate" else "no_peak",
                    q_star_nm_inv=0.4 if stage == "separate_islands_candidate" else "", D_corr_nm=15.7 if stage == "separate_islands_candidate" else "")
        island = dict(module="polymer_embed", mode1_probability=0.6, mode1_R_number_mean_nm=2.0 + 0.05 * i,
                      mode1_D_nm=15.0 + 0.1 * i if base["feature_status"] == "bilateral" else "no peak",
                      mode1_radius_sigma_log=0.2, mode1_burial_base_depth_nm=0.5, mode1_polymer_thickness_nm=40.0,
                      mode1_spectral_domain_nm="n/a", mode1_metal_fraction="n/a", mode1_film_thickness_nm="n/a")
        film = dict(module="TS_layer", mode1_probability=0.7, mode1_R_number_mean_nm="n/a", mode1_D_nm="n/a",
                    mode1_radius_sigma_log="n/a", mode1_burial_base_depth_nm="n/a", mode1_polymer_thickness_nm="n/a",
                    mode1_spectral_domain_nm=60.0 + i, mode1_metal_fraction=0.5, mode1_film_thickness_nm=9.0)
        rows.append(dict(base, **(film if stage in late else island)))
        other.append(dict(base, **(island if stage in late else film)))
    binned = folder / "binned"
    binned.mkdir(parents=True)
    for name, table in (("frames.csv", rows), ("frames_alt.csv", other)) if alternate else (("frames.csv", rows),):
        with (binned / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_the_physical_table_follows_one_path_with_network_values(kit, tmp_path):
    struct = importlib.import_module("sputtergisaxs.sputter_struct")
    physical = importlib.import_module("sputtergisaxs.sputter_physical")
    _invented_series(tmp_path)
    struct.write_structure(tmp_path / "binned", tmp_path, metal="Cu")
    physical.write_physical(tmp_path, rate_nm_s=0.05, t0_s=4.0)
    per_window = _read(tmp_path / "structure.csv")
    rows = _read(tmp_path / "structure_physical.csv")
    assert list(rows[0])[:len(per_window[0])] == list(per_window[0])
    assert set(physical.NEW_COLUMNS) <= set(rows[0])

    levels = {name: i for i, name in enumerate(physical.PATH_STAGES)}
    path = [levels[row["stage_path"]] for row in rows if row["stage_path"]]
    assert len(path) == len(rows) - 1 and all(a <= b for a, b in zip(path, path[1:]))
    # A window without counts gets no stage and no values; pre-nucleation windows get no island values.
    empty = rows[0]
    assert empty["stage_path"] == "" and empty["module_path"] == "" and empty["film"] == "false"
    assert empty["R_nm"] == "" and empty["network_R_nm"] == "" and empty["R_reason"] == physical.NO_DATA_REASON
    for row in rows[1:3]:
        assert row["stage_path"] == "pre_nucleation" and row["R_nm"] == "" and row["R_reason"] == physical.PRE_REASON
        assert row["t_eq_nm"] == "" and row["R_trend_nm"] == "" and row["sigma_log"] == ""
    # Unresolved windows carry the stage of their neighbours and say so.
    assert rows[4]["stage"] == "stage_unresolved" and rows[4]["stage_path"] == "islands" and rows[4]["stage_imputed"] == "true"
    assert rows[3]["stage_imputed"] == "false" and rows[4]["R_nm"]
    assert all(row["stage_path"] == "islands" for row in rows if row["R_nm"] or row["R_trend_nm"] or row["t_eq_nm"])
    stray_ts, stray_island = rows[15], rows[34]
    assert stray_ts["stage"] == "coalescence_candidate" and stray_ts["module"] == "TS_layer"
    assert stray_ts["module_path"] == "polymer_embed" and stray_ts["path_changed"] == "true" and stray_ts["film"] == "false"
    assert float(stray_ts["R_nm"]) == pytest.approx(2.0 + 0.05 * 15)  # the island network's value, from frames_alt.csv
    assert stray_island["module_path"] == "TS_layer" and stray_island["path_changed"] == "true"
    assert float(stray_island["L_TS_nm"]) == pytest.approx(94.0) and stray_island["R_nm"] == ""
    for row in rows:
        if row["module_path"] == "polymer_embed" and row["R_nm"]:
            assert row["R_grade"] == "P" and row["R_source"] == "network"
            assert float(row["t_eq_lo_nm"] or 0) <= float(row["t_eq_nm"] or 0) <= float(row["t_eq_hi_nm"] or 0)
        if row["film"] == "true":
            assert row["film_thickness_status"].startswith("not determined by GISAXS")
            assert row["TS_film_thickness_nm_conditional"] == "" and row["t_eq_nm"] == ""
        else:
            assert row["film_thickness_status"] == ""
        assert float(row["nominal_thickness_nm"]) == pytest.approx(0.05 * max(0.0, float(row["time_s"]) - 4.0))

    summary = json.loads((tmp_path / "structure_physical_summary.json").read_text(encoding="utf-8"))
    for key in ("windows", "metal", "R", "D", "grades", "reliability", "film_onset_s", "film_onset_interval_s",
                "rotation_period_s", "rotation_detection", "reversals", "thickness", "notes"):
        assert key in summary, key
    assert summary["film_onset_s"] == pytest.approx(float(rows[26]["time_s"]))  # the first film window
    low, high = summary["film_onset_interval_s"]
    assert low <= summary["film_onset_s"] <= high and summary["film_onset_no_TS_within_slack"] is False
    assert summary["no_data_windows"] == 1 and summary["stage_imputed_windows"] == 10
    assert summary["stage_path_counts_imputed"]["islands"] == 6 and summary["stage_path_counts_imputed"]["film"] == 4
    assert summary["island_phase_s"][0] == pytest.approx(float(rows[3]["time_s"]))
    # 'after' counts the per-window values of the path, 'after_trend' the smoothed lines.
    assert summary["reversals"]["after"]["R_zigzag"] == physical.zigzags([float(r["R_nm"]) for r in rows if r["R_nm"]])
    assert set(summary["reversals"]["after_trend"]) == {"R_zigzag", "D_zigzag"}
    assert summary["reversals"]["before"]["module_revert"] >= 2 and summary["reversals"]["after"]["module_revert"] == 0
    assert summary["reversals"]["after"]["stage"] == 0
    assert set(summary["reversals"]["before"]) == {"stage", "module_revert", "R_zigzag", "D_zigzag"}
    assert summary["thickness"]["rate_nm_s"] == 0.05 and summary["thickness"]["mass_continuity"]["ratio"] > 0
    assert summary["grades"]["P"] and summary["R"]["physical_P"] > 0
    assert any("Ag model" in note and "reference only" in note for note in summary["notes"])


def test_without_the_other_network_the_physical_table_says_what_is_missing(kit, tmp_path):
    struct = importlib.import_module("sputtergisaxs.sputter_struct")
    physical = importlib.import_module("sputtergisaxs.sputter_physical")
    _invented_series(tmp_path, alternate=False)
    struct.write_structure(tmp_path / "binned", tmp_path, metal="Ag")
    physical.write_physical(tmp_path)
    rows = _read(tmp_path / "structure_physical.csv")
    assert rows[15]["R_nm"] == "" and rows[15]["R_reason"] == physical.ALT_MISSING
    assert rows[34]["L_TS_nm"] == ""
    assert all(row["nominal_thickness_nm"] == "" for row in rows)
    summary = json.loads((tmp_path / "structure_physical_summary.json").read_text(encoding="utf-8"))
    assert summary["thickness"]["mass_continuity"] is None and summary["thickness"]["rate_nm_s"] is None
    # A rate without a start time: t0 = 0 s is an assumption, and the note says so.
    physical.write_physical(tmp_path, rate_nm_s=0.05)
    thickness = json.loads((tmp_path / "structure_physical_summary.json").read_text(encoding="utf-8"))["thickness"]
    assert thickness["t0_s"] == 0.0 and thickness["t0_assumed"] is True and "t0 not given" in thickness["nominal_note"]


def test_the_onset_interval_also_weighs_no_ts_at_all(kit):
    physical = importlib.import_module("sputtergisaxs.sputter_physical")

    def onset(stages):
        modules = ["TS_layer" if physical.is_late(level) else "polymer_embed" for level in physical.stage_path(stages)]
        return physical._onset(modules, stages, [0.5 * i for i in range(len(stages))], len(stages))

    islands, coalescence = "separate_islands_candidate", "coalescence_candidate"
    first, extras = onset([islands] * 10 + [coalescence] * 3 + [islands] * 2)
    assert first == 10 and extras["film_onset_s"] == 5.0 and extras["film_onset_interval_s"] == [4.0, 7.0]
    assert extras["film_onset_no_TS_disagreeing_windows"] == 3 and extras["film_onset_no_TS_within_slack"] is True
    notes = physical._onset_notes(extras, {"film": 0})
    assert any(note.startswith("TS onset not established") for note in notes)
    assert any("never the film stage" in note for note in notes)
    # No onset on the path, yet the onsets within the slack are still given.
    first, extras = onset([islands] * 10 + [coalescence] + [islands] * 2)
    assert first is None and extras["film_onset_s"] is None and extras["film_onset_interval_s"] == [4.5, 6.0]
    assert any(note.startswith("No TS onset on the path") for note in physical._onset_notes(extras, {"film": 0}))


def test_the_physical_command_rebuilds_the_files(tmp_path):
    _invented_series(tmp_path)
    script = KIT_DIR / "scripts" / "sputter_struct_v1.py"
    table = subprocess.run([sys.executable, "-B", str(script), "table", "--binned", str(tmp_path / "binned"),
                            "--out", str(tmp_path), "--metal", "Au"], capture_output=True, text=True, timeout=300)
    assert table.returncode == 0, table.stderr
    (tmp_path / "structure_physical.csv").unlink()
    done = subprocess.run([sys.executable, "-B", str(script), "physical", "--folder", str(tmp_path), "--rate-nm-s", "0.1"],
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith("SPUTTER_PROGRESS\tdone\t1.000\t")
    summary = json.loads((tmp_path / "structure_physical_summary.json").read_text(encoding="utf-8"))
    assert summary["thickness"]["rate_nm_s"] == 0.1 and summary["metal"] == "Au"


# ------------------------------------------------------------------------- the networks --
def test_the_shipped_policies_load_with_their_checkpoints(kit):
    pytest.importorskip("torch")
    infer = importlib.import_module("sputtergisaxs.neural_refine_infer")
    for module, name in (("polymer", "polymer_embed"), ("TS", "TS_layer")):
        folder = KIT_DIR / "experts" / module
        expert = infer.NRExpert(folder / "model" / "best.pt", folder / "policy.json", device="cpu")
        assert expert.declaration["bank"]["module"] == name
        recorded = expert.declaration["bank"]["input_policy_sha256"]
        policy = json.loads((folder / "policy.json").read_text(encoding="utf-8"))
        infer.check_policy(policy, recorded)
        with pytest.raises(ValueError):
            infer.check_policy(dict(policy, common_signal_indices=policy["common_signal_indices"][1:]), recorded)
        with pytest.raises(ValueError):
            infer.check_policy(dict(policy, D_window_ratio=[0.8, 1.4]), recorded)
