"""Replay the ordinary Analyze export from the public data, including its drawn mask and dark frame."""
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.gimap.app.launcher import launch
from src.gimap.app.reproduce import reproduce_record
from src.gimap.features.analyze.application import AnalysisRequest, AnalyzeFrame, Corrections, ExportAnalysis
from src.gimap.features.analyze.domain import GisaxsCutSettings, GiwaxsSettings
from src.gimap.features.analyze.domain.mask_edits import MaskShape
from src.gimap.features.analyze.infrastructure.adapters import CsvCurveWriter, DetectorIoFrameSource
from src.gimap.shared.reproduction import compare_column, pack_column, unpack_column


PUBLIC = Path(__file__).parent / "data" / "external"


@pytest.fixture(scope="module", params=[
    ("gisaxs_galaxi", "galaxi_data.tif", "gisaxs", "horizontal"),
    ("giwaxs_xeuss_pbte", "PbTe_3010_r1_PT_1_10_0_00028.edf", "giwaxs", "azimuthal"),
])
def exported(request, tmp_path_factory):
    dataset, name, mode, key = request.param
    folder = tmp_path_factory.mktemp(mode)
    frame = folder / name
    shutil.copyfile(PUBLIC / dataset / name, frame)
    poni = next((PUBLIC / dataset).glob("*.poni"))
    shutil.copyfile(poni, folder / poni.name)
    source = DetectorIoFrameSource()
    shape = source.load(frame, 0).data.shape
    dark = folder / "dark.tif"
    Image.fromarray(np.full(shape, 3, dtype=np.uint16)).save(dark)
    circle = MaskShape("circle", ((50.5, 50.5), (59.5, 50.5)))
    analysis = AnalyzeFrame(source, None)(AnalysisRequest(
        path=frame, mode=mode, incidence_deg=0.463 if mode == "gisaxs" else 0.3,
        gisaxs=GisaxsCutSettings(horizontal_row=50.5),
        giwaxs=GiwaxsSettings(chi_q_window=(1.8, 2.0), bins=113),
        corrections=Corrections(dark_path=str(dark), dark_scale=0.75, mask_shapes=(circle,), lorentz=mode == "giwaxs"),
        with_map=False,
    ))
    assert analysis.drawn_mask[50, 50] and not analysis.drawn_mask[50, 62]
    curve = analysis.reduction.curve(key)
    assert len(curve.x) > 5
    files = ExportAnalysis(CsvCurveWriter()).curves(analysis, [key], folder / "export")
    record = next(path for path in files if path.suffix == ".json")
    return record, curve, analysis.drawn_mask, dark


def test_public_cut_and_azimuth_reproduce_bit_for_bit(exported, tmp_path, capsys, monkeypatch):
    record, curve, mask, _ = exported
    payload = json.loads(record.read_text(encoding="utf-8"))
    recipe = payload["reproduction"]
    assert recipe["mask"]["sha256"] == hashlib.sha256(mask.astype("u1").tobytes()).hexdigest()
    assert recipe["software_version"] and len(recipe["commit"]) == 40

    def refuse_mask_redraw(*args, **kwargs):
        pytest.fail("Replay must use the saved mask for every reduction, including ring orientation.")

    monkeypatch.setattr(AnalyzeFrame, "_user_mask", refuse_mask_redraw)
    assert launch(["reproduce", str(record), "--out", str(tmp_path)]) == 0
    assert "identical" in capsys.readouterr().out
    comparison = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["status"] == "identical"
    result = comparison["tables"][curve.key]
    assert all(col == {"status": "identical", "max_relative_difference": 0}
               for col in result["columns"].values())
    csv_path = tmp_path / result["file"]
    with csv_path.open(encoding="utf-8") as stream:
        columns = np.genfromtxt((line for line in stream if not line.startswith("#")), delimiter=",", names=True)
    for name, original in (("x", curve.x), ("intensity", curve.intensity), ("sigma", curve.sigma),
                           ("pixels", curve.pixels)):
        assert columns[name].astype(original.dtype).tobytes() == original.tobytes()


def test_missing_and_changed_dark_named_even_with_preserved_mtime(exported, tmp_path):
    record, _, _, dark = exported
    moved = dark.with_suffix(".backup")
    dark.rename(moved)
    try:
        with pytest.raises(ValueError, match=r"Missing referenced file: .*dark\.tif"):
            reproduce_record(record, tmp_path)
        data = bytearray(moved.read_bytes())
        data[-1] ^= 1
        dark.write_bytes(data)
        stat = moved.stat()
        os.utime(dark, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with pytest.raises(ValueError, match=r"SHA-256 mismatch for .*dark\.tif"):
            reproduce_record(record, tmp_path)
        assert not list(tmp_path.iterdir())
    finally:
        moved.replace(dark)


def test_changed_reference_produces_column_difference_and_preserves_prior_outputs(exported, tmp_path):
    record, curve, _, _ = exported
    payload = json.loads(record.read_text(encoding="utf-8"))
    columns = payload["reproduction"]["reference"][curve.key]
    reference = unpack_column(columns["intensity"]).copy()
    index = np.flatnonzero(np.isfinite(reference) & (reference != 0))[0]
    reference[index] *= 2
    columns["intensity"] = pack_column(reference)
    modified_record = tmp_path / "comparison_example.json"
    modified_record.write_text(json.dumps(payload), encoding="utf-8")
    target = tmp_path / "results"
    assert launch(["reproduce", str(modified_record), "--out", str(target)]) == 2
    comparison = target / "comparison.json"
    saved = comparison.read_bytes()
    differences = json.loads(saved)["tables"][curve.key]["columns"]
    assert differences["intensity"] == {"status": "different", "max_relative_difference": 0.5}
    assert differences["x"]["status"] == "identical"
    assert launch(["reproduce", str(record), "--out", str(target)]) == 1
    assert comparison.read_bytes() == saved


def test_relative_difference_and_lossless_comparison():
    ref = np.array([-4.0, 0.0, np.inf, np.nan])
    assert compare_column(unpack_column(pack_column(ref)), ref)["status"] == "identical"
    actual = ref.copy()
    actual[1] = -0.0
    assert compare_column(ref, actual) == {"status": "different", "max_relative_difference": 0.0}
    actual[0] = -5.0
    assert compare_column(ref, actual)["max_relative_difference"] == 0.25
    actual[1] = 2.0
    assert compare_column(ref, actual)["max_relative_difference"] == "infinity"


def test_legacy_record_reports_missing_recipe_without_starting_qt(tmp_path, capsys):
    record = tmp_path / "old_analysis.json"
    record.write_text('{"software": "GIMaP Analyze 1"}', encoding="utf-8")
    assert launch(["reproduce", str(record)]) == 1
    assert "no reproduction recipe" in capsys.readouterr().err


def test_summed_nexus_names_every_chunk_and_replays_calibration_terms(tmp_path):
    from src.gimap.features.analyze.domain.normalisation import Normalisation
    from src.gimap.integrations.state import InMemoryInstrumentProfileRepository
    from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile
    from src.gimap.shared.geometry.detector_response import DetectorResponse
    from tests.test_analyze_nexus_sources import H, W, _master, _stack

    frame = _master(tmp_path, "example", chunks=(2, 2))
    background = _stack(tmp_path / "background.h5", 3)
    flat = tmp_path / "flat.npy"
    np.save(flat, 1 + np.arange(H * W, dtype=float).reshape(H, W) / (H * W))
    geometry = DetectorGeometry(1.72e-4, 1.72e-4, 2.0, 16.5, 20.0, 1.34, 0.2)
    sigma = tmp_path / "sigma.npy"
    np.save(sigma, np.linspace(1, 3, H * W).reshape(H, W))
    response = DetectorResponse(counts_per_adu=1.5, offset_adu=0.25, error_model="sigma", sigma_path=str(sigma))
    profile = InstrumentProfile("test", geometry, None, (H, W), detector_response=response)
    request = AnalysisRequest(frame, frame_index=3, summed_frames=((frame, 1),), mode="gisaxs",
                              gisaxs=GisaxsCutSettings(horizontal_row=8.5), profile_name="test", with_map=False,
                              corrections=Corrections(
                                  bad_pixels=False, flat_field_path=str(flat), background_path=str(background),
                                  background_frame=2, background_scale=0.25, normalisation=Normalisation(transmission=0.8)))
    analysis = AnalyzeFrame(DetectorIoFrameSource(), InMemoryInstrumentProfileRepository([profile]))(request)
    files = ExportAnalysis(CsvCurveWriter()).curves(analysis, ["horizontal"], tmp_path / "export")
    path = next(file for file in files if file.suffix == ".json")
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["frame_index"] == 3 and record["summed_frames"][0]["frame_index"] == 1
    chunks = sorted(tmp_path.glob("example_data_*.h5"))
    refs = record["reproduction"]["inputs"]
    assert len(chunks) == 2 and all(refs[str(file)]["sha256"] for file in [*chunks, frame, flat, background, sigma])
    assert record["corrections"]["error_model"] == "sigma"
    # Resolve the detector profile's sigma file by its verified input path, also when the inputs move.
    moved_sigma = sigma.rename(path.parent / sigma.name)
    refs[str(sigma)]["path"] = moved_sigma.name
    path.write_text(json.dumps(record), encoding="utf-8")
    result = reproduce_record(path)
    assert result["status"] == "identical"
    chunks[0].unlink()
    with pytest.raises(ValueError, match="Missing referenced file.*" + chunks[0].name):
        reproduce_record(path, tmp_path / "incomplete")
