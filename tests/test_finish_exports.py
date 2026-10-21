"""The batch's human-readable record uses the same units as its data tables."""

from pathlib import Path

from src.gimap.features.analyze.application.batch_export import BatchChoices
from src.gimap.features.analyze.application.batch_report import BatchRun, readme_text


def test_batch_readme_uses_plain_scientific_spellings_for_the_held_ring_cut(tmp_path):
    choices = BatchChoices(per_frame=True, fit_input=True)
    run = BatchRun(choices, tmp_path, "scan")
    run.series_cut_text = "I(χ) ring: q 0.912–0.950 Å⁻¹ for every frame"
    run.sources = [{"name": "scan", "summary": "2 module files · 3 frames · stitched"}]
    text = readme_text(choices, batch="scan", frame="frame", frames=3,
                       curves=[("azimuthal", "Ring A")], run=run)

    assert "Cuts: I(chi) ring: q 0.912-0.950 1/A for every frame." in text
    assert "azimuthal: Ring A" in text
    assert text.isascii()


def test_batch_readme_keeps_the_exact_names_of_sources_and_written_files(tmp_path):
    name = "薄膜_10°C_χ"
    choices = BatchChoices(per_frame=True)
    run = BatchRun(choices, tmp_path, name)
    warning = f"{name}_m01.nxs is still being written"
    run.sources = [{"name": name, "summary": "1 file · 3 frames", "warnings": [warning]}]
    text = readme_text(choices, batch=name, frame="frame", frames=3,
                       curves=[("radial", "环 χ — I(q)")], run=run)

    assert f"GIMaP Analyze batch export - {name}" in text
    assert f"Source {name}: 1 file * 3 frames" in text
    assert f"{name}_batch.json" in text
    assert "radial: 环 χ — I(q)" in text
    assert warning in text  # reader warnings may contain exact source filenames
    saved = Path(tmp_path) / "README.txt"
    saved.write_text(text, encoding="utf-8")
    assert not saved.read_bytes().startswith(b"\xef\xbb\xbf")
    assert name in saved.read_text(encoding="utf-8")
