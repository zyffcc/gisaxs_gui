"""Synthetic shared-q tables keep independent curves, units and column provenance."""

from pathlib import Path

import numpy as np
import pytest

from src.gimap.features.fitting.application.models import InSituSourceFrame, LoadCurveRequest
from src.gimap.features.fitting.application.use_cases import LoadCurve
from src.gimap.features.fitting.infrastructure.adapters.local_files import LocalCurveRepository
from src.gimap.features.fitting.infrastructure.adapters.scattering_curve_files import load_xy_any


def _write(tmp_path, text, name="curves.dat"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _sources(path, layout="shared_q"):
    from src.gimap.features.fitting.application.curve_sources import sources_from_table
    use_case = LoadCurve(LocalCurveRepository())
    preview = use_case.inspect_table(path)
    return preview, sources_from_table(preview, layout)


def _load(source, unit="angstrom"):
    result = LoadCurve(LocalCurveRepository()).execute(LoadCurveRequest(Path(source.token), unit))
    assert result.succeeded, result.error
    return result.value


def test_invalid_sigma_never_shifts_later_errors_onto_earlier_rows(tmp_path):
    path = _write(tmp_path, "0.1 11 1\n0.2 22 broken\n0.3 33 3\n0.4 44 4\n")
    curve = load_xy_any(str(path))
    np.testing.assert_array_equal(curve.q, [0.1, 0.3, 0.4])
    np.testing.assert_array_equal(curve.I, [11, 33, 44])
    np.testing.assert_array_equal(curve.err, [1, 3, 4])


@pytest.mark.parametrize("separator", [" ", "\t", ",", ";"])
def test_shared_q_curves_preserve_values_and_header_unit(tmp_path, separator):
    rows = [("q (1/nm)", "I1", "I2"), ("0.1", "11", "21"), ("0.2", "12", "22")]
    path = _write(tmp_path, "\n".join(separator.join(row) for row in rows))
    preview, sources = _sources(path)
    assert preview.suggested_layout == "shared_q"
    assert preview.unit == "nm" and preview.allow_shared_q
    assert len(sources) == 2 and sources[0].token != sources[1].token
    for source, intensity in zip(sources, ([11, 12], [21, 22])):
        curve = _load(source)
        np.testing.assert_array_equal(curve.q, [0.1, 0.2])
        np.testing.assert_array_equal(curve.intensity, intensity)
        assert curve.error is None
        assert curve.q_source_unit == "nm" and curve.q_unit_from == "stated in the file"
        assert curve.source_path == source.token
        assert curve.observation["curve_source"]["columns"] == source.columns.to_dict()
        assert curve.observation["curve_source"]["sigma_source"] == "none"


def test_each_intensity_column_has_its_own_row_mask(tmp_path):
    path = _write(tmp_path, "q,I1,I2\n0.1,,21\n0.2,12,22\n0.3,nan,23\n0.4,14,\n0.5,15,25\n")
    _, sources = _sources(path)
    first, second = map(_load, sources)
    np.testing.assert_array_equal(first.q, [0.2, 0.4, 0.5])
    np.testing.assert_array_equal(first.intensity, [12, 14, 15])
    np.testing.assert_array_equal(second.q, [0.1, 0.2, 0.3, 0.5])
    np.testing.assert_array_equal(second.intensity, [21, 22, 23, 25])


def test_explicit_sigma_mapping_uses_one_complete_row_mask(tmp_path):
    from src.gimap.features.fitting.application.curve_sources import ColumnMapping
    path = _write(tmp_path, "q I1 sigma1 I2\n0.1 11 1 21\n0.2 12 bad 22\n0.3 13 3 23\n")
    source = InSituSourceFrame(path, columns=ColumnMapping(0, 1, 2))
    curve = _load(source)
    np.testing.assert_array_equal(curve.q, [0.1, 0.3])
    np.testing.assert_array_equal(curve.intensity, [11, 13])
    np.testing.assert_array_equal(curve.error, [1, 3])


@pytest.mark.parametrize("header", ["", "# q I sigma\n", "q intensity error\n"])
def test_single_layout_preserves_legacy_arrays_exactly(tmp_path, header):
    path = _write(tmp_path, header + "0.1 11 1\n0.2 22 2\n0.3 33 3\n")
    expected = load_xy_any(str(path))
    preview, sources = _sources(path, "single")
    assert preview.suggested_layout == ("single" if header else "ambiguous")
    curve = _load(sources[0])
    np.testing.assert_array_equal(curve.q, expected.q)
    np.testing.assert_array_equal(curve.intensity, expected.I)
    np.testing.assert_array_equal(curve.error, expected.err)


@pytest.mark.parametrize("header", [
    "# GIMaP Analyze fit input (q in 1/A)\n# columns: qy (1/A)  I  sigma  pixels\n",
    "q I sigma pixels\n",
])
def test_analyze_tables_cannot_be_split_into_intensity_columns(tmp_path, header):
    from src.gimap.features.fitting.application.curve_sources import sources_from_table
    path = _write(tmp_path, header + "0.1 11 1 5\n0.2 22 2 6\n")
    preview, sources = _sources(path, "single")
    assert preview.analyze_export and preview.suggested_layout == "single"
    assert not preview.allow_shared_q
    with pytest.raises(ValueError, match="Analyze"):
        sources_from_table(preview, "shared_q")
    curve = _load(sources[0])
    np.testing.assert_array_equal(curve.error, [1, 2])
    if header.startswith("# GIMaP"):
        np.testing.assert_array_equal(curve.pixels, [5, 6])


@pytest.mark.parametrize("changed_header", ["q I2 I1", "q I1", ""])
def test_column_changes_fail_instead_of_silently_reading_a_different_curve(tmp_path, changed_header):
    path = _write(tmp_path, "q I1 I2\n0.1 11 21\n0.2 12 22\n")
    _, sources = _sources(path)
    path.write_text(changed_header + "\n0.1 11 21\n0.2 12 22\n", encoding="utf-8")
    result = LoadCurve(LocalCurveRepository()).execute(LoadCurveRequest(Path(sources[1].token)))
    assert not result.succeeded
    assert result.error.code == "invalid_data"
    assert "column" in result.error.message.lower()


def test_tokens_round_trip_unicode_special_paths_and_column_metadata(tmp_path):
    from src.gimap.features.fitting.application.curve_sources import ColumnMapping
    mapping = ColumnMapping(0, 2, None, ("q", "I 一", "I / 二"), 3)
    source = InSituSourceFrame(tmp_path / "测量 #% sample.dat", columns=mapping)
    assert InSituSourceFrame.from_token(source.token) == source
    assert InSituSourceFrame.from_token(str(source.path)).columns is None
    old = InSituSourceFrame(source.path, frame_index=4)
    assert InSituSourceFrame.from_token(old.token) == old
    assert "I / 二" in source.display_name
    assert ColumnMapping.from_dict(mapping.to_dict()) == mapping


def test_preview_contains_ten_rows_without_changing_numeric_values(tmp_path):
    path = _write(tmp_path, "\n".join(f"{i / 100} {i + 10}" for i in range(1, 15)))
    preview, sources = _sources(path, "single")
    assert len(preview.sample_rows) == 10
    assert preview.column_count == 2 and not preview.has_header
    assert preview.unit is None and preview.suggested_layout == "single"
    assert not preview.allow_shared_q
    assert _load(sources[0], "nm").q_source_unit == "nm"


def test_preview_stops_after_header_and_ten_rows(tmp_path, monkeypatch):
    path = _write(tmp_path, "q I1 I2\n" + "\n".join(f"{i} {i + 10} {i + 20}" for i in range(100)))
    original_open, counts = Path.open, []

    class PrefixOnly:
        def __init__(self, stream):
            self.stream, self.count = stream, 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            counts.append(self.count)
            self.stream.close()

        def __iter__(self):
            return self

        def __next__(self):
            self.count += 1
            assert self.count <= 11, "Preview read beyond its ten-row prefix"
            return next(self.stream)

        def read(self, *_args):
            pytest.fail("Preview must not read the complete file")

    def guarded_open(target, *args, **kwargs):
        stream = original_open(target, *args, **kwargs)
        return PrefixOnly(stream) if target == path else stream

    monkeypatch.setattr(Path, "open", guarded_open)
    preview = LoadCurve(LocalCurveRepository()).inspect_table(path)
    assert len(preview.sample_rows) == 10 and max(counts) == 11


def test_source_record_and_stem_keep_physical_path_and_column_identity(tmp_path):
    from src.gimap.features.fitting.application.curve_sources import source_record, source_stem
    path = _write(tmp_path, "q,I one,I/two\n0.1,11,21\n0.2,12,22\n", name="测量 # table.dat")
    _, sources = _sources(path)
    record = source_record(sources[1].token)
    assert record == {"path": str(path), "frame_index": 0, "source_kind": "curve",
                      "columns": sources[1].columns.to_dict()}
    assert source_stem(sources[1].token) == "测量 # table_I3"
    assert source_stem(str(path)) == path.stem


def test_missing_whitespace_cell_cannot_shift_a_different_intensity_into_its_place(tmp_path):
    path = _write(tmp_path, "q I1 I2\n0.1 11 21\n0.2 22\n0.3 13 23\n")
    _, sources = _sources(path)
    first = _load(sources[0])
    np.testing.assert_array_equal(first.q, [0.1, 0.3])
    np.testing.assert_array_equal(first.intensity, [11, 13])


def test_empty_sigma_row_and_nonfinite_row_do_not_shift_sigma(tmp_path):
    path = _write(tmp_path, "0.1,11,1\n0.2,22,\n0.3,33,3\n0.4,44,inf\n0.5,55,5\n")
    curve = load_xy_any(str(path))
    np.testing.assert_array_equal(curve.q, [0.1, 0.3, 0.5])
    np.testing.assert_array_equal(curve.I, [11, 33, 55])
    np.testing.assert_array_equal(curve.err, [1, 3, 5])


def test_column_width_drift_after_preview_is_rejected_at_load(tmp_path):
    path = _write(tmp_path, "\n".join(f"{i} {i + 10} {i + 20}" for i in range(1, 11)))
    _, sources = _sources(path)
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\n0.9 99 88 77\n")
    result = LoadCurve(LocalCurveRepository()).execute(LoadCurveRequest(Path(sources[0].token)))
    assert not result.succeeded and "column count changed" in result.error.message


@pytest.mark.parametrize("payload", ["broken!", "e30=", "bnVsbA=="])
def test_invalid_source_tokens_fail_before_touching_a_file(tmp_path, payload):
    token = str(tmp_path / "missing.dat") + "::gimap-curve=" + payload
    result = LoadCurve(LocalCurveRepository()).execute(LoadCurveRequest(Path(token)))
    assert not result.succeeded and result.error.code == "invalid_data"


def test_added_header_is_rejected_for_a_saved_headerless_selection(tmp_path):
    path = _write(tmp_path, "0.1 11 21\n0.2 12 22\n")
    _, sources = _sources(path)
    path.write_text("q I1 I2\n0.1 11 21\n0.2 12 22\n", encoding="utf-8")
    result = LoadCurve(LocalCurveRepository()).execute(LoadCurveRequest(Path(sources[0].token)))
    assert not result.succeeded and "header changed" in result.error.message


def test_conflicting_units_are_not_resolved_by_the_layout_choice(tmp_path):
    path = _write(tmp_path, "# q in 1/A\nq (1/nm) I1 I2\n0.1 11 21\n0.2 12 22\n")
    with pytest.raises(ValueError, match="both"):
        LoadCurve(LocalCurveRepository()).inspect_table(path)


def test_column_comment_is_not_replaced_by_later_instrument_comments(tmp_path):
    path = _write(tmp_path, "# q I1 I2\n# instrument: synthetic\n0.1 11 21\n0.2 12 22\n")
    preview, sources = _sources(path)
    assert preview.column_names == ("q", "I1", "I2")
    assert preview.has_header and preview.suggested_layout == "shared_q"
    np.testing.assert_array_equal(_load(sources[1]).intensity, [21, 22])


@pytest.mark.parametrize("header", ["q I sigma", "q I1 sigma1 I2 sigma2", "q I1 err_1 I2 err_2"])
def test_named_uncertainty_columns_cannot_be_imported_as_intensities(tmp_path, header):
    from src.gimap.features.fitting.application.curve_sources import sources_from_table, ColumnMapping
    count = len(header.split())
    path = _write(tmp_path, header + "\n" + " ".join(str(i) for i in range(count)) +
                  "\n" + " ".join(str(i + 1) for i in range(count)))
    preview, _ = _sources(path, "single")
    assert preview.uncertainty_columns and not preview.allow_shared_q
    with pytest.raises(ValueError, match="uncertainty"):
        sources_from_table(preview, "shared_q")
    selection = InSituSourceFrame(path, columns=ColumnMapping(0, 2))
    result = LoadCurve(LocalCurveRepository()).execute(LoadCurveRequest(Path(selection.token)))
    assert not result.succeeded and "uncertainty" in result.error.message


@pytest.mark.parametrize("header", ["# q (1/nm) I1 I2", "# q (1/nm) I sigma"])
def test_q_unit_in_column_comment_survives_later_metadata(tmp_path, header):
    path = _write(tmp_path, header + "\n# instrument: synthetic\n# sample: A-1\n0.1 11 21\n0.2 12 22\n")
    use_case = LoadCurve(LocalCurveRepository())
    assert use_case.inspect_table(path).unit == "nm"
    legacy = use_case.execute(LoadCurveRequest(path))
    assert legacy.succeeded and legacy.value.q_source_unit == "nm"
    layout = "shared_q" if "I1" in header else "single"
    _, sources = _sources(path, layout)
    assert _load(sources[0]).q_source_unit == "nm"


def test_metadata_with_another_quantity_unit_is_not_a_q_unit_header(tmp_path):
    path = _write(tmp_path, "# quality (1/nm): synthetic\n# sample: A-1\n0.1 11 21\n0.2 12 22\n")
    preview, sources = _sources(path)
    assert preview.unit is None
    assert _load(sources[0]).q_source_unit == "angstrom"


def test_a_first_row_with_no_intensity_does_not_replace_the_q_header(tmp_path):
    path = _write(tmp_path, "q (1/nm),I1,I2\n0.05,,\n0.1,11,21\n0.2,12,22\n")
    preview, sources = _sources(path)
    assert preview.column_names == ("q (1/nm)", "I1", "I2")
    assert preview.unit == "nm" and preview.suggested_layout == "shared_q"
    for source in sources:
        curve = _load(source)
        np.testing.assert_array_equal(curve.q, [0.1, 0.2])
        assert curve.q_source_unit == "nm"


@pytest.mark.parametrize("header", ["q I1 I2 sigma", "q I1 I2"])
def test_single_layout_cannot_use_a_named_intensity_as_sigma(tmp_path, header):
    from src.gimap.features.fitting.application.curve_sources import ColumnMapping, sources_from_table
    suffix = " 1" if "sigma" in header else ""
    path = _write(tmp_path, header + f"\n0.1 11 21{suffix}\n0.2 12 22{suffix}\n")
    preview = LoadCurve(LocalCurveRepository()).inspect_table(path)
    message = "uncertainty column is elsewhere" if "sigma" in header else "intensity column cannot be used as sigma"
    with pytest.raises(ValueError, match=message):
        sources_from_table(preview, "single")
    saved = InSituSourceFrame(path, columns=ColumnMapping(0, 1, 2))
    outcome = LoadCurve(LocalCurveRepository()).execute(LoadCurveRequest(Path(saved.token)))
    assert not outcome.succeeded and message in outcome.error.message
    if "sigma" in header:
        explicit = _load(InSituSourceFrame(path, columns=ColumnMapping(0, 1, 3)))
        np.testing.assert_array_equal(explicit.error, [1, 1])


def test_interleaved_sigma_table_can_still_select_its_first_legacy_curve(tmp_path):
    path = _write(tmp_path, "q I1 sigma1 I2 sigma2\n0.1 11 1 21 2\n0.2 12 3 22 4\n")
    _, sources = _sources(path, "single")
    curve = _load(sources[0])
    np.testing.assert_array_equal(curve.intensity, [11, 12])
    np.testing.assert_array_equal(curve.error, [1, 3])


@pytest.mark.parametrize("error_name", ["I_err", "I_error", "I_sigma"])
def test_intensity_error_aliases_remain_sigma_columns(tmp_path, error_name):
    path = _write(tmp_path, f"q I {error_name}\n0.1 11 1\n0.2 12 2\n")
    preview, sources = _sources(path, "single")
    assert preview.uncertainty_columns == (2,) and preview.intensity_columns == (1,)
    np.testing.assert_array_equal(_load(sources[0]).error, [1, 2])
