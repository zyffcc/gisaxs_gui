"""Scientific signed TIFF decoding, independent of public dataset availability."""
import struct

import numpy as np

from src.gimap.shared.detector_io import load_detector_image


def signed_white_is_zero_tiff(path, values):
    """Minimal uncompressed signed-int32 TIFF, as used by GALAXI detectors.

    WhiteIsZero is display metadata; scientific pixel values stay unchanged.
    This fixture uses the TIFF format directly, not a second image dependency.
    """
    height, width = values.shape
    tags = [
        (256, 4, width), (257, 4, height), (258, 3, 32), (259, 3, 1),
        (262, 3, 0), (273, 4, 0), (277, 3, 1), (278, 4, height),
        (279, 4, values.size * 4), (339, 3, 2),
    ]
    data_offset = 8 + 2 + 12 * len(tags) + 4
    entries = [struct.pack("<HHII", tag, kind, 1, data_offset if tag == 273 else value)
               for tag, kind, value in tags]
    path.write_bytes(b"II" + struct.pack("<HIH", 42, 8, len(tags))
                     + b"".join(entries) + struct.pack("<I", 0)
                     + np.asarray(values, dtype="<i4").tobytes())


def test_signed_tiff_preserves_values_orientation_and_negative_observations(tmp_path):
    values = np.array([[-1, -6, 0], [16, 32768, 247736]], dtype=np.int32)
    source = tmp_path / "signed_detector.tif"
    signed_white_is_zero_tiff(source, values)

    loaded = load_detector_image(source)

    np.testing.assert_array_equal(loaded.data, values)
    assert not loaded.mask.any()  # Negative intensities are not generic TIFF bad-pixel codes.
    assert loaded.metadata["reader"] in {"pillow", "fabio"}
    assert loaded.metadata["transformations"] == []
    source.unlink()  # Scientific fallback must release its handle on Windows.


def _nexus_frame(path, values):
    import h5py

    with h5py.File(path, "w") as handle:
        handle.create_dataset("/entry/instrument/detector/data", data=values[None])
    return path


def test_float_nexus_frames_keep_their_negative_values_as_data(tmp_path):
    """A dark- or background-subtracted NeXus frame: the reader says it is float, so Analyze keeps the negatives."""
    from src.gimap.features.analyze.application.frame_preprocessing import floating
    from src.gimap.features.analyze.domain import valid_pixels

    values = np.full((16, 24), 0.5, dtype=np.float32)
    values[:4, :] = -0.25    # noise below zero after the subtraction: 96 measured pixels
    values[8:10, :] = np.nan  # the detector gap: 48 pixels without data
    loaded = load_detector_image(_nexus_frame(tmp_path / "subtracted.nxs", values))

    assert loaded.metadata["stored_dtype"] == "float32" and floating(loaded.metadata)
    np.testing.assert_array_equal(loaded.data, values)  # values and orientation as stored
    assert int(loaded.mask.sum()) == 48 and loaded.metadata["transformations"] == []
    valid = valid_pixels(loaded.data, loaded.mask, negatives_valid=floating(loaded.metadata))
    assert int(valid.sum()) == values.size - 48 and valid[:4, :].all()


def test_integer_nexus_frames_still_lose_their_negative_gap_codes(tmp_path):
    from src.gimap.features.analyze.application.frame_preprocessing import floating
    from src.gimap.features.analyze.domain import valid_pixels

    values = np.full((16, 24), 7, dtype=np.int32)
    values[8:10, :] = -1  # a counting detector's gap code
    loaded = load_detector_image(_nexus_frame(tmp_path / "counts.nxs", values))

    assert loaded.metadata["stored_dtype"] == "int32" and not floating(loaded.metadata)
    np.testing.assert_array_equal(loaded.data, values)  # the reader keeps the stored values
    valid = valid_pixels(loaded.data, loaded.mask, negatives_valid=floating(loaded.metadata))
    assert not valid[8:10, :].any() and int(valid.sum()) == values.size - 48
