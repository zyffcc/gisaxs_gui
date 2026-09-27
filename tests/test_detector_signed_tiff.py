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
