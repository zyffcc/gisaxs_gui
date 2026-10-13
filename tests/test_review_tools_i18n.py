"""Chinese texts of the tool windows that name another control must name its Chinese label."""

from __future__ import annotations

import pytest

from src.gimap.app.presentation.i18n import ZH

RANGE_TOOLTIP = (
    "Sample-to-detector distances the search tries. Custom uses the bounds in "
    "Advanced configuration."
)
# (English text, the control label it refers to); both go through the zh tables.
REFERENCES = (
    ("Not detected — enter it under Advanced configuration", "Advanced configuration"),
    (RANGE_TOOLTIP, "Advanced configuration"),
    (RANGE_TOOLTIP, "Custom"),
    (
        "Calibration standard. Auto Detect compares the ring patterns of all known standards.",
        "Auto Detect",
    ),
    (
        "Detector model; sets the pixel size. Auto detected keeps the pixel size from the file "
        "metadata.",
        "Auto detected",
    ),
)


@pytest.mark.parametrize(("text", "label"), REFERENCES)
def test_a_chinese_cross_reference_names_the_translated_label(text, label) -> None:
    assert text in ZH, text
    assert label in ZH, f"the label {label!r} has no Chinese text, so the reference cannot match"
    assert f"“{ZH[label]}”" in ZH[text], (ZH[text], ZH[label])
