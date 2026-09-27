"""Empirical detector-row and neighbouring-frame checks of counting noise."""

import json
from pathlib import Path
import sys
import warnings

import fabio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.gimap.features.fitting.domain.detector_image import (
    DetectorPreprocessing,
    prepare_detector_image,
)

OUT = ROOT / "validation/cause_audit_20260921"


def band(path, region):
    raw = fabio.open(str(path))
    image = raw.data.copy()
    header = dict(raw.header)
    raw.close()
    state = prepare_detector_image(
        image, DetectorPreprocessing(mask_negative_pixels=True, invalid_margin_px=3), revision=1
    )
    r0, r1, x0, x1 = region
    return state.analysis_image[r0 : r1 + 1, x0 : x1 + 1].astype(float), header


def main():
    provenance = json.loads((OUT / "provenance.json").read_text())
    inputs = json.loads((OUT / "inputs.json").read_text())
    region = provenance["mask"]["pixel_region"]
    target = ROOT / "TestSAXSdata/jg_gisaxs_4nm_old_3ml_insitu_ds03_00001_00033.cbf"
    values, header = band(target, region)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        means = np.nanmean(values, axis=0)
    valid = np.isfinite(means)
    values = values[:, valid]
    means = means[valid]
    expected = np.r_[inputs["negative"]["y"][::-1], inputs["positive"]["y"]]
    q = np.r_[-np.array(inputs["negative"]["q"])[::-1], inputs["positive"]["q"]]
    np.testing.assert_allclose(means, expected, rtol=0, atol=1e-10)
    count = np.isfinite(values).sum(axis=0)
    pixel_stats = []
    for side, sgn in [("positive", 1), ("negative", -1)]:
        for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 5), (0, 5)]:
            sel = (q * sgn > 0) & (abs(q) >= lo) & (abs(q) < hi) & (means > 0) & (count > 1)
            pearson = np.nansum((values[:, sel] - means[sel]) ** 2 / means[sel])
            df = np.sum(count[sel] - 1)
            pixel_stats.append(
                dict(
                    side=side,
                    q_range=[lo, hi],
                    columns=int(sel.sum()),
                    pearson_over_df=float(pearson / df),
                    assumption="Equal mean within the selected 6-row band; row gradients/heterogeneity increase this statistic.",
                )
            )
    frame_stats = []
    for path in sorted(
        (ROOT / "TestSAXSdata").glob("jg_gisaxs_4nm_old_3ml_insitu_ds03_00001_*.cbf")
    ):
        if path == target:
            continue
        other, other_header = band(path, region)
        other = other[:, valid]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            other_mean = np.nanmean(other, axis=0)
        n = np.isfinite(other).sum(axis=0)
        good = np.isfinite(other_mean) & (n > 0) & (means > 0) & (other_mean > 0)
        fit_scale = good & (abs(q) > 0.2) & (abs(q) < 2)
        scale = float(np.median(means[fit_scale] / other_mean[fit_scale]))
        variance = means / count + scale**2 * other_mean / np.maximum(n, 1)
        z = (scale * other_mean - means) / np.sqrt(np.maximum(variance, 1e-30))
        frame_stats.append(
            dict(
                file=path.name,
                scale_to_0033=scale,
                points=int(good.sum()),
                scaled_logrmse=float(
                    np.sqrt(np.mean(np.log(scale * other_mean[good] / means[good]) ** 2))
                ),
                noise_normalized_difference_rms=float(np.sqrt(np.mean(z[good] ** 2))),
                fraction_abs_z_gt3=float(np.mean(abs(z[good]) > 3)),
                header={
                    k: str(v)
                    for k, v in other_header.items()
                    if any(s in k.lower() for s in ["time", "date", "exposure"])
                },
            )
        )
    result = dict(
        source=str(target),
        verified_same_observations=True,
        region=region,
        pixel_stats=pixel_stats,
        frame_comparisons=frame_stats,
        note="Other frames are time-separated in-situ observations, not certified stationary replicate exposures. Do not pool as a noise measurement unless stationarity is independently established.",
        header={
            k: str(v)
            for k, v in header.items()
            if any(s in k.lower() for s in ["time", "date", "exposure"])
        },
    )
    (OUT / "counting_checks.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
