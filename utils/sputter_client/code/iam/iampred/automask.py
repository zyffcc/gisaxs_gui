'Automatic pixel mask of a scan from its own frames (2026-11-15): what a user would otherwise draw by hand.'

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, label, median_filter

N_SAMPLE = 120
DEAD_MIN = 5.0
HOT_RATIO, HOT_SIGMA = 8.0, 12.0
LEVEL_SIZE, LEVEL_MIN = 41, 20.0
SHADOW_RATIO, SHADOW_MIN, SHADOW_GROW = 0.05, 40, 4
HOLDER_SIZE, HOLDER_RATIO = 15, 0.5
HORIZON_MARGIN = 8
PIXEL_MM = 0.172


def summed(files, n_sample=N_SAMPLE):
    '(sum of the sampled frames, pixels invalid in any of them, number of frames used).'
    import sys
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent))
    import iam_paths as IAMP
    sys.path.insert(0, str(IAMP.DIST))
    from sputtergisaxs.polymer_dev_intake import load_frame
    ids = np.unique(np.linspace(0, len(files) - 1, min(n_sample, len(files))).round().astype(int))
    total, invalid = None, None
    for i in ids:
        try:
            image, _ = load_frame(files[i])
        except OSError:
            continue
        bad = ~np.isfinite(image) | (image < 0)
        good = np.where(bad, 0.0, image)
        total = good if total is None else total + good
        invalid = bad if invalid is None else invalid | bad
    return total, invalid, len(ids)


def horizon_row(beam_row, sdd_mm, alpha_i_deg, pixel_mm=PIXEL_MM):
    'Detector row of the sample horizon (alpha_f = 0): the direct beam is alpha_i below it (rows grow downwards).'
    return beam_row - sdd_mm * np.tan(np.radians(alpha_i_deg)) / pixel_mm


def _level(image, keep, size, block=4):
    'Median over about size x size pixels, on blocks of block x block (mean of the usable pixels): the local level of'
    h, w = image.shape
    H, W = -(-h // block), -(-w // block)
    sums, counts = np.zeros((H * block, W * block)), np.zeros((H * block, W * block))
    sums[:h, :w], counts[:h, :w] = np.where(keep, image, 0.0), keep
    sums = sums.reshape(H, block, W, block).sum(axis=(1, 3))
    counts = counts.reshape(H, block, W, block).sum(axis=(1, 3))
    small = median_filter(np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0), size=max(3, size // block | 1),
                          mode="nearest")
    return np.repeat(np.repeat(small, block, axis=0), block, axis=1)[:h, :w]


def auto_mask(total, invalid, base=None, horizon=None):
    '(mask: True = use, report): the rules of the module docstring on a summed image.'
    shape = total.shape
    report = dict(shape=list(shape), pixels=int(np.prod(shape)))
    keep = ~invalid
    if base is not None and base.shape != shape:
        report["base"] = f"not used: shape {list(base.shape)} differs from the frames {list(shape)}"
        base = None
    if base is not None:
        keep &= base
        report["base_masked"] = int((~base).sum())
    local = median_filter(np.where(keep, total, 0.0), size=5, mode="nearest")
    dead = keep & (total <= 0) & (local >= DEAD_MIN)
    hot = keep & (total > HOT_RATIO * np.maximum(local, 1.0)) & (total - local > HOT_SIGMA * np.sqrt(local + 1.0))
    level = _level(total, keep, LEVEL_SIZE)
    dark = keep & (level >= LEVEL_MIN) & (total < SHADOW_RATIO * level)
    if horizon is not None:
        rows = np.arange(shape[0])[:, None]
        dark &= rows < (horizon - HORIZON_MARGIN)
        report["horizon_row"] = round(float(horizon), 1)
    labels, n = label(dark, structure=np.ones((3, 3), bool))
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    big = np.flatnonzero(sizes >= SHADOW_MIN)
    big = big[big > 0]
    shadow = np.isin(labels, big)
    holder = np.zeros(shape, bool)
    if shadow.any():
        near = _level(total, keep, HOLDER_SIZE, block=2)
        thin = keep & (near >= LEVEL_MIN) & (total < HOLDER_RATIO * near) & ~shadow
        if horizon is not None:
            thin &= np.arange(shape[0])[:, None] < (horizon - HORIZON_MARGIN)
        parts, m = label(thin | binary_dilation(shadow, iterations=2), structure=np.ones((3, 3), bool))
        touching = np.unique(parts[shadow])
        holder = np.isin(parts, touching[touching > 0]) & thin
        shadow = binary_dilation(shadow | holder, iterations=SHADOW_GROW) & keep
    mask = keep & ~dead & ~hot & ~shadow
    report.update(invalid=int(invalid.sum()), dead=int(dead.sum()), hot=int(hot.sum()), shadow=int(shadow.sum()),
                  shadow_regions=int(len(big)), holder=int(holder.sum()), kept=int(mask.sum()))
    return mask, report


def build(frames_dir, base_path=None, beam_row=None, sdd_mm=None, alpha_i_deg=None, n_sample=N_SAMPLE, on_sum=None):
    'Mask and report for a folder of CBF frames (and an optional base mask .npy). ``on_sum(total)``: gets the summed'
    files = sorted(Path(frames_dir).glob("*.cbf")) if Path(frames_dir).is_dir() else [Path(frames_dir)]
    if not files:
        raise ValueError(f"{frames_dir}: no CBF frames")
    total, invalid, used = summed(files, n_sample)
    if on_sum is not None and total is not None:
        on_sum(total)
    base = np.load(base_path).astype(bool) if base_path else None
    horizon = horizon_row(beam_row, sdd_mm, alpha_i_deg) if None not in (beam_row, sdd_mm, alpha_i_deg) else None
    mask, report = auto_mask(total, invalid, base, horizon)
    report.update(frames=len(files), frames_summed=used, base_file=Path(base_path).name if base_path else None,
                  rules=dict(dead_min=DEAD_MIN, hot_ratio=HOT_RATIO, hot_sigma=HOT_SIGMA, level_size=LEVEL_SIZE,
                             level_min=LEVEL_MIN, shadow_ratio=SHADOW_RATIO, shadow_min=SHADOW_MIN,
                             shadow_grow=SHADOW_GROW, holder_size=HOLDER_SIZE, holder_ratio=HOLDER_RATIO))
    return mask, report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frames", required=True)
    ap.add_argument("--mask", help="base mask (.npy, True = use)")
    ap.add_argument("--beam-row", type=float)
    ap.add_argument("--sdd-mm", type=float)
    ap.add_argument("--alpha-i", type=float)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    mask, report = build(a.frames, a.mask, a.beam_row, a.sdd_mm, a.alpha_i)
    if a.out:
        np.save(a.out, mask)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
