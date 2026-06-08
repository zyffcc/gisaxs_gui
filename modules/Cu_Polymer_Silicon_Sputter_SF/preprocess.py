import os
import numpy as np

try:
    from scipy.ndimage import binary_dilation, binary_fill_holes, median_filter, zoom
except Exception:
    binary_dilation = None  # type: ignore
    binary_fill_holes = None  # type: ignore
    median_filter = None  # type: ignore
    zoom = None  # type: ignore


def _resolve_path(path: str, module_folder: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(module_folder or "", path))


def _crop(img: np.ndarray, cfg: dict) -> np.ndarray:
    if not isinstance(cfg, dict):
        return img
    if all(k in cfg for k in ("y0", "y1", "x0", "x1")):
        y0 = int(cfg.get("y0", 0))
        y1 = int(cfg.get("y1", img.shape[0]))
        x0 = int(cfg.get("x0", 0))
        x1 = int(cfg.get("x1", img.shape[1]))
    else:
        left = int(cfg.get("left", 0))
        up = int(cfg.get("up", 0))
        down = int(cfg.get("down", 0))
        right = int(cfg.get("right", 0))
        y0 = up
        y1 = max(up, img.shape[0] - down)
        x0 = left
        x1 = max(left, img.shape[1] - right)
    y0 = max(0, min(y0, img.shape[0]))
    y1 = max(y0, min(y1, img.shape[0]))
    x0 = max(0, min(x0, img.shape[1]))
    x1 = max(x0, min(x1, img.shape[1]))
    return img[y0:y1, x0:x1]


def _resize(img: np.ndarray, cfg) -> np.ndarray:
    target = cfg.get("shape", cfg.get("size")) if isinstance(cfg, dict) else cfg
    if not isinstance(target, (list, tuple)) or len(target) != 2:
        return img
    th, tw = int(target[0]), int(target[1])
    if zoom is not None:
        zh = float(th) / float(max(img.shape[0], 1))
        zw = float(tw) / float(max(img.shape[1], 1))
        return zoom(img, [zh, zw], order=0).astype(np.float32)
    ys = np.linspace(0, img.shape[0] - 1, th).astype(np.int32)
    xs = np.linspace(0, img.shape[1] - 1, tw).astype(np.int32)
    return img[np.ix_(ys, xs)].astype(np.float32, copy=False)


def _load_mask(cfg: dict, shape: tuple[int, int], module_folder: str) -> np.ndarray:
    path = cfg.get("path") if isinstance(cfg, dict) else None
    if not isinstance(path, str) or not path:
        return np.zeros(shape, dtype=bool)
    path = _resolve_path(path, module_folder)
    if not os.path.isfile(path):
        return np.zeros(shape, dtype=bool)
    mask = np.load(path)
    mask = np.squeeze(mask)
    if mask.shape != shape:
        if zoom is not None:
            zh = float(shape[0]) / float(max(mask.shape[0], 1))
            zw = float(shape[1]) / float(max(mask.shape[1], 1))
            mask = zoom(mask.astype(np.float32), [zh, zw], order=0)
        else:
            ys = np.linspace(0, mask.shape[0] - 1, shape[0]).astype(np.int32)
            xs = np.linspace(0, mask.shape[1] - 1, shape[1]).astype(np.int32)
            mask = mask[np.ix_(ys, xs)]
    return mask != 0


def _circular_mask(shape: tuple[int, int], center: tuple[int, int], radius: int) -> np.ndarray:
    h, w = shape
    y, x = np.ogrid[:h, :w]
    cy, cx = int(center[0]), int(center[1])
    r = int(radius)
    return (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2


def _build_experiment_mask(img: np.ndarray, cfg: dict | None = None) -> np.ndarray:
    c = cfg or {}
    gap_value_threshold = float(c.get("gap_value_threshold", 0.0))
    row_gap_fraction = float(c.get("row_gap_fraction", 0.8))
    col_gap_fraction = float(c.get("col_gap_fraction", 0.8))
    bad_sigma = float(c.get("bad_sigma", 1e6))
    median_size = int(c.get("median_size", 5))
    dilate_iter = int(c.get("dilate_iter", 1))
    beamstops = c.get("beamstops", [])

    data = np.asarray(img, dtype=np.float32)
    invalid = ~np.isfinite(data) | (data <= gap_value_threshold)

    row_bad = np.mean(invalid, axis=1) > row_gap_fraction
    col_bad = np.mean(invalid, axis=0) > col_gap_fraction

    gap_mask = invalid.copy()
    gap_mask[row_bad, :] = True
    gap_mask[:, col_bad] = True

    safe_data = data.copy()
    finite_vals = np.isfinite(safe_data)
    fill_value = float(np.nanmedian(safe_data[finite_vals])) if np.any(finite_vals) else 0.0
    safe_data[~finite_vals] = fill_value

    med = median_filter(safe_data, size=median_size) if median_filter is not None else safe_data
    resid = safe_data - med
    valid_for_stats = ~gap_mask & np.isfinite(resid)

    if np.any(valid_for_stats):
        med_resid = float(np.median(resid[valid_for_stats]))
        mad = float(np.median(np.abs(resid[valid_for_stats] - med_resid)))
        robust_sigma = 1.4826 * mad if mad > 0 else float(np.std(resid[valid_for_stats]))
    else:
        robust_sigma = 0.0

    if robust_sigma > 0:
        badpixel_mask = np.abs(resid) > (bad_sigma * robust_sigma)
    else:
        badpixel_mask = np.zeros_like(data, dtype=bool)

    beamstop_mask = np.zeros_like(data, dtype=bool)
    if isinstance(beamstops, list):
        for bs in beamstops:
            if not isinstance(bs, dict):
                continue
            center = bs.get("center")
            radius = bs.get("radius")
            if not isinstance(center, (list, tuple)) or len(center) != 2 or radius is None:
                continue
            beamstop_mask |= _circular_mask(data.shape, (int(center[0]), int(center[1])), int(radius))

    mask = gap_mask | badpixel_mask | beamstop_mask
    if dilate_iter > 0 and binary_dilation is not None:
        mask = binary_dilation(mask, iterations=dilate_iter)
    if binary_fill_holes is not None:
        mask = binary_fill_holes(mask)
    return mask.astype(bool, copy=False)


def _log_normalize(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = img.astype(np.float32, copy=True)
    max_val = float(np.max(x)) if x.size else 0.0
    scale = float(np.e) / (max_val + float(eps))
    y = np.log(x * scale + float(eps))
    y = np.where(np.isfinite(y), y, -1.0).astype(np.float32, copy=False)
    return y


def _detector_cleanup(img: np.ndarray, cfg: dict | None = None) -> np.ndarray:
    c = cfg or {}
    threshold = float(c.get("saturation_threshold", 1e6))
    x = img.astype(np.float32, copy=True)
    bad = ~np.isfinite(x) | (x >= threshold)
    if not np.any(bad):
        return x
    valid = np.isfinite(x) & (x >= 0) & (~bad)
    fill_value = float(np.nanmax(x[valid])) if np.any(valid) else 0.0
    x[bad] = fill_value
    return x


def run(
    image: np.ndarray,
    preprocess_cfg: dict | None = None,
    *,
    module_folder: str = "",
    return_steps: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[dict]]:
    if image is None:
        raise ValueError("image is None")
    img = image.astype(np.float32, copy=True)
    cfg = preprocess_cfg or {}
    params = cfg.get("params", {}) if isinstance(cfg, dict) else {}
    steps = cfg.get("steps", []) if isinstance(cfg, dict) else []
    mask_bool: np.ndarray | None = None
    exp_mask: np.ndarray | None = None
    snapshots: list[dict] = []

    def record(label: str, value: np.ndarray) -> None:
        if return_steps:
            snapshots.append({"step": label, "label": label, "image": np.squeeze(value).copy()})

    for step in steps:
        name = str(step).lower()
        if name == "crop":
            img = _crop(img, params.get("crop", {}))
            record("crop", img)
        elif name == "resize":
            img = _resize(img, params.get("resize", {}))
            record("resize", img)
        elif name in ("detector_cleanup", "cleanup"):
            img = _detector_cleanup(img, params.get("detector_cleanup", {}))
            record("detector_cleanup", img)
        elif name in ("experiment_mask", "auto_mask", "gisaxs_mask"):
            exp_mask = _build_experiment_mask(img, params.get("experiment_mask", {}))
            record("experiment_mask", exp_mask.astype(np.float32))
        elif name in ("log_and_normalize", "lognormalize", "log_norm"):
            eps = float(params.get("log_and_normalize", {}).get("eps", 1e-8))
            if exp_mask is not None:
                tmp = img.astype(np.float32, copy=True)
                valid_for_fill = np.isfinite(tmp) & (~exp_mask)
                finite_vals = tmp[valid_for_fill]
                if finite_vals.size:
                    fill_value = float(np.nanmax(finite_vals))
                else:
                    finite_all = tmp[np.isfinite(tmp)]
                    fill_value = float(np.nanmax(finite_all)) if finite_all.size else 0.0
                invalid = ~np.isfinite(tmp) | (tmp < 0)
                tmp[invalid | exp_mask] = fill_value
                img = _log_normalize(tmp, eps)
                img[exp_mask] = -1.0
            else:
                img = _log_normalize(img, eps)
            record("log_and_normalize", img)
        elif name in ("fixed_mask", "mask"):
            mask_bool = _load_mask(params.get("mask", {}), tuple(img.shape[:2]), module_folder)
            img = img.copy()
            img[mask_bool] = float(params.get("mask", {}).get("mask_value", -1))
            record("fixed_mask", img)
        elif name == "add_mask_channel":
            if mask_bool is None:
                mask_bool = _load_mask(params.get("mask", {}), tuple(img.shape[:2]), module_folder)
            img = np.stack([img.astype(np.float32), mask_bool.astype(np.float32)], axis=-1)
            if return_steps:
                snapshots.append({"step": "add_mask_channel", "label": "add_mask_channel", "image": img[..., 0].copy()})
                snapshots.append({"step": "mask_channel", "label": "mask_channel", "image": img[..., 1].copy()})

    if img.ndim == 2:
        mask_bool = _load_mask(params.get("mask", {}), tuple(img.shape[:2]), module_folder)
        img[mask_bool] = -1.0
        img = np.stack([img.astype(np.float32), mask_bool.astype(np.float32)], axis=-1)

    return (img.astype(np.float32, copy=False), snapshots) if return_steps else img.astype(np.float32, copy=False)
