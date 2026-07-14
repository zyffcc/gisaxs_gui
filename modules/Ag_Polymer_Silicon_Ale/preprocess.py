import os
import numpy as np

try:
    from scipy.ndimage import binary_dilation, binary_fill_holes, median_filter
except Exception:
    binary_dilation = None  # type: ignore
    binary_fill_holes = None  # type: ignore
    median_filter = None  # type: ignore


def _crop(img: np.ndarray, cfg: dict) -> np.ndarray:
    if not isinstance(cfg, dict):
        return img
    if all(k in cfg for k in ("y0", "y1", "x0", "x1")):
        y0 = int(cfg.get("y0", 0)); y1 = int(cfg.get("y1", img.shape[0]))
        x0 = int(cfg.get("x0", 0)); x1 = int(cfg.get("x1", img.shape[1]))
    else:
        left = int(cfg.get("left", 0)); up = int(cfg.get("up", 0))
        down = int(cfg.get("down", 0)); right = int(cfg.get("right", 0))
        y0 = up; y1 = max(up, img.shape[0] - down)
        x0 = left; x1 = max(left, img.shape[1] - right)
    y0 = max(0, min(y0, img.shape[0])); y1 = max(y0, min(y1, img.shape[0]))
    x0 = max(0, min(x0, img.shape[1])); x1 = max(x0, min(x1, img.shape[1]))
    return img[y0:y1, x0:x1]


def _resize(img: np.ndarray, size) -> np.ndarray:
    method = 'bilinear'
    target = size
    if isinstance(size, dict):
        target = size.get('shape', size.get('size'))
        method = str(size.get('method', 'bilinear')).lower()

    if not isinstance(target, (list, tuple)) or len(target) != 2:
        return img
    th, tw = int(target[0]), int(target[1])

    if method == 'nearest':
        try:
            from scipy.ndimage import zoom  # type: ignore
            zh = float(th) / float(max(img.shape[0], 1))
            zw = float(tw) / float(max(img.shape[1], 1))
            return zoom(img, [zh, zw], order=0).astype(np.float32)
        except Exception:
            ys = np.linspace(0, img.shape[0] - 1, th).astype(np.int32)
            xs = np.linspace(0, img.shape[1] - 1, tw).astype(np.int32)
            return img[np.ix_(ys, xs)]

    tf_method = 'bilinear'
    np_order = 0 if method == 'nearest' else 1
    try:
        import tensorflow as tf  # type: ignore
        t = tf.convert_to_tensor(img[None, ..., None], dtype=tf.float32)
        r = tf.image.resize(t, [th, tw], method=tf_method)
        return r.numpy()[0, ..., 0]
    except Exception:
        try:
            from scipy.ndimage import zoom  # type: ignore
            zh = float(th) / float(max(img.shape[0], 1))
            zw = float(tw) / float(max(img.shape[1], 1))
            return zoom(img, [zh, zw], order=np_order).astype(np.float32)
        except Exception:
            ys = np.linspace(0, img.shape[0] - 1, th).astype(np.int32)
            xs = np.linspace(0, img.shape[1] - 1, tw).astype(np.int32)
            return img[np.ix_(ys, xs)]


def _set_invalid(img: np.ndarray, cfg: dict) -> np.ndarray:
    nan_val = float(cfg.get("nan", -1)) if isinstance(cfg, dict) else -1.0
    neg_val = float(cfg.get("negative", -1)) if isinstance(cfg, dict) else -1.0
    out = img.copy()
    out[np.isnan(out)] = nan_val
    out[out < 0] = neg_val
    return out


def _cut_columns(img: np.ndarray, cfg: dict) -> np.ndarray:
    if not isinstance(cfg, dict):
        return img
    start = int(cfg.get("start", 0)); end = int(cfg.get("end", 0))
    val = float(cfg.get("value", -1))
    out = img.copy()
    if 0 <= start < out.shape[1]:
        end = max(start, min(end, out.shape[1]))
        out[:, start:end] = val
    return out


def _cut_rows(img: np.ndarray, cfg: dict) -> np.ndarray:
    if not isinstance(cfg, dict):
        return img
    start = int(cfg.get("start", 0)); end = int(cfg.get("end", 0))
    val = float(cfg.get("value", -1))
    out = img.copy()
    if 0 <= start < out.shape[0]:
        end = max(start, min(end, out.shape[0]))
        out[start:end, :] = val
    return out


def _mask(img: np.ndarray, cfg: dict, module_folder: str) -> np.ndarray:
    if not isinstance(cfg, dict) or not cfg.get("apply", True):
        return img
    path = cfg.get("path")
    if isinstance(path, str) and not os.path.isabs(path):
        path = os.path.abspath(os.path.join(module_folder or "", path))
    if not path or not os.path.isfile(path):
        return img
    try:
        mask = np.load(path)
        crop_m = cfg.get("crop_mask") if isinstance(cfg, dict) else None
        if isinstance(crop_m, dict):
            left = int(crop_m.get("left", 0)); up = int(crop_m.get("up", 0))
            down = int(crop_m.get("down", 0)); right = int(crop_m.get("right", 0))
            y0 = up; y1 = max(up, mask.shape[0] - down)
            x0 = left; x1 = max(left, mask.shape[1] - right)
            mask = mask[y0:y1, x0:x1]
        resize = cfg.get("resize")
        if isinstance(resize, (list, tuple)) and len(resize) == 2:
            mh, mw = int(resize[0]), int(resize[1])
            try:
                import tensorflow as tf  # type: ignore
                mt = tf.convert_to_tensor(mask[None, ..., None], dtype=tf.float32)
                mr = tf.image.resize(mt, [mh, mw], method='nearest')
                mask = mr.numpy()[0, ..., 0]
            except Exception:
                ys = np.linspace(0, mask.shape[0] - 1, mh).astype(np.int32)
                xs = np.linspace(0, mask.shape[1] - 1, mw).astype(np.int32)
                mask = mask[np.ix_(ys, xs)]
        mv = float(cfg.get("mask_value", -1))
        out = img.copy()
        bad = mask != 0
        if bad.shape == out.shape:
            out[bad] = mv
        return out
    except Exception:
        return img


def _circular_mask(shape: tuple[int, int], center: tuple[int, int], radius: int) -> np.ndarray:
    h, w = shape
    y, x = np.ogrid[:h, :w]
    cy, cx = int(center[0]), int(center[1])
    r = int(radius)
    return (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2


def _build_experiment_mask(img: np.ndarray, cfg: dict | None = None) -> np.ndarray:
    """
    Build an experiment-data mask aligned with FF_test.ipynb behavior.
    """
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

    if median_filter is not None:
        med = median_filter(safe_data, size=median_size)
    else:
        med = safe_data
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
            if not isinstance(center, (list, tuple)) or len(center) != 2:
                continue
            if radius is None:
                continue
            beamstop_mask |= _circular_mask(data.shape, (int(center[0]), int(center[1])), int(radius))

    mask = gap_mask | badpixel_mask | beamstop_mask
    if dilate_iter > 0 and binary_dilation is not None:
        mask = binary_dilation(mask, iterations=dilate_iter)
    if binary_fill_holes is not None:
        mask = binary_fill_holes(mask)
    return mask.astype(bool, copy=False)


def _log_and_normalize(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Match FF_training.py clean-input preprocessing:
    scale = e / (max(img) + eps), then log(img * scale + eps), invalid -> -1.
    """
    x = img.astype(np.float32, copy=True)
    max_val = float(np.max(x)) if x.size else 0.0
    scale = float(np.e) / (max_val + float(eps))
    y = np.log(x * scale + float(eps))
    y = np.where(np.isfinite(y), y, -1.0).astype(np.float32, copy=False)
    return y


def run(
    image: np.ndarray,
    preprocess_cfg: dict | None = None,
    *,
    module_folder: str = "",
    return_steps: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[dict]]:
    """
    Modular preprocessing runner.
    Honors explicit steps order in preprocess_cfg['steps'] when provided.
    Falls back to model-aligned default used for experimental FF inference:
    crop (optional) -> resize -> experiment_mask (optional) -> log_and_normalize -> mask.

    When return_steps is True, returns a tuple (img, steps) where steps is a list of
    {"step": name, "image": snapshot_after_step} to aid debugging and UI logging.
    """
    if image is None:
        raise ValueError("image is None")
    img = image.astype(np.float32, copy=True)
    cfg = preprocess_cfg or {}
    params = cfg.get('params', {}) if isinstance(cfg, dict) else {}
    steps = cfg.get('steps', []) if isinstance(cfg, dict) else []

    steps_log: list[dict] = []
    exp_mask: np.ndarray | None = None

    def record_step(label: str) -> None:
        if not return_steps:
            return
        try:
            steps_log.append({"step": str(label), "image": img.copy()})
        except Exception:
            pass

    def do_step(step):
        nonlocal img, exp_mask
        if isinstance(step, dict):
            # {name: config}
            if len(step) != 1:
                return
            name, scfg = next(iter(step.items()))
        else:
            name, scfg = str(step), params.get(step, {})
        name = str(name).lower()
        if name == 'crop':
            img = _crop(img, scfg if isinstance(scfg, dict) else params.get('crop', {}))
            record_step('crop')
        elif name == 'resize':
            size = scfg if isinstance(scfg, (list, tuple)) else params.get('resize', None)
            img = _resize(img, size)
            record_step('resize')
        elif name in ('set_invalid', 'invalid', 'setinvalid'):
            img = _set_invalid(img, scfg if isinstance(scfg, dict) else params.get('set_invalid', {}))
            record_step('set_invalid')
        elif name in ('cut_columns', 'cutcols', 'vertical_cut'):
            img = _cut_columns(img, scfg if isinstance(scfg, dict) else params.get('cut_columns', {}))
            record_step('cut_columns')
        elif name in ('cut_rows', 'cutrows', 'bottom_cut'):
            img = _cut_rows(img, scfg if isinstance(scfg, dict) else params.get('cut_rows', {}))
            record_step('cut_rows')
        elif name in ('experiment_mask', 'auto_mask', 'gisaxs_mask'):
            ecfg = scfg if isinstance(scfg, dict) else params.get('experiment_mask', {})
            exp_mask = _build_experiment_mask(img, ecfg if isinstance(ecfg, dict) else {})
            record_step('experiment_mask')
        elif name == 'mask':
            mc = scfg if isinstance(scfg, dict) else params.get('mask', {})
            img = _mask(img, mc, module_folder)
            record_step('mask')
        elif name in ('log_and_normalize', 'lognormalize', 'log_norm'):
            lcfg = scfg if isinstance(scfg, dict) else params.get('log_and_normalize', {})
            eps = float(lcfg.get('eps', 1e-8)) if isinstance(lcfg, dict) else 1e-8
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
                img = _log_and_normalize(tmp, eps=eps)
                img[exp_mask] = -1.0
            else:
                img = _log_and_normalize(img, eps=eps)
            record_step('log_and_normalize')
        # Unrecognized names are ignored

    if steps:
        for st in steps:
            do_step(st)
        img_out = img.astype(np.float32, copy=False)
        return (img_out, steps_log) if return_steps else img_out

    # Default (matches FF_test exp flow + FF_training clean-input inference)
    # 1) crop
    if 'crop' in params:
        img = _crop(img, params.get('crop', {}))
        record_step('crop')
    # 2) resize
    if 'resize' in params:
        img = _resize(img, params.get('resize'))
        record_step('resize')
    # 3) optional experiment auto-mask
    if 'experiment_mask' in params and isinstance(params.get('experiment_mask'), dict):
        exp_mask = _build_experiment_mask(img, params.get('experiment_mask', {}))
        record_step('experiment_mask')
    # 4) log and normalize
    lcfg = params.get('log_and_normalize', {}) if isinstance(params, dict) else {}
    eps = float(lcfg.get('eps', 1e-8)) if isinstance(lcfg, dict) else 1e-8
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
        img = _log_and_normalize(tmp, eps=eps)
        img[exp_mask] = -1.0
    else:
        img = _log_and_normalize(img, eps=eps)
    record_step('log_and_normalize')
    # 5) fixed mask from file
    img = _mask(img, params.get('mask', {}), module_folder)
    record_step('mask')

    img_out = img.astype(np.float32, copy=False)
    return (img_out, steps_log) if return_steps else img_out
