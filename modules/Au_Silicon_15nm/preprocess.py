import os
import numpy as np


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
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        return img
    th, tw = int(size[0]), int(size[1])
    try:
        import tensorflow as tf  # type: ignore
        t = tf.convert_to_tensor(img[None, ..., None], dtype=tf.float32)
        r = tf.image.resize(t, [th, tw], method='bilinear')
        return r.numpy()[0, ..., 0]
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


def _log_and_normalize(img: np.ndarray) -> np.ndarray:
    try:
        # Use the existing project utility for parity
        from utils.tools.Preprocessing import Preprocessing  # type: ignore
        return Preprocessing(img).log_and_normalize()
    except Exception:
        # Fallback: simple safe log1p and min-max
        x = np.log(img + 1e-8)
        x = np.nan_to_num(x, nan=-1.0, posinf=0.0, neginf=-1.0)
        mn = float(np.min(x))
        mx = float(np.max(x))
        if mx - mn > 1e-12:
            x = (x - mn) / (mx - mn)
        return x


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
    Falls back to notebook-aligned default: crop -> resize -> set_invalid -> cut_columns -> log_and_normalize -> mask -> cut_columns -> cut_rows.

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

    def record_step(label: str) -> None:
        if not return_steps:
            return
        try:
            steps_log.append({"step": str(label), "image": img.copy()})
        except Exception:
            pass

    def do_step(step):
        nonlocal img
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
        elif name == 'mask':
            mc = scfg if isinstance(scfg, dict) else params.get('mask', {})
            img = _mask(img, mc, module_folder)
            record_step('mask')
        elif name in ('log_and_normalize', 'lognormalize', 'log_norm'):
            img = _log_and_normalize(img)
            record_step('log_and_normalize')
        # Unrecognized names are ignored

    if steps:
        for st in steps:
            do_step(st)
        img_out = img.astype(np.float32, copy=False)
        return (img_out, steps_log) if return_steps else img_out

    # Default (matches notebook Cells ~5–12 pipeline)
    # 1) crop
    if 'crop' in params:
        img = _crop(img, params.get('crop', {}))
        record_step('crop')
    # 2) resize
    if 'resize' in params:
        img = _resize(img, params.get('resize'))
        record_step('resize')
    # 3) set invalid values
    img = _set_invalid(img, params.get('set_invalid', {'nan': -1, 'negative': -1}))
    record_step('set_invalid')
    # 4) vertical cut of first N columns
    img = _cut_columns(img, params.get('cut_columns', {'start': 0, 'end': 20, 'value': -1}))
    record_step('cut_columns_pre')
    # 5) log and normalize
    img = _log_and_normalize(img)
    record_step('log_and_normalize')
    # 6) mask
    img = _mask(img, params.get('mask', {}), module_folder)
    record_step('mask')
    # 7) vertical cut again (after normalization)
    img = _cut_columns(img, params.get('cut_columns', {'start': 0, 'end': 20, 'value': -1}))
    record_step('cut_columns_post')
    # 8) bottom rows cut
    img = _cut_rows(img, params.get('cut_rows', {'start': 240, 'end': 256, 'value': -1}))
    record_step('cut_rows')

    img_out = img.astype(np.float32, copy=False)
    return (img_out, steps_log) if return_steps else img_out
