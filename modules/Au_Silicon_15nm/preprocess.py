import os
import numpy as np


_DEFAULT_STEPS = (
    "crop",
    "resize",
    "set_invalid",
    "mask",
    "cut_columns",
    "log_and_normalize",
    "mask",
    "cut_columns",
    "cut_rows",
)


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


def _resize(img: np.ndarray, size) -> np.ndarray:
    method = "nearest"
    target = size
    if isinstance(size, dict):
        target = size.get("shape", size.get("size"))
        method = str(size.get("method", method)).strip().lower()
    if not isinstance(target, (list, tuple)) or len(target) != 2:
        raise ValueError("resize requires shape: [height, width]")

    th, tw = int(target[0]), int(target[1])
    if th <= 0 or tw <= 0 or img.ndim != 2 or not all(img.shape):
        raise ValueError(f"invalid resize from {img.shape} to {(th, tw)}")
    interpolation_order = {"nearest": 0, "bilinear": 1}.get(method)
    if interpolation_order is None:
        raise ValueError(f"unsupported resize method: {method!r}")

    # The Au model's experiment notebook defines resize_matrix with
    # scipy.ndimage.zoom(order=0).  This exact pixel mapping matters because the
    # model flattens its convolution output into a very large dense layer.
    from scipy.ndimage import zoom

    factors = (float(th) / img.shape[0], float(tw) / img.shape[1])
    resized = zoom(img, factors, order=interpolation_order)
    if resized.shape != (th, tw):
        raise ValueError(f"resize produced {resized.shape}, expected {(th, tw)}")
    return resized.astype(np.float32, copy=False)


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
    start = int(cfg.get("start", 0))
    end = int(cfg.get("end", 0))
    val = float(cfg.get("value", -1))
    out = img.copy()
    if 0 <= start < out.shape[1]:
        end = max(start, min(end, out.shape[1]))
        out[:, start:end] = val
    return out


def _cut_rows(img: np.ndarray, cfg: dict) -> np.ndarray:
    if not isinstance(cfg, dict):
        return img
    start = int(cfg.get("start", 0))
    end = int(cfg.get("end", 0))
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
        raise FileNotFoundError(f"configured prediction mask not found: {path}")

    mask = np.load(path)
    crop_m = cfg.get("crop_mask") if isinstance(cfg, dict) else None
    if isinstance(crop_m, dict):
        left = int(crop_m.get("left", 0))
        up = int(crop_m.get("up", 0))
        down = int(crop_m.get("down", 0))
        right = int(crop_m.get("right", 0))
        y0 = up
        y1 = max(up, mask.shape[0] - down)
        x0 = left
        x1 = max(left, mask.shape[1] - right)
        mask = mask[y0:y1, x0:x1]
    resize = cfg.get("resize")
    if isinstance(resize, (list, tuple)) and len(resize) == 2:
        mask = _resize(mask.astype(np.float32), {"shape": resize, "method": "nearest"})
    if mask.shape != img.shape:
        raise ValueError(
            f"prediction mask shape {mask.shape} does not match image shape {img.shape}"
        )

    mv = float(cfg.get("mask_value", -1))
    out = img.copy()
    out[mask != 0] = mv
    return out


def _log_and_normalize(img: np.ndarray, cfg: dict) -> np.ndarray:
    """Apply the max-scaled logarithm used to train and validate this model."""
    config = cfg if isinstance(cfg, dict) else {}
    eps = float(config.get("eps", 1e-8))
    if eps <= 0:
        raise ValueError("log_and_normalize.eps must be positive")

    max_source = img.astype(np.float32, copy=True)
    exclude = config.get("exclude_from_max", {})
    if isinstance(exclude, dict):
        columns = exclude.get("columns")
        if isinstance(columns, (list, tuple)) and len(columns) == 2:
            max_source[:, int(columns[0]) : int(columns[1])] = -1.0
        rows = exclude.get("rows")
        if isinstance(rows, (list, tuple)) and len(rows) == 2:
            max_source[int(rows[0]) : int(rows[1]), :] = -1.0

    finite = max_source[np.isfinite(max_source)]
    max_value = float(np.max(finite)) if finite.size else float("nan")
    if not np.isfinite(max_value) or max_value <= 0:
        raise ValueError("log_and_normalize requires at least one positive finite intensity")

    log_argument = img.astype(np.float32, copy=False) * (np.e / (max_value + eps)) + eps
    output = np.full(img.shape, -1.0, dtype=np.float32)
    valid = np.isfinite(log_argument) & (log_argument > 0)
    output[valid] = np.log(log_argument[valid])
    return output


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
    Falls back to the same sequence as the module YAML.  The resize and
    normalization parameters remain configuration-owned; there is no alternate
    silent preprocessing algorithm.

    When return_steps is True, returns a tuple (img, steps) where steps is a list of
    {"step": name, "image": snapshot_after_step} to aid debugging and UI logging.
    """
    if image is None:
        raise ValueError("image is None")
    img = image.astype(np.float32, copy=True)
    cfg = preprocess_cfg or {}
    params = cfg.get("params", {}) if isinstance(cfg, dict) else {}
    steps = cfg.get("steps", []) if isinstance(cfg, dict) else []
    configured_steps = list(steps) if steps else list(_DEFAULT_STEPS)

    steps_log: list[dict] = []
    step_totals: dict[str, int] = {}
    step_seen: dict[str, int] = {}

    for configured_step in configured_steps:
        if isinstance(configured_step, dict):
            if len(configured_step) != 1:
                raise ValueError("inline preprocessing step must contain exactly one entry")
            configured_name = str(next(iter(configured_step)))
        else:
            configured_name = str(configured_step)
        canonical_name = configured_name.strip().lower()
        step_totals[canonical_name] = step_totals.get(canonical_name, 0) + 1

    logarithm_applied = False

    def record_step(label: str, *, masked_value: float | None = None) -> None:
        if not return_steps:
            return
        occurrence = step_seen.get(label, 0) + 1
        step_seen[label] = occurrence
        total = step_totals.get(label, 1)
        display_label = f"{label} ({occurrence}/{total})" if total > 1 else label
        snapshot = {
            "step": str(label),
            "label": display_label,
            "image": img.copy(),
            # Raw detector intensities need logarithmic display to make weak
            # scattering visible. This is preview metadata only; the snapshot
            # and model tensor retain their exact scientific values.
            "display_scale": "linear" if logarithm_applied else "log_positive",
        }
        if masked_value is not None:
            snapshot["masked_value"] = float(masked_value)
        steps_log.append(snapshot)

    def do_step(step):
        nonlocal img, logarithm_applied
        if isinstance(step, dict):
            if len(step) != 1:
                raise ValueError("inline preprocessing step must contain exactly one entry")
            name, scfg = next(iter(step.items()))
        else:
            name, scfg = str(step), params.get(step, {})
        name = str(name).strip().lower()
        if name == "crop":
            img = _crop(img, scfg if isinstance(scfg, dict) else params.get("crop", {}))
            record_step("crop")
        elif name == "resize":
            img = _resize(img, scfg)
            record_step("resize")
        elif name in ("set_invalid", "invalid", "setinvalid"):
            invalid_cfg = scfg if isinstance(scfg, dict) else {}
            img = _set_invalid(img, invalid_cfg)
            record_step("set_invalid", masked_value=float(invalid_cfg.get("negative", -1)))
        elif name in ("cut_columns", "cutcols", "vertical_cut"):
            cut_cfg = scfg if isinstance(scfg, dict) else {}
            img = _cut_columns(img, cut_cfg)
            record_step("cut_columns", masked_value=float(cut_cfg.get("value", -1)))
        elif name in ("cut_rows", "cutrows", "bottom_cut"):
            cut_cfg = scfg if isinstance(scfg, dict) else {}
            img = _cut_rows(img, cut_cfg)
            record_step("cut_rows", masked_value=float(cut_cfg.get("value", -1)))
        elif name == "mask":
            mc = scfg if isinstance(scfg, dict) else {}
            img = _mask(img, mc, module_folder)
            record_step("mask", masked_value=float(mc.get("mask_value", -1)))
        elif name in ("log_and_normalize", "lognormalize", "log_norm"):
            img = _log_and_normalize(img, scfg if isinstance(scfg, dict) else {})
            logarithm_applied = True
            record_step("log_and_normalize", masked_value=-1.0)
        else:
            raise ValueError(f"unknown preprocessing step: {name!r}")

    for configured_step in configured_steps:
        do_step(configured_step)

    img_out = img.astype(np.float32, copy=False)
    return (img_out, steps_log) if return_steps else img_out
