'Pilatus CBF reading (fabio if available, otherwise a minimal byte_offset decoder).'
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_HDR = {
    "exposure_s": r"Exposure_time ([\d.eE+-]+)",
    "period_s": r"Exposure_period ([\d.eE+-]+)",
    "threshold_ev": r"Threshold_setting:?\s*([\d.]+)",
    "timestamp": r"# (\d{4}-\d\d-\d\dT[\d:.]+)",
}


def _parse_header(text: str) -> dict:
    out = {}
    for key, pat in _HDR.items():
        m = re.search(pat, text)
        if m:
            out[key] = m.group(1) if key == "timestamp" else float(m.group(1))
    return out


def _decode_byte_offset(raw: bytes) -> tuple[np.ndarray, str]:
    start = raw.find(b"\x0c\x1a\x04\xd5")
    hdr = raw[:start].decode("latin-1", "replace")
    nx = int(re.search(r"X-Binary-Size-Fastest-Dimension:\s*(\d+)", hdr).group(1))
    ny = int(re.search(r"X-Binary-Size-Second-Dimension:\s*(\d+)", hdr).group(1))
    nbytes = int(re.search(r"X-Binary-Size:\s*(\d+)", hdr).group(1))
    buf = np.frombuffer(raw[start + 4:start + 4 + nbytes], dtype=np.int8)
    ub = buf.view(np.uint8)
    n = nx * ny
    deltas = np.empty(n, dtype=np.int64)
    out_i = pos = 0
    for e in np.flatnonzero(buf == -128):
        if e < pos:
            continue
        k = e - pos
        deltas[out_i:out_i + k] = buf[pos:e]
        out_i += k
        v = int(np.frombuffer(ub[e + 1:e + 3].tobytes(), "<i2")[0])
        if v == -32768:
            v = int(np.frombuffer(ub[e + 3:e + 7].tobytes(), "<i4")[0])
            pos = e + 7
        else:
            pos = e + 3
        deltas[out_i] = v
        out_i += 1
        if out_i >= n:
            break
    if out_i < n:
        deltas[out_i:] = buf[pos:pos + n - out_i]
    return np.cumsum(deltas).reshape(ny, nx), hdr


def read_cbf(path, count_limit: float = 1e6) -> tuple[np.ndarray, dict]:
    'Return (image float64 with NaN for invalid pixels, header dict).'
    path = Path(path)
    try:
        import fabio
        obj = fabio.open(str(path))
        img = obj.data.astype(np.float64)
        hdr_text = "\n".join(f"# {k} {v}" for k, v in obj.header.items()) + str(obj.header.get("_array_data.header_contents", ""))
    except Exception:
        arr, hdr_text = _decode_byte_offset(path.read_bytes())
        img = arr.astype(np.float64)
    img[(img < 0) | (img > count_limit)] = np.nan
    return img, _parse_header(hdr_text)


def sum_frames(paths) -> tuple[np.ndarray, np.ndarray, float]:
    'Sum frames; returns (sum of counts, number of valid frames per pixel, total exposure).'
    total = None
    nvalid = None
    exposure = 0.0
    for p in paths:
        img, hdr = read_cbf(p)
        ok = np.isfinite(img)
        if total is None:
            total = np.zeros_like(img)
            nvalid = np.zeros(img.shape, dtype=np.int32)
        total[ok] += img[ok]
        nvalid += ok
        exposure += float(hdr.get("exposure_s", np.nan))
    return total, nvalid, exposure
