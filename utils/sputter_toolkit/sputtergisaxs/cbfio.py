"""Pilatus CBF reading (fabio if available, otherwise a minimal byte_offset decoder)."""
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
_MARK = b"\x0c\x1a\x04\xd5"


class CBFDamagedError(ValueError):
    """Frame cannot give the header-declared pixel grid (truncated or undecodable payload)."""

    def __init__(self, reason: str, **detail):
        super().__init__(f"{reason} {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


def _layout(head: bytes) -> dict:
    """Binary-section geometry from the CBF header bytes (before the payload marker)."""
    start = head.find(_MARK)
    if start < 0:
        raise CBFDamagedError("CBF binary marker missing")
    out = dict(payload_start=start + 4)
    for key, pat in (("payload_bytes", rb"X-Binary-Size:\s*(\d+)"),
                     ("nx", rb"X-Binary-Size-Fastest-Dimension:\s*(\d+)"),
                     ("ny", rb"X-Binary-Size-Second-Dimension:\s*(\d+)")):
        m = re.search(pat, head[:start])
        if m is None:
            raise CBFDamagedError("CBF binary header incomplete", missing=key)
        out[key] = int(m.group(1))
    out["declared_end_bytes"] = out["payload_start"] + out["payload_bytes"]
    return out


def _head(stream, chunk: int = 65536) -> bytes:
    """The file bytes up to and including the binary-section marker (the whole file when there is none)."""
    head = b""
    while True:
        block = stream.read(chunk)
        if not block:
            return head
        head += block
        if head.find(_MARK, max(0, len(head) - len(block) - len(_MARK) + 1)) >= 0:
            return head


def payload_check(path) -> dict:
    """Header-only check: the declared compressed payload must lie inside the file."""
    path = Path(path)
    with path.open("rb") as stream:
        layout = _layout(_head(stream))
    layout["actual_bytes"] = path.stat().st_size
    if layout["actual_bytes"] < layout["declared_end_bytes"]:
        raise CBFDamagedError("CBF payload truncated before declared X-Binary-Size end", **layout)
    return layout


def _parse_header(text: str) -> dict:
    out = {}
    for key, pat in _HDR.items():
        m = re.search(pat, text)
        if m:
            out[key] = m.group(1) if key == "timestamp" else float(m.group(1))
    return out


def _decode_byte_offset(raw: bytes) -> tuple[np.ndarray, str]:
    layout = _layout(raw)
    start = layout["payload_start"] - 4
    hdr = raw[:start].decode("latin-1", "replace")
    nx, ny, nbytes = layout["nx"], layout["ny"], layout["payload_bytes"]
    if len(raw) < layout["declared_end_bytes"]:
        raise CBFDamagedError("CBF payload truncated before declared X-Binary-Size end", actual_bytes=len(raw), **layout)
    buf = np.frombuffer(raw[start + 4:start + 4 + nbytes], dtype=np.int8)
    ub = buf.view(np.uint8)
    n = nx * ny
    deltas = np.empty(n, dtype=np.int64)
    out_i = pos = 0
    for e in np.flatnonzero(buf == -128):
        if e < pos:
            continue
        k = e - pos
        if out_i + k >= n or e + 3 > len(buf):
            raise CBFDamagedError("CBF byte_offset stream inconsistent with header dimensions", decoded_pixels=out_i, expected_pixels=n)
        deltas[out_i:out_i + k] = buf[pos:e]
        out_i += k
        v = int(np.frombuffer(ub[e + 1:e + 3].tobytes(), "<i2")[0])
        if v == -32768:
            if e + 7 > len(buf):
                raise CBFDamagedError("CBF byte_offset escape runs past payload end", decoded_pixels=out_i, expected_pixels=n)
            v = int(np.frombuffer(ub[e + 3:e + 7].tobytes(), "<i4")[0])
            pos = e + 7
        else:
            pos = e + 3
        deltas[out_i] = v
        out_i += 1
        if out_i >= n:
            break
    if out_i < n:
        tail = buf[pos:pos + n - out_i]
        if len(tail) < n - out_i:
            raise CBFDamagedError("CBF payload decodes to fewer pixels than header dimensions",
                                  decoded_pixels=out_i + len(tail), expected_pixels=n)
        deltas[out_i:] = tail
    return np.cumsum(deltas).reshape(ny, nx), hdr


def read_cbf(path, count_limit: float = 1e6) -> tuple[np.ndarray, dict]:
    """Return (image float64 with NaN for invalid pixels, header dict).

    A truncated or undecodable frame raises CBFDamagedError (a ValueError);
    no partial image is returned and no pixel is invented.
    """
    path = Path(path)
    layout = payload_check(path)
    try:
        import fabio  # type: ignore
        obj = fabio.open(str(path))
        img = obj.data.astype(np.float64)
        hdr_text = "\n".join(f"# {k} {v}" for k, v in obj.header.items()) + str(obj.header.get("_array_data.header_contents", ""))
    except Exception:
        arr, hdr_text = _decode_byte_offset(path.read_bytes())
        img = arr.astype(np.float64)
    if img.shape != (layout["ny"], layout["nx"]):
        raise CBFDamagedError("CBF decoded shape differs from header dimensions", decoded_shape=list(img.shape), **layout)
    img[(img < 0) | (img > count_limit)] = np.nan
    return img, _parse_header(hdr_text)


def sum_frames(paths) -> tuple[np.ndarray, np.ndarray, float]:
    """Sum frames; returns (sum of counts, number of valid frames per pixel, total exposure)."""
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
