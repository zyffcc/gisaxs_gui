"""Lossless row-major bitmap of excluded pixels, independent of mask drawing and rule evaluation."""

import base64
import hashlib
import zlib

import numpy as np


def bitmap_record(mask, shape):
    """Alternating runs, beginning with unmasked pixels. Hash one byte (0 or 1) per pixel."""
    flat = (np.zeros(shape, dtype="u1") if mask is None
            else np.asarray(mask, dtype=bool).astype("u1", order="C")).ravel()
    changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    lengths = np.diff(np.r_[0, changes, flat.size])
    if flat[0]:
        lengths = np.r_[0, lengths]
    return {
        "shape": list(shape), "encoding": "rle-u64le-zlib-base64", "order": "C",
        "runs": base64.b64encode(zlib.compress(lengths.astype("<u8").tobytes())).decode("ascii"),
        "sha256": hashlib.sha256(flat).hexdigest(),
    }


def read_bitmap(record, frame_shape):
    """Validate against the detector's actual size before expanding any recorded runs."""
    if tuple(record["shape"]) != tuple(frame_shape):
        raise ValueError("The recorded mask shape differs from the detector frame.")
    if record["encoding"] != "rle-u64le-zlib-base64" or record["order"] != "C":
        raise ValueError("Unrecognised mask bitmap encoding.")
    size = int(frame_shape[0]) * int(frame_shape[1])
    try:
        decoder = zlib.decompressobj()
        raw = decoder.decompress(base64.b64decode(record["runs"], validate=True), (size + 1) * 8 + 1)
        if not decoder.eof or decoder.unused_data or len(raw) > (size + 1) * 8:
            raise ValueError("Invalid run lengths.")
        lengths = np.frombuffer(raw, dtype="<u8")
        if not lengths.size or np.any(lengths > size) or np.any(lengths[1:] == 0) or int(lengths.sum()) != size:
            raise ValueError("Invalid run lengths.")
        flat = np.repeat(np.arange(lengths.size) % 2 == 1, lengths.astype(np.intp))
    except (ValueError, zlib.error) as exc:
        raise ValueError(f"Invalid recorded mask bitmap: {exc}") from exc
    if hashlib.sha256(flat.astype("u1")).hexdigest() != record["sha256"]:
        raise ValueError("SHA-256 mismatch for the recorded mask bitmap.")
    return flat.reshape(frame_shape)
