"""Cross-feature normalization for user-entered local file paths."""

from __future__ import annotations

import os
import unicodedata
from urllib.parse import unquote, urlparse


def normalize_path(path: object) -> str:
    """Preserve Unicode, environment expansion and ``file://`` compatibility."""

    if path is None:
        return ""
    text = os.fspath(path) if hasattr(path, "__fspath__") else str(path)
    text = text.strip().strip("\"'")
    if not text:
        return ""

    if text.startswith("file:"):
        parsed = urlparse(text)
        if parsed.scheme == "file":
            netloc = parsed.netloc
            local_path = unquote(parsed.path)
            if os.name == "nt":
                text = f"//{netloc}{local_path}" if netloc else local_path.lstrip("/")
            else:
                text = local_path
        else:
            text = unquote(text)
    else:
        text = unquote(text)

    text = os.path.expandvars(os.path.expanduser(text))
    candidates = (
        text,
        unicodedata.normalize("NFC", text),
        unicodedata.normalize("NFD", text),
    )
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return os.path.normpath(candidate)
    return os.path.normpath(candidates[1])
