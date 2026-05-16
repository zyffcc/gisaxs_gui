# -*- coding: utf-8 -*-
"""Path helpers for user-entered Qt/file-system paths."""

from __future__ import annotations

import os
import unicodedata
from urllib.parse import unquote, urlparse


def normalize_path(path: object) -> str:
    """Return a filesystem path that is safe for Unicode and file:// inputs."""
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
                if netloc:
                    text = f"//{netloc}{local_path}"
                else:
                    text = local_path.lstrip("/")
            else:
                text = local_path
        else:
            text = unquote(text)
    else:
        text = unquote(text)

    text = os.path.expandvars(os.path.expanduser(text))

    # macOS filesystems often store Unicode filenames in decomposed form.
    # Try both canonical forms and keep the one that actually exists.
    candidates = [
        text,
        unicodedata.normalize("NFC", text),
        unicodedata.normalize("NFD", text),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return os.path.normpath(candidate)

    return os.path.normpath(candidates[1])
