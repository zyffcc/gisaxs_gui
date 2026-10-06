"""How a recent file, folder or project is named in File ▸ Open Recent and on the Start page.

``name — parent folder name`` tells two frames of the same name apart without the whole path (the
path is the tooltip), and a tag marks folders and projects.
"""

from __future__ import annotations

from pathlib import Path

from .i18n import tr

PROJECT_SUFFIX = ".gimap"


def recent_tag(path) -> str:
    """`` (folder)`` or `` (project)`` in the interface language, or ``""`` for a frame."""
    path = Path(path)
    if path.suffix.lower() == PROJECT_SUFFIX:
        return tr(" (project)")
    try:
        if path.is_dir():
            return tr(" (folder)")
    except OSError:
        pass
    return ""


def recent_label(path, *, mnemonic_safe: bool = True) -> str:
    """``name — parent folder name`` plus the tag; ``&`` doubled for buttons and menus."""
    path = Path(path)
    name = path.name or str(path)
    parent = path.parent
    parent_name = parent.name or str(parent)  # a drive root (E:\) has no name
    text = name if parent == path or parent_name in ("", ".") else f"{name} — {parent_name}"
    if mnemonic_safe:
        text = text.replace("&", "&&")
    return text + recent_tag(path)


__all__ = ["PROJECT_SUFFIX", "recent_label", "recent_tag"]
