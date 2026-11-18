"""The ``trace`` folder of an Analyze result: the detail files of its steps and GIMaP's record of its step tree.

A detail file is read only when the user opens its step (``read_detail``: its JSON, and the arrays of the ``.npz``
it names, without pickles and within ``DETAIL_LIMIT``); nothing is kept here. GIMaP writes the tree it was told during
the run as ``trace/gimap_steps.json`` (``save_steps``) so that the result shows it later; a result without it but
with a ``trace`` folder gets its tree rebuilt (``application/live_steps.py`` ``StepTree.from_files``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

from ..application.live_steps import STEPS_RECORD, TRACE_FOLDER
from ..application.step_details import BAD_DETAIL, NO_DETAIL, StepDetail, detail_kind

DETAIL_LIMIT = 32 << 20
"""The largest detail file read (bytes; the package writes a few MB at most)."""


def _plain_name(name: str) -> bool:
    return bool(name) and Path(name).name == name and name not in (".", "..")


def detail_path(folders, name: str) -> Optional[Path]:
    """The first of ``folders`` that holds the detail file ``name`` (None when none does, or the name is a path)."""
    if not _plain_name(str(name)):
        return None
    for folder in folders:
        path = Path(folder) / name
        if path.is_file():
            return path
    return None


def read_detail(folders, name: str) -> StepDetail:
    """The detail file ``name`` from the first of ``folders`` that has it, with the arrays its JSON names. Raises
    ``FileNotFoundError`` / ``ValueError`` with the reason."""
    folders = [Path(folder) for folder in folders]
    path = detail_path(folders, name)
    if path is None:
        raise FileNotFoundError(NO_DETAIL.format(folder=folders[0] if folders else "", name=name))
    try:
        if path.stat().st_size > DETAIL_LIMIT:
            raise ValueError("the file is too large")
        doc = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            raise ValueError("not a JSON object")
        arrays = {}
        named = doc.get("arrays")
        if isinstance(named, str) and _plain_name(named) and (path.parent / named).is_file():
            if (path.parent / named).stat().st_size > DETAIL_LIMIT:
                raise ValueError("the arrays are too large")
            with np.load(path.parent / named, allow_pickle=False) as stored:
                arrays = {key: np.array(stored[key]) for key in stored.files}
    except (OSError, ValueError, EOFError, KeyError) as exc:
        raise ValueError(BAD_DETAIL.format(reason=f"{type(exc).__name__}: {exc}")) from exc
    return StepDetail(detail_kind(name), str(name), doc, arrays)


def trace_files(folder: Path) -> Optional[set]:
    """The names in a result's ``trace`` folder (None when it has none)."""
    trace = Path(folder) / TRACE_FOLDER
    try:
        return {path.name for path in trace.iterdir() if path.is_file()} if trace.is_dir() else None
    except OSError:
        return None


def steps_record(folder: Path) -> Optional[dict]:
    """GIMaP's record of the step tree of a result (None without one)."""
    try:
        record = json.loads((Path(folder) / TRACE_FOLDER / STEPS_RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def save_steps(folder: Path, record: dict) -> Path:
    """Keep the step tree with the result (``trace/gimap_steps.json``); raises ``OSError`` when it cannot."""
    trace = Path(folder) / TRACE_FOLDER
    trace.mkdir(parents=True, exist_ok=True)
    temporary = trace / f".{STEPS_RECORD}.tmp"
    temporary.write_text(json.dumps(record, indent=1), encoding="utf-8")
    os.replace(temporary, trace / STEPS_RECORD)
    return trace / STEPS_RECORD


__all__ = ["DETAIL_LIMIT", "detail_path", "read_detail", "save_steps", "steps_record", "trace_files"]
