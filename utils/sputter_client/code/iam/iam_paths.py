'Machine paths of the IAM scripts: iam_paths.json next to this file, each entry overridable by an environment'

from __future__ import annotations

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
_CFG = json.loads((HERE / "iam_paths.json").read_text(encoding="utf-8")) if (HERE / "iam_paths.json").exists() else {}


def _get(key, default):
    v = os.environ.get("IAM_" + key.upper()) or _CFG.get(key) or default
    p = Path(v)
    return p if p.is_absolute() else (HERE / p).resolve()


GIMAP = _get("gimap", "../../gimap_forward")
RAW = _get("raw", "raw")
DIST = _get("dist", "../../dist/sputter_struct_v1")
DATA = _get("data", "../_scratch/iam_data")
WORK = _get("work", "../_scratch/iam")
WORK.mkdir(parents=True, exist_ok=True)
