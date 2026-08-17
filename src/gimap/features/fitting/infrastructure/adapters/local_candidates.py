"""现有 AI candidate JSON 格式 adapter。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonCandidateRepository:
    def load(self, output_dir: Path) -> tuple[dict[str, Any], ...]:
        path = Path(output_dir) / "top20_candidates.json"
        if not path.is_file():
            raise FileNotFoundError(f"AI candidate results not found: {path}")
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("AI candidate results must contain a JSON list")
        return tuple(dict(row) for row in rows if isinstance(row, dict))
