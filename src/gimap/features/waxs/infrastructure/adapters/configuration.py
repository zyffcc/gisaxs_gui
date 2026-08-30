"""Local JSON adapter for portable WAXS configurations."""

import json
from pathlib import Path


class LocalWaxsConfigurationAdapter:
    def load(self, path: Path) -> dict:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("WAXS configuration JSON must contain an object.")
        if int(payload.get("version", 1)) != 1:
            raise ValueError("Unsupported WAXS configuration version.")
        return payload

    def save(self, path: Path, values: dict) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(values, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return target
