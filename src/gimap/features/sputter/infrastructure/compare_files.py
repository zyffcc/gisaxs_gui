"""Files of Compare Scans: the results analysed under a results root, a result's report table, the exported tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


class CompareFolderFiles:
    def analysed(self, root: Path, limit: int) -> list:
        """``<root>/<scan>-<hash>/iampred-*`` (and ``<root>/iampred-*``) folders whose ``analysis.json`` says ``ok``
        and that hold ``main/main.json``: the newest ``limit``."""
        root = Path(root)
        if not root.is_dir():
            return []
        found = []
        for analysis in list(root.glob("*/iampred-*/analysis.json")) + list(root.glob("iampred-*/analysis.json")):
            folder = analysis.parent
            try:
                status = json.loads(analysis.read_text(encoding="utf-8")).get("status")
                stamp = (folder / "main" / "main.json").stat().st_mtime
            except (OSError, ValueError, AttributeError):
                continue
            if status == "ok":
                found.append((stamp, folder))
        return [folder for _stamp, folder in sorted(found, key=lambda item: item[0], reverse=True)[:limit]]

    @staticmethod
    def report_text(path: Optional[Path]) -> str:
        if path is None:
            return ""
        try:
            return Path(path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return ""

    @staticmethod
    def write_text(path: Path, text: str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path


__all__ = ["CompareFolderFiles"]
