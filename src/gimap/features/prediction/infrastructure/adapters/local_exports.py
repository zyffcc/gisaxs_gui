"""Local filesystem adapter for prediction text exports."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class LocalPredictionExportRepository:
    def write_text(self, path: Path, content: str) -> Path:
        target = Path(path)
        target.write_text(content, encoding="utf-8")
        return target

    def write_array(
        self,
        path: Path,
        values,
        *,
        fmt: str,
        header: str,
        comments: str,
    ) -> Path:
        target = Path(path)
        np.savetxt(
            target,
            np.asarray(values),
            fmt=fmt,
            header=header,
            comments=comments,
        )
        return target
