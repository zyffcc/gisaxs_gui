"""joblib classification model repository。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ...domain import SavedModelPackage


class LazyJoblibPipeline:
    """延迟读取 worker 产生的 sklearn pipeline artifact。"""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._value = None

    def load(self):
        if self._value is None:
            import joblib

            self._value = joblib.load(self.path)
        return self._value

    def __getattr__(self, name):
        return getattr(self.load(), name)


class JoblibClassificationModelRepository:
    def save(self, path: Path, package: SavedModelPackage) -> None:
        import joblib

        pipeline = package.pipeline
        if isinstance(pipeline, LazyJoblibPipeline):
            package = replace(package, pipeline=pipeline.load())
        joblib.dump(package, Path(path))

    def load(self, path: Path) -> SavedModelPackage:
        import joblib

        package = joblib.load(Path(path))
        if not isinstance(package, SavedModelPackage):
            if isinstance(package, dict) and package.get("dr_type") == "t-SNE":
                raise ValueError(
                    "This legacy model uses t-SNE as a classification feature and cannot transform new samples."
                )
            raise ValueError("Unsupported classification model package")
        return package
