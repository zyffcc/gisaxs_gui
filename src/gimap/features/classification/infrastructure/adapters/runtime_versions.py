"""Installed distribution version adapter；不导入 ML runtime。"""

from importlib.metadata import PackageNotFoundError, version


class ImportlibRuntimeVersionAdapter:
    def version(self, distribution: str) -> str:
        try:
            return version(distribution)
        except PackageNotFoundError:
            return "not-installed"
