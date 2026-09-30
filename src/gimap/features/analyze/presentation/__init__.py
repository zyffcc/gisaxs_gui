"""Analyze presentation public API."""

from .view_model import AnalyzeState, AnalyzeViewModel

__all__ = ["AnalyzePage", "AnalyzeState", "AnalyzeViewModel"]


def __getattr__(name):
    # The page pulls in pyqtgraph; import it only when a caller asks for it.
    if name == "AnalyzePage":
        from .page import AnalyzePage

        return AnalyzePage
    raise AttributeError(name)
