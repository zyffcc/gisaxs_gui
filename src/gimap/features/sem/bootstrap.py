"""Compose local SEM measurement and comparison without a model runtime."""

from .application import SemAnalysisService
from .infrastructure import LocalSemCsvReader, LocalSemExporter, LocalSemImageReader, LocalSemInstanceSegmenter


def create_sem_service() -> SemAnalysisService:
    return SemAnalysisService(LocalSemImageReader(), LocalSemCsvReader(), LocalSemExporter(), LocalSemInstanceSegmenter())


def create_sem_window(parent=None):
    from .presentation.view_model import SemViewModel
    from .presentation.window import SemWindow

    return SemWindow(SemViewModel(create_sem_service()), parent=parent)
