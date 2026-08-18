"""现有 classification data service 的 port adapter。"""

from __future__ import annotations

from ...application.models import ImportedDataset
from .dataset import ClassificationDataService


class LegacyClassificationDatasetAdapter:
    """保留既有格式/QC/预处理语义，同时把实现隔离在 infrastructure。"""

    def __init__(self, service=None):
        if service is None:
            service = ClassificationDataService()
        self.service = service

    def import_sources(self, sources, *, on_progress=None, is_cancelled=None):
        samples = self.service.scan_sources(sources)
        self.service.load_samples(
            samples,
            progress=on_progress,
            is_cancelled=is_cancelled,
        )
        return ImportedDataset(tuple(samples), self.service.validate_dataset(samples))

    def build_feature_matrix(self, samples, preprocessing, *, require_labels):
        return self.service.build_feature_matrix(
            list(samples), preprocessing, require_labels=require_labels
        )

    def validate_dataset(self, samples):
        return self.service.validate_dataset(list(samples))

    def summarize_by_label(self, samples):
        return self.service.summarize_by_label(list(samples))
