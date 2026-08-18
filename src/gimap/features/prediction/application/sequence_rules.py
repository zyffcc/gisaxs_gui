"""Application commands for prediction file sequence rules."""

from __future__ import annotations

from ..domain import build_complete_batches, extract_cbf_index, parse_index_range


class PredictionSequenceRules:
    def file_index(self, file_name: str):
        return extract_cbf_index(file_name)

    def index_range(self, text: str):
        return parse_index_range(text)

    def complete_batches(self, paths, batch_size: int):
        return build_complete_batches(paths, batch_size)
