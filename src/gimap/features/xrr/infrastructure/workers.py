"""Spawn-safe JobRunner handler for streaming XRR extraction."""

from __future__ import annotations

from .adapters.detector_series import LocalXrrSeriesRepository
from .serialization import (
    encode_preview,
    point_to_payload,
    request_from_payload,
    result_to_payload,
)
from ..application import ExtractXrrSeries


def process_xrr_series_job(payload, report, is_cancelled):
    request = request_from_payload(payload)
    workflow = ExtractXrrSeries(LocalXrrSeriesRepository())

    def progress(value) -> None:
        report(
            value.completed,
            value.total,
            f"Extracted {value.point.source_name}",
            {
                "point": point_to_payload(value.point),
                "preview": encode_preview(value.preview),
            },
        )

    return result_to_payload(
        workflow.execute(request, on_progress=progress, is_cancelled=is_cancelled)
    )


__all__ = ["process_xrr_series_job"]
