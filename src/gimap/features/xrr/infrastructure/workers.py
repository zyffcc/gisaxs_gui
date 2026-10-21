"""Spawn-safe JobRunner handler for streaming XRR extraction."""

from __future__ import annotations

from .adapters.detector_series import LocalXrrSeriesRepository
from .serialization import (
    encode_preview,
    layout_to_payload,
    point_to_payload,
    request_from_payload,
    result_to_payload,
)
from ..application import ExtractXrrSeries


def process_xrr_series_job(payload, report, is_cancelled):
    request = request_from_payload(payload)
    workflow = ExtractXrrSeries(LocalXrrSeriesRepository())

    def progress(value) -> None:
        details = {
            "point": point_to_payload(value.point),
            "preview": encode_preview(value.preview),
        }
        if value.completed == 1 and value.layout is not None:
            # Said once, with the first frame: a job that is cancelled returns no result.
            details["layout"] = layout_to_payload(value.layout)
        report(value.completed, value.total, f"Extracted {value.point.source_name}", details)

    return result_to_payload(
        workflow.execute(request, on_progress=progress, is_cancelled=is_cancelled)
    )


__all__ = ["process_xrr_series_job"]
