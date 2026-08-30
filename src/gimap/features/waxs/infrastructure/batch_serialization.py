"""WAXS batch JobRunner JSON serialization。"""

from pathlib import Path

from ..application import (
    WaxsBatchItem,
    WaxsBatchRequest,
    WaxsBatchResult,
    WaxsBatchSource,
)


def request_to_payload(request: WaxsBatchRequest) -> dict:
    return {
        "folder": str(request.folder),
        "pattern": request.pattern,
        "output_folder": str(request.output_folder),
        "export_images": request.export_images,
        "export_curves": request.export_curves,
        "export_background_subtracted": request.export_background_subtracted,
        "display": request.display,
        "geometry": request.geometry,
        "integration": request.integration,
        "mask_min": request.mask_min,
        "mask_max": request.mask_max,
        "timeout_seconds": request.timeout_seconds,
        "continue_on_error": request.continue_on_error,
        "sources": [
            {
                "folder": str(source.folder),
                "pattern": source.pattern,
                "output_subfolder": source.output_subfolder,
            }
            for source in request.sources
        ],
        "export_q_images": request.export_q_images,
        "export_curve_images": request.export_curve_images,
        "q_range": request.q_range,
        "calibration_enabled": request.calibration_enabled,
        "calibration_target_q": request.calibration_target_q,
        "calibration_half_width": request.calibration_half_width,
        "normalization_enabled": request.normalization_enabled,
        "normalization_target_q": request.normalization_target_q,
        "normalization_half_width": request.normalization_half_width,
        "normalization_target_intensity": request.normalization_target_intensity,
        "normalization_mode": request.normalization_mode,
    }


def request_from_payload(value: dict) -> WaxsBatchRequest:
    payload = dict(value)
    payload["folder"] = Path(payload["folder"])
    payload["output_folder"] = Path(payload["output_folder"])
    payload["sources"] = tuple(
        WaxsBatchSource(
            folder=Path(source["folder"]),
            pattern=str(source.get("pattern", "*.tif")),
            output_subfolder=source.get("output_subfolder"),
        )
        for source in payload.get("sources", ())
    )
    return WaxsBatchRequest(**payload)


def result_to_payload(result: WaxsBatchResult) -> dict:
    return {
        "items": [
            {
                "path": str(item.path),
                "frame_index": item.frame_index,
                "name": item.name,
                "status": item.status,
                "error_message": item.error_message,
            }
            for item in result.items
        ],
        "cancelled": result.cancelled,
    }


def result_from_payload(value: dict) -> WaxsBatchResult:
    return WaxsBatchResult(
        tuple(
            WaxsBatchItem(
                Path(item["path"]),
                int(item["frame_index"]),
                item["name"],
                item["status"],
                item.get("error_message"),
            )
            for item in value["items"]
        ),
        bool(value.get("cancelled", False)),
    )
