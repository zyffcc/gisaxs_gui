"""WAXS JobRunner handlers。"""

import time
from pathlib import Path

from .adapters.detector_images import CalibrationWaxsImageRepository
from .adapters.local_files import LocalWaxsExportAdapter, LocalWaxsFileCatalog
from .batch_serialization import request_from_payload, result_to_payload
from ..application import ProcessWaxsBatch


def process_waxs_batch_job(payload, report, is_cancelled):
    values = dict(payload)
    control_file = Path(values.pop("_control_file"))
    request = request_from_payload(values)
    workflow = ProcessWaxsBatch(
        CalibrationWaxsImageRepository(),
        LocalWaxsFileCatalog(),
        LocalWaxsExportAdapter(),
    )

    def progress(value):
        report(
            value.completed,
            value.total,
            f"Processed {value.name}",
            {"name": value.name, "status": value.status},
        )

    def wait_if_paused():
        while (
            control_file.exists()
            and control_file.read_text(encoding="utf-8").strip() == "paused"
            and not is_cancelled()
        ):
            time.sleep(0.1)

    return result_to_payload(
        workflow.execute(
            request,
            on_progress=progress,
            is_cancelled=is_cancelled,
            wait_if_paused=wait_if_paused,
        )
    )
