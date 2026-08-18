"""QProcess adapter for portable local Trainset jobs."""

from __future__ import annotations

import re
from pathlib import Path

from PyQt5.QtCore import QObject, QProcess

from ...application.models import TrainsetLocalProcessRequest


class QtTrainsetLocalProcessAdapter(QObject):
    def __init__(self):
        super().__init__()
        self._process: QProcess | None = None
        self._control_file: Path | None = None
        self._output_buffer = ""

    def is_running(self) -> bool:
        return bool(
            self._process is not None
            and self._process.state() != QProcess.NotRunning
        )

    def start(
        self,
        request: TrainsetLocalProcessRequest,
        *,
        on_started,
        on_progress,
        on_log,
        on_finished,
        on_error,
    ) -> None:
        if self.is_running():
            raise RuntimeError("A local generation/training process is already running.")
        process = QProcess(self)
        process.setWorkingDirectory(str(request.package_dir))
        process.setProcessChannelMode(QProcess.MergedChannels)
        self._control_file = Path(request.package_dir) / ".local_control"
        self._control_file.write_text("running", encoding="utf-8")
        arguments = list(request.arguments) + [
            "--control-file",
            str(self._control_file),
        ]
        self._output_buffer = ""

        def read_output() -> None:
            self._output_buffer += bytes(
                process.readAllStandardOutput()
            ).decode(errors="replace")
            lines = self._output_buffer.splitlines(keepends=True)
            if lines and not lines[-1].endswith(("\n", "\r")):
                self._output_buffer = lines.pop()
            else:
                self._output_buffer = ""
            for raw_line in lines:
                line = raw_line.strip()
                if not line:
                    continue
                match = re.match(r"^PROGRESS\s+(\d+)\s+(\d+)\s+(.*)$", line)
                if match:
                    completed = int(match.group(1))
                    total = max(1, int(match.group(2)))
                    percent = max(
                        0, min(100, int(round(100.0 * completed / total)))
                    )
                    on_progress(percent, match.group(3))
                else:
                    on_log(line)

        process.readyReadStandardOutput.connect(read_output)
        process.started.connect(on_started)
        process.finished.connect(lambda exit_code, _status: on_finished(exit_code))
        process.errorOccurred.connect(lambda _error: on_error(process.errorString()))
        self._process = process
        process.start(str(request.python_executable), arguments)

    def set_paused(self, paused: bool) -> bool:
        if not self.is_running() or self._control_file is None:
            return False
        self._control_file.write_text(
            "paused" if paused else "running", encoding="utf-8"
        )
        return True

    def cancel(self) -> bool:
        if not self.is_running() or self._control_file is None:
            return False
        self._control_file.write_text("cancelled", encoding="utf-8")
        return True
