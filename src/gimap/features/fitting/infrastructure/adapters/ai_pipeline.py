"""现有 AI fitting pipeline 的 Predictor/JobRunner adapter。"""

from __future__ import annotations

import contextlib
import json
import os
import re
import runpy
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.gimap.app.jobs import JobRequest

from ...application.ai_models import CandidateGenerationRequest, CandidateGenerationResult
from ...application.fitting_profiles import FittingProfile
from .fitting_pipeline import FittingRequest, fitting_pipeline


class AiPipelinePredictor:
    def create_job_request(self, request: CandidateGenerationRequest) -> JobRequest:
        output_dir = Path(request.output_dir).expanduser().resolve()
        if request.clear_output_dir:
            self._clear_reusable_output(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        input_csv = output_dir / "input_curve.csv"
        np.savetxt(
            input_csv,
            np.column_stack([request.q, request.intensity, request.sigma]),
            delimiter=",",
            header="q,I,sigma",
            comments="",
        )
        constraints_path = None
        if request.constraints:
            constraints_path = output_dir / "constraints.json"
            constraints_path.write_text(
                json.dumps(request.constraints, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        profile = FittingProfile(**request.profile)
        fitting_request = FittingRequest(
            model_dir=Path(request.model_path),
            input_csv=input_csv,
            output_dir=output_dir,
            profile=profile,
            constraints_json=constraints_path,
            exact_nonempty=request.exact_nonempty,
            allow_unsafe_lambda=request.allow_unsafe_lambda,
        )
        args = fitting_pipeline.build_args(fitting_request)
        fitting_pipeline.write_request_metadata(fitting_request)
        return JobRequest(
            handler=(
                "src.gimap.features.fitting.infrastructure.adapters.ai_pipeline:"
                "run_ai_pipeline_job"
            ),
            payload={
                "script_path": str(fitting_pipeline.script_path),
                "args": args[1:],
                "output_dir": str(output_dir),
                "profile": request.profile,
                "working_directory": str(Path.cwd()),
            },
            timeout_seconds=profile.time_budget_seconds,
        )

    @staticmethod
    def _clear_reusable_output(output_dir: Path) -> None:
        """只清理 GUI 约定的可复用目录，拒绝任意递归删除目标。"""
        if output_dir.name != "current_prediction":
            raise ValueError(
                "Refusing to clear an AI output directory not named current_prediction"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        for child in output_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    def decode_result(self, request, value) -> CandidateGenerationResult:
        if not isinstance(value, dict):
            raise ValueError("AI fitting worker returned an invalid result")
        summary = value.get("summary") or {}
        rows = value.get("candidates") or []
        return CandidateGenerationResult(
            output_dir=Path(value.get("output_dir") or request.output_dir),
            profile_name=str(summary.get("profile", request.profile.get("name", ""))),
            runtime_seconds=float(summary.get("runtime_seconds", 0.0)),
            configured_candidates=int(
                summary.get("configured_candidates", request.profile.get("candidate_count", 0))
            ),
            candidates=tuple(dict(row) for row in rows),
            best_log_rmse=(
                None
                if summary.get("best_log_rmse") is None
                else float(summary["best_log_rmse"])
            ),
            exit_code=int(summary.get("exit_code", 0)),
        )


class _ProgressWriter:
    def __init__(self, report):
        self._report = report
        self._buffer = ""

    def write(self, text):
        self._buffer += str(text)
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._emit(line.rstrip())
        return len(text)

    def flush(self):
        if self._buffer.strip():
            self._emit(self._buffer.rstrip())
        self._buffer = ""

    def _emit(self, line: str):
        if not line:
            return
        match = re.search(r"Progress\s+(\d+)/(\d+)", line)
        if match:
            completed, total = int(match.group(1)), max(1, int(match.group(2)))
        else:
            completed, total = 0, 1
        self._report(completed, total, line, {"line": line})


def run_ai_pipeline_job(payload, report, is_cancelled):
    """在 JobRunner worker 内运行脚本；不在 GUI 进程 import TensorFlow。"""
    script = Path(payload["script_path"]).resolve()
    output_dir = Path(payload["output_dir"]).resolve()
    profile = FittingProfile(**payload["profile"])
    if not script.is_file():
        raise FileNotFoundError(f"AI fitting pipeline script is missing: {script}")
    if is_cancelled():
        raise RuntimeError("AI fitting job was cancelled before startup")

    previous_argv = sys.argv[:]
    previous_cwd = Path.cwd()
    started = time.perf_counter()
    exit_code = 0
    writer = _ProgressWriter(report)
    try:
        os.chdir(payload.get("working_directory") or previous_cwd)
        sys.path.insert(0, str(script.parent))
        sys.argv = [str(script), *[str(value) for value in payload.get("args", [])]]
        with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
            try:
                runpy.run_path(str(script), run_name="__main__")
            except SystemExit as exc:
                exit_code = int(exc.code or 0)
                if exit_code != 0:
                    raise RuntimeError(f"AI fitting pipeline exited with code {exit_code}") from exc
    finally:
        writer.flush()
        sys.argv = previous_argv
        os.chdir(previous_cwd)

    summary = fitting_pipeline.summarize(
        output_dir,
        profile,
        time.perf_counter() - started,
        exit_code,
    )
    fitting_pipeline.write_summary(output_dir, summary)
    candidates_path = output_dir / "top20_candidates.json"
    candidates = []
    if candidates_path.is_file():
        loaded = json.loads(candidates_path.read_text(encoding="utf-8"))
        candidates = loaded if isinstance(loaded, list) else []
    report(1, 1, "AI fitting completed", {"result_candidates": len(candidates)})
    return {
        "output_dir": str(output_dir),
        "summary": asdict(summary),
        "candidates": candidates,
    }
