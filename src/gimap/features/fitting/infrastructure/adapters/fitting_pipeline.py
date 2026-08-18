"""GUI-independent command planning and result handling for AI fitting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Dict, Sequence

from ...application.fitting_profiles import FittingProfile, profile_registry


@dataclass(frozen=True)
class FittingRequest:
    model_dir: Path
    input_csv: Path
    output_dir: Path
    profile: FittingProfile
    constraints_json: Path | None = None
    exact_nonempty: int | None = None
    initial_candidates_json: tuple[Path, ...] = ()
    allow_unsafe_lambda: bool = True


@dataclass(frozen=True)
class FittingRunSummary:
    profile: str
    runtime_seconds: float
    configured_candidates: int
    result_candidates: int
    best_log_rmse: float | None
    exit_code: int
    cancelled: bool = False


class FittingPipeline:
    """One pipeline whose cost/robustness is controlled only by a profile."""

    def __init__(self, script_path: Path | None = None) -> None:
        self.script_path = script_path or (
            Path(__file__).resolve().parents[6]
            / "utils"
            / "ML_Fitting_1D_GISAXS"
            / "Training"
            / "predict_topk.py"
        )

    @staticmethod
    def normalize_model_dir(path: Path) -> Path:
        path = Path(path)
        if path.is_file():
            return path.parent
        if path.name == "saved_model" and (path / "saved_model.pb").is_file():
            return path.parent
        return path

    def validate_request(self, request: FittingRequest) -> None:
        if not self.script_path.is_file():
            raise FileNotFoundError(f"AI fitting pipeline script is missing: {self.script_path}")
        model_dir = self.normalize_model_dir(request.model_dir)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"AI fitting model directory is missing: {model_dir}")
        if not any((model_dir / name).exists() for name in ("model.keras", "saved_model")):
            raise FileNotFoundError(f"No model.keras or saved_model artifact found in {model_dir}")
        if not request.input_csv.is_file():
            raise FileNotFoundError(f"AI fitting input curve is missing: {request.input_csv}")
        if request.exact_nonempty is not None and not 1 <= int(request.exact_nonempty) <= 4:
            raise ValueError("exact_nonempty must be between 1 and 4")

    def build_args(self, request: FittingRequest) -> list[str]:
        self.validate_request(request)
        profile = request.profile
        args = [
            str(self.script_path),
            "--model_dir",
            str(self.normalize_model_dir(request.model_dir)),
            "--input_csv",
            str(request.input_csv),
            "--output_dir",
            str(request.output_dir),
            "--num_samples",
            str(profile.candidate_count),
            "--top_k",
            str(profile.top_k),
            "--sampling_std",
            str(profile.sampling_std),
            "--sampling_scales",
            ",".join(str(scale) for scale in profile.sampling_scales),
            "--refine_top_n",
            str(profile.refinement_count),
            "--refine_max_nfev",
            str(profile.max_evaluations),
            "--refine_q_stride",
            str(profile.q_stride),
            "--refine_ftol",
            str(profile.tolerance),
            "--refine_xtol",
            str(profile.tolerance),
            "--refine_gtol",
            str(profile.tolerance),
            "--refine_target_logrmse",
            str(profile.target_log_rmse),
            "--refine_stall_patience",
            str(profile.stall_patience),
            "--refine_stall_tol",
            str(profile.stall_tolerance),
            "--complexity_penalty",
            str(profile.complexity_penalty),
            "--parameter_mode_radius",
            str(profile.parameter_mode_radius),
            "--progress_interval",
            str(profile.progress_interval),
            "--seed",
            str(profile.random_seed),
            "--score_mode",
            "hybrid_log_relative",
            "--rank_mode",
            "physics",
            "--include_mean_candidate",
        ]
        if profile.compare_full_candidates:
            args.append("--refine_best_per_k")
        if request.allow_unsafe_lambda:
            args.append("--allow_unsafe_lambda")
        if request.exact_nonempty is not None:
            args.extend(["--exact_nonempty", str(int(request.exact_nonempty))])
        if request.constraints_json is not None:
            args.extend(["--constraints_json", str(request.constraints_json)])
        for path in request.initial_candidates_json:
            args.extend(["--initial_candidates_json", str(path)])
        return args

    def write_request_metadata(self, request: FittingRequest) -> Path:
        request.output_dir.mkdir(parents=True, exist_ok=True)
        path = request.output_dir / "fitting_request.json"
        payload = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model_dir": str(self.normalize_model_dir(request.model_dir)),
            "input_csv": str(request.input_csv),
            "constraints_json": str(request.constraints_json) if request.constraints_json else None,
            "exact_nonempty": request.exact_nonempty,
            "initial_candidates_json": [str(item) for item in request.initial_candidates_json],
            "profile": request.profile.to_dict(),
            "command_args": self.build_args(request),
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @staticmethod
    def summarize(
        output_dir: Path,
        profile: FittingProfile,
        runtime_seconds: float,
        exit_code: int,
        cancelled: bool = False,
    ) -> FittingRunSummary:
        rows: list[Dict[str, Any]] = []
        results_path = Path(output_dir) / "top20_candidates.json"
        try:
            loaded = json.loads(results_path.read_text(encoding="utf-8"))
            rows = loaded if isinstance(loaded, list) else []
        except Exception:
            rows = []
        best = None
        if rows:
            value = rows[0].get("best_log_rmse")
            try:
                best = float(value)
            except (TypeError, ValueError):
                best = None
        return FittingRunSummary(
            profile=profile.name,
            runtime_seconds=float(runtime_seconds),
            configured_candidates=profile.candidate_count,
            result_candidates=len(rows),
            best_log_rmse=best,
            exit_code=int(exit_code),
            cancelled=bool(cancelled),
        )

    @staticmethod
    def write_summary(output_dir: Path, summary: FittingRunSummary) -> Path:
        path = Path(output_dir) / "fitting_run_summary.json"
        path.write_text(json.dumps(asdict(summary), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def run(
        self,
        request: FittingRequest,
        python_executable: str = sys.executable,
        progress: Callable[[str], None] | None = None,
    ) -> FittingRunSummary:
        """Synchronous entry point for scripts/tests; Qt calls ``build_args`` via QProcess."""
        self.write_request_metadata(request)
        started = time.perf_counter()
        process = subprocess.Popen(
            [python_executable, *self.build_args(request)],
            cwd=str(self.script_path.parents[3]),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            if progress is not None:
                progress(line.rstrip())
        exit_code = process.wait(timeout=request.profile.time_budget_seconds)
        summary = self.summarize(
            request.output_dir, request.profile, time.perf_counter() - started, exit_code
        )
        self.write_summary(request.output_dir, summary)
        return summary


fitting_pipeline = FittingPipeline()
