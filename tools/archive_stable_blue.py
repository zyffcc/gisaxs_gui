"""Archive release evidence/source snapshots without copying large training data."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "modules/Fitting_1D_Model/Workflow_v5"
EVIDENCE = BUNDLE / "development/evidence/stable_blue_20260922"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest_path = BUNDLE / "IMPORT_MANIFEST.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    legacy = {k: v for k, v in previous["files"].items() if k.startswith("conditional_v2/")}
    assert all(sha(BUNDLE / name) == digest for name, digest in legacy.items())
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    copied = {}

    def copy(source, relative):
        target = EVIDENCE / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied[str(relative).replace("\\", "/")] = dict(
            original_path=source.relative_to(ROOT).as_posix(), sha256=sha(target)
        )

    for path in sorted((ROOT / "validation/stable_blue_20260922").iterdir()):
        if path.is_file():
            copy(path, path.name)
    verified = ROOT / "validation/stable_release_20260922"
    for origin, name in (
        ("VERIFIED.json", "VERIFIED.json"),
        ("PRE_POLISH_VERIFIED.json", "PRE_POLISH_VERIFIED.json"),
        ("worker/VERIFIED.json", "worker_verification.json"),
        ("ui/VERIFIED.json", "gui_verification.json"),
        ("ui/stable_prediction.png", "stable_prediction.png"),
        ("ui/stable_insitu.png", "stable_insitu.png"),
    ):
        copy(verified / origin, name)

    sources = [
        "application/workflow_v5.py", "domain/blue_rc_forward.py", "domain/cbf_observations.py",
        "infrastructure/adapters/stable_blue.py", "infrastructure/adapters/workflow_v5.py",
        "infrastructure/adapters/experimental_fit.py", "presentation/workflow_v5_dialog.py",
        "presentation/ai_controls.py", "presentation/bindings/workflow_v5_binding.py",
        "presentation/bindings/ai_job_execution.py", "presentation/bindings/insitu_cut_processing.py",
        "presentation/bindings/insitu_recipe_binding.py", "presentation/bindings/fit_result_export.py",
        "presentation/bindings/insitu_refinement_lifecycle.py", "presentation/views/insitu_workflow_controls.py",
    ]
    for relative in sources:
        path = Path("src/gimap/features/fitting") / relative
        copy(ROOT / path, Path("source_snapshot") / path)
    for name in ("probe_blue_calibration", "check_blue_release_domain", "convert_blue_numpy",
                 "check_stable_predict_workflow", "report_stable_blue", "archive_stable_blue"):
        copy(ROOT / "tools" / f"{name}.py", Path("tools") / f"{name}.py")
    for name in ("test_stable_blue_core", "test_stable_metadata_ui", "test_stable_predict_workflow", "test_stable_exports"):
        copy(ROOT / "tests" / f"{name}.py", Path("tests") / f"{name}.py")
    (EVIDENCE / "ARTIFACT_MANIFEST.json").write_text(json.dumps(dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        note="Source snapshots are research/maintenance records; execute tools from the complete GUI project. Prior training data remain in the sibling blue_curve_distill_20260921 archive.",
        files=copied,
    ), indent=2), encoding="utf-8")
    paths = [p for p in BUNDLE.rglob("*") if p.is_file() and p != manifest_path
             and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    previous.update(integrated_on="2026-09-22", paths_relative_to="Workflow_v5",
                    files={p.relative_to(BUNDLE).as_posix(): sha(p) for p in sorted(paths)})
    manifest_path.write_text(json.dumps(previous, indent=2), encoding="utf-8")
    assert all(sha(BUNDLE / name) == digest for name, digest in previous["files"].items())
    print(json.dumps(dict(files=len(paths), bytes=sum(p.stat().st_size for p in paths),
                          frozen_legacy_files_unchanged=len(legacy),
                          archived_release_files=len(copied))))


if __name__ == "__main__":
    main()
