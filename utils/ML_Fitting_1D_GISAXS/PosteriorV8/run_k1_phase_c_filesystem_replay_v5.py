"""CLI for a local/Slurm worker to replay a complete K1 Phase-C snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .grouped_artifact_v5 import canonical_json
from .k1_phase_c_evaluation_v5 import assess_v5_k1_phase_c_replay_receipt
from .k1_phase_c_filesystem_replay_v5 import V5K1PhaseCFilesystemReplayAdapter
from .k1_phase_c_plan_v5 import build_v5_k1_phase_c_plan
from .k1_phase_c_replay_runner_v5 import (
    read_and_replay_v5_k1_phase_c_receipt,
    run_v5_k1_phase_c_replay,
)
from .k1_phase_c_writer_capability_v5 import (
    verify_v5_k1_phase_c_writer_receipt,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay hash-bound lossless K1 Phase-C filesystem artifacts."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-file-sha256", required=True)
    parser.add_argument(
        "--writer-receipt",
        type=Path,
        default=None,
        help="Required for formal replay; verified into one live adapter capability.",
    )
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--nonformal",
        action="store_true",
        help="Run a non-claiming fixture plan instead of the frozen formal plan.",
    )
    parser.add_argument(
        "--parents-per-branch",
        type=int,
        default=None,
        help="Required only to select a non-default nonformal fixture population.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.nonformal and args.parents_per_branch is not None:
        raise ValueError("formal replay uses the frozen parents-per-branch count")
    plan = build_v5_k1_phase_c_plan(
        formal=not args.nonformal,
        parents_per_branch=args.parents_per_branch,
    )
    if plan.formal and args.writer_receipt is None:
        raise ValueError("formal replay requires --writer-receipt")
    capability = (
        None
        if args.writer_receipt is None
        else verify_v5_k1_phase_c_writer_receipt(args.writer_receipt)
    )
    adapter = V5K1PhaseCFilesystemReplayAdapter(
        args.manifest,
        expected_manifest_file_sha256=args.manifest_file_sha256,
        writer_capability=capability,
    )
    run_v5_k1_phase_c_replay(plan=plan, port=adapter, receipt_path=args.receipt)
    checked = read_and_replay_v5_k1_phase_c_receipt(
        args.receipt,
        plan=plan,
        port=adapter,
    )
    assessment = assess_v5_k1_phase_c_replay_receipt(checked, plan=plan)
    persisted = json.loads(args.receipt.read_text(encoding="utf-8"))
    print(
        canonical_json(
            {
                "receipt": str(args.receipt.resolve()),
                "receipt_sha256": persisted["receipt_sha256"],
                "formal": persisted["formal"],
                "claim_eligible": persisted["claim_eligible"],
                "assessment": assessment,
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through worker entrypoint
    raise SystemExit(main())
