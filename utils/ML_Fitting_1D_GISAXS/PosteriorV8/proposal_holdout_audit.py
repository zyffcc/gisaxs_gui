"""Frozen-holdout diagnostics for a trained global-target V8 proposal model.

This audit measures discrete beam recall and oracle-branch continuous proposal
coverage.  It deliberately does not call the exact refiner and therefore is
not a scientific acceptance report.  Calibration rows are never accepted as
an evaluation split.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from numbers import Integral
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import numpy as np
from scipy.special import expit

from .branch_catalog import branch_pattern_id
from .dataset import (
    RANGE_CODE,
    SPLIT_CODE,
    NumpyShard,
    load_shard,
)
from .inference_proposals import (
    BranchCondition,
    ContinuousProposalOutput,
    DiscreteProposalOutput,
    rank_joint_branches,
    sample_bounded_global_mixture,
)
from .proposal_training_audit import (
    HISTORY_FILE,
    MANIFEST_FILE,
    MODEL_FILE,
    RUN_MANIFEST_SCHEMA,
    inspect_phase2_shards,
    load_trained_proposal_model,
)
from .reference_bank import CompetingBranch


HOLDOUT_AUDIT_SCHEMA = "gisaxs.posterior_v8.global_proposal_holdout/v1"
HOLDOUT_AUDIT_VERSION = "posterior_v8_global_target_holdout_audit_v1"
SUPPORTED_EVALUATION_SPLITS = ("test",)
SUPPORTED_RANGE_SELECTIONS = ("all", "full", "wide", "narrow")


def _integer(value: int, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


@dataclass(frozen=True, kw_only=True)
class HoldoutAuditConfig:
    split: str = "test"
    range_selection: str = "full"
    maximum_examples: int = 512
    batch_size: int = 64
    topology_recall_k: int = 8
    joint_branch_recall_k: int = 32
    mixture_components: int = 12
    samples_per_mixture: int = 2
    seed: int = 20260902

    def __post_init__(self) -> None:
        split = str(self.split).strip().lower()
        if split not in SUPPORTED_EVALUATION_SPLITS:
            raise ValueError("proposal holdout audit only accepts the frozen test split")
        selection = str(self.range_selection).strip().lower()
        if selection not in SUPPORTED_RANGE_SELECTIONS:
            raise ValueError(
                f"range_selection must be one of {SUPPORTED_RANGE_SELECTIONS}"
            )
        for name in (
            "maximum_examples",
            "batch_size",
            "topology_recall_k",
            "joint_branch_recall_k",
            "mixture_components",
            "samples_per_mixture",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum=1))
        if self.topology_recall_k > 34:
            raise ValueError("topology_recall_k must be <= 34")
        object.__setattr__(self, "seed", _integer(self.seed, "seed"))
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "range_selection", selection)


def _read_json(path: Path) -> dict[str, object]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON artifact {path}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"JSON artifact must contain one object: {path}")
    return result


def _file_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular non-symlink file: {path}")
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    names = (
        "proposal_holdout_audit.py",
        "proposal_training_audit.py",
        "inference_proposals.py",
        "model.py",
        "dataset.py",
        "branch_catalog.py",
        "branch_codec.py",
        "contract.py",
    )
    return {name: _file_sha256(root / name) for name in names}


def _verified_model(run_dir: Path, dataset_audit):
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise ValueError("training run must be a regular directory")
    manifest = _read_json(run_dir / MANIFEST_FILE)
    history = _read_json(run_dir / HISTORY_FILE)
    if manifest.get("schema_version") != RUN_MANIFEST_SCHEMA:
        raise ValueError("training manifest schema is unsupported")
    if history.get("status") != "complete":
        raise ValueError("training history is not complete")
    if history.get("completed_epochs") != history.get("target_epochs"):
        raise ValueError("training stopped before its declared target epoch")
    if history.get("resume_contract_sha256") != manifest.get("resume_contract_sha256"):
        raise ValueError("training history and manifest disagree")
    trained_dataset = manifest.get("dataset_audit")
    if not isinstance(trained_dataset, Mapping) or trained_dataset.get(
        "fingerprint_sha256"
    ) != dataset_audit.payload.get("fingerprint_sha256"):
        raise ValueError("holdout shards do not match the training dataset fingerprint")
    model_path = run_dir / MODEL_FILE
    digest = _file_sha256(model_path)
    if digest != history.get("best_model_sha256"):
        raise ValueError("published model checksum does not match training history")
    model = load_trained_proposal_model(model_path)
    return model, manifest, history, digest


def _selection_priority(seed: int, recipe_index: int, view_index: int) -> int:
    encoded = f"{seed}:{recipe_index}:{view_index}".encode("ascii")
    return int.from_bytes(sha256(encoded).digest()[:8], "big")


def _eligible_indices(shard: NumpyShard, config: HoldoutAuditConfig) -> np.ndarray:
    selected = shard.arrays["assigned_split"] == SPLIT_CODE[config.split]
    if config.range_selection != "all":
        selected &= shard.arrays["range_regime"] == RANGE_CODE[config.range_selection]
    return np.flatnonzero(selected)


def _select_rows(dataset_audit, config: HoldoutAuditConfig):
    references = []
    for path in dataset_audit.shard_paths:
        shard = load_shard(path)
        for row in _eligible_indices(shard, config):
            recipe = int(shard.arrays["recipe_index"][row])
            view = int(shard.arrays["view_index"][row])
            references.append(
                (
                    _selection_priority(config.seed, recipe, view),
                    recipe,
                    view,
                    path,
                    int(row),
                )
            )
    references.sort(key=lambda value: value[:3])
    if not references:
        raise ValueError("no holdout rows match the requested split/range selection")
    return tuple(references[: config.maximum_examples]), len(references)


def _materialize_rows(references):
    inputs: dict[str, list[np.ndarray]] = {}
    labels: dict[str, list[np.ndarray]] = {}
    provenance = []
    current_path = None
    shard = None
    shard_inputs = shard_labels = None
    for priority, recipe, view, path, row in references:
        if path != current_path:
            shard = load_shard(path)
            shard_inputs, shard_labels = shard.training_data(split=None)
            current_path = path
        assert shard is not None and shard_inputs is not None and shard_labels is not None
        for name, value in shard_inputs.items():
            inputs.setdefault(name, []).append(np.asarray(value[row]))
        for name, value in shard_labels.items():
            labels.setdefault(name, []).append(np.asarray(value[row]))
        provenance.append(
            {
                "selection_priority": priority,
                "recipe_index": recipe,
                "view_index": view,
                "physical_cell_id": shard.arrays["physical_cell_id"][row].decode("ascii"),
                "component_count": int(shard.arrays["component_count"][row]),
                "noise_id": int(shard.arrays["noise_id"][row]),
                "q_window_id": int(shard.arrays["q_window_id"][row]),
                "range_regime": int(shard.arrays["range_regime"][row]),
            }
        )
    return (
        {name: np.stack(value) for name, value in inputs.items()},
        {name: np.stack(value) for name, value in labels.items()},
        provenance,
    )


def _model_outputs(model, inputs, batch_size: int):
    collected: dict[str, list[np.ndarray]] = {}
    count = inputs["x"].shape[0]
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        batch = {name: value[start:stop] for name, value in inputs.items()}
        outputs = model(batch, training=False)
        if not isinstance(outputs, Mapping):
            raise ValueError("proposal model must return named outputs")
        for name, value in outputs.items():
            array = np.asarray(value)
            if not np.all(np.isfinite(array)):
                raise FloatingPointError(f"model output {name!r} contains NaN/Inf")
            collected.setdefault(name, []).append(array)
    return {name: np.concatenate(value, axis=0) for name, value in collected.items()}


def _continuous_distances(output, condition, target, active, config, sample_seed):
    selected_mixtures = min(config.mixture_components, output.mixture_logits.size)
    sampled = sample_bounded_global_mixture(
        output,
        condition,
        mixture_limit=selected_mixtures,
        samples_per_mixture=config.samples_per_mixture,
        seed=sample_seed,
    )
    sample_distances = [
        float(
            np.sqrt(
                np.mean(
                    np.square(np.asarray(item.global_unit)[active] - target[active])
                )
            )
        )
        for item in sampled
    ]
    centers = expit(output.mixture_loc[:, active])
    center_distances = np.sqrt(np.mean(np.square(centers - target[active]), axis=1))
    return float(np.min(sample_distances)), float(np.min(center_distances)), len(sampled)


def _evaluate_examples(inputs, labels, outputs, provenance, config):
    records = []
    for row, metadata in enumerate(provenance):
        discrete = DiscreteProposalOutput(
            topology_logits=outputs["topology_logits"][row],
            branch_pattern_logits=outputs["branch_pattern_logits"][row],
        )
        ranked = rank_joint_branches(discrete, topology_limit=34)
        topology_id = int(labels["topology_id"][row])
        pattern_id = int(labels["branch_pattern_id"][row])
        truth = CompetingBranch(topology_id=topology_id, pattern_id=pattern_id)
        scored = next(item for item in ranked if item.branch == truth)
        condition = BranchCondition(
            scored_branch=scored,
            branch_low=tuple(float(value) for value in inputs["branch_low"][row]),
            branch_high=tuple(float(value) for value in inputs["branch_high"][row]),
            active_dimension_mask=tuple(
                bool(value) for value in inputs["active_dimension_mask"][row]
            ),
        )
        continuous = ContinuousProposalOutput(
            mixture_logits=outputs["mixture_logits"][row],
            mixture_loc=outputs["mixture_loc"][row],
            mixture_logscale=outputs["mixture_logscale"][row],
        )
        target = np.asarray(labels["target_unit"][row], dtype=np.float64)
        active = np.asarray(labels["active_dimension_mask"][row], dtype=bool)
        sample_seed = int(
            np.random.SeedSequence(
                [config.seed, metadata["recipe_index"], metadata["view_index"]]
            ).generate_state(1, dtype=np.uint32)[0]
        )
        best_sample, best_center, sample_count = _continuous_distances(
            continuous,
            condition,
            target,
            active,
            config,
            sample_seed,
        )
        topology_prediction = int(
            sorted(range(34), key=lambda value: (-discrete.topology_logits[value], value))[0]
        )
        records.append(
            {
                **metadata,
                "topology_id": topology_id,
                "branch_pattern_id": pattern_id,
                "expected_branch_pattern_id": branch_pattern_id(
                    truth.d_present + (False,) * (4 - len(truth.d_present)),
                    truth.resolution_present,
                ),
                "active_dimension_count": int(np.count_nonzero(active)),
                "topology_prediction": topology_prediction,
                "topology_rank": scored.topology_rank,
                "pattern_rank_given_truth_topology": scored.pattern_rank_within_topology,
                "joint_branch_rank": scored.joint_rank,
                "truth_joint_log_score": scored.joint_log_score,
                "oracle_branch_best_sample_rms": best_sample,
                "oracle_branch_best_mixture_center_rms": best_center,
                "oracle_branch_sample_count": sample_count,
            }
        )
    return records


def _quantiles(values):
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0:
        return None
    return {
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "maximum": float(np.max(array)),
    }


def _summary(records, config):
    count = len(records)
    return {
        "example_count": count,
        "topology_accuracy": float(
            np.mean([item["topology_rank"] == 1 for item in records])
        ),
        f"topology_recall_at_{config.topology_recall_k}": float(
            np.mean(
                [item["topology_rank"] <= config.topology_recall_k for item in records]
            )
        ),
        "branch_pattern_accuracy_given_truth_topology": float(
            np.mean([item["pattern_rank_given_truth_topology"] == 1 for item in records])
        ),
        f"joint_branch_recall_at_{config.joint_branch_recall_k}": float(
            np.mean(
                [
                    item["joint_branch_rank"] <= config.joint_branch_recall_k
                    for item in records
                ]
            )
        ),
        "oracle_branch_best_sample_rms": _quantiles(
            item["oracle_branch_best_sample_rms"] for item in records
        ),
        "oracle_branch_best_center_rms": _quantiles(
            item["oracle_branch_best_mixture_center_rms"] for item in records
        ),
        "oracle_branch_sample_recall_rms_0_05": float(
            np.mean([item["oracle_branch_best_sample_rms"] <= 0.05 for item in records])
        ),
        "oracle_branch_sample_recall_rms_0_10": float(
            np.mean([item["oracle_branch_best_sample_rms"] <= 0.10 for item in records])
        ),
    }


def _stratified(records, config):
    result = {}
    for field in ("component_count", "noise_id", "q_window_id"):
        result[field] = {}
        for value in sorted({item[field] for item in records}):
            selected = [item for item in records if item[field] == value]
            result[field][str(value)] = _summary(selected, config)
    return result


def _write_json_exclusive(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite holdout audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite holdout audit: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def run_holdout_audit(
    shard_paths: Sequence[str | os.PathLike[str]],
    training_run: str | os.PathLike[str],
    output: str | os.PathLike[str],
    config: HoldoutAuditConfig = HoldoutAuditConfig(),
) -> dict[str, object]:
    """Evaluate a completed model on an untouched deterministic holdout subset."""

    if not isinstance(config, HoldoutAuditConfig):
        raise TypeError("config must be a HoldoutAuditConfig")
    dataset_audit = inspect_phase2_shards(shard_paths)
    run_dir = Path(training_run).resolve()
    model, manifest, history, model_sha256 = _verified_model(run_dir, dataset_audit)
    references, eligible_count = _select_rows(dataset_audit, config)
    inputs, labels, provenance = _materialize_rows(references)
    outputs = _model_outputs(model, inputs, config.batch_size)
    records = _evaluate_examples(inputs, labels, outputs, provenance, config)
    sources = _source_hashes()
    payload = {
        "schema": HOLDOUT_AUDIT_SCHEMA,
        "version": HOLDOUT_AUDIT_VERSION,
        "interpretation": (
            "supervised global-target proposal diagnostic; not exact-forward acceptance"
        ),
        "config": asdict(config),
        "training_run": {
            "path": str(run_dir),
            "resume_contract_sha256": manifest["resume_contract_sha256"],
            "dataset_fingerprint_sha256": dataset_audit.payload["fingerprint_sha256"],
            "best_epoch": history["best_epoch"],
            "best_validation_loss": history["best_validation_loss"],
            "model_sha256": model_sha256,
        },
        "selection": {
            "eligible_row_count": eligible_count,
            "evaluated_row_count": len(records),
            "calibration_rows_consumed": 0,
        },
        "summary": _summary(records, config),
        "stratified": _stratified(records, config),
        "examples": records,
        "source_sha256": sources,
        "source_sha256_aggregate": sha256(
            json.dumps(sources, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    _write_json_exclusive(Path(output).resolve(), payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", nargs="+", required=True, type=Path)
    parser.add_argument("--training-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--range-selection", choices=SUPPORTED_RANGE_SELECTIONS, default="full")
    parser.add_argument("--maximum-examples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--topology-recall-k", type=int, default=8)
    parser.add_argument("--joint-branch-recall-k", type=int, default=32)
    parser.add_argument("--mixture-components", type=int, default=12)
    parser.add_argument("--samples-per-mixture", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260902)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = HoldoutAuditConfig(
        range_selection=args.range_selection,
        maximum_examples=args.maximum_examples,
        batch_size=args.batch_size,
        topology_recall_k=args.topology_recall_k,
        joint_branch_recall_k=args.joint_branch_recall_k,
        mixture_components=args.mixture_components,
        samples_per_mixture=args.samples_per_mixture,
        seed=args.seed,
    )
    payload = run_holdout_audit(args.shards, args.training_run, args.output, config)
    print(json.dumps(payload["summary"], sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "HOLDOUT_AUDIT_SCHEMA",
    "HOLDOUT_AUDIT_VERSION",
    "HoldoutAuditConfig",
    "main",
    "run_holdout_audit",
]
