from __future__ import annotations

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.proposal_holdout_audit import (
    HoldoutAuditConfig,
    _summary,
)


def test_config_rejects_calibration_and_invalid_budgets():
    with pytest.raises(ValueError, match="frozen test"):
        HoldoutAuditConfig(split="calibration")
    with pytest.raises(ValueError, match="topology_recall_k"):
        HoldoutAuditConfig(topology_recall_k=35)
    with pytest.raises(ValueError, match="samples_per_mixture"):
        HoldoutAuditConfig(samples_per_mixture=0)


def test_summary_uses_ranks_and_oracle_continuous_distances():
    records = [
        {
            "topology_rank": 1,
            "pattern_rank_given_truth_topology": 1,
            "joint_branch_rank": 4,
            "oracle_branch_best_sample_rms": 0.02,
            "oracle_branch_best_mixture_center_rms": 0.03,
        },
        {
            "topology_rank": 9,
            "pattern_rank_given_truth_topology": 2,
            "joint_branch_rank": 40,
            "oracle_branch_best_sample_rms": 0.12,
            "oracle_branch_best_mixture_center_rms": 0.2,
        },
    ]
    summary = _summary(records, HoldoutAuditConfig(maximum_examples=2))

    assert summary["topology_accuracy"] == 0.5
    assert summary["topology_recall_at_8"] == 0.5
    assert summary["branch_pattern_accuracy_given_truth_topology"] == 0.5
    assert summary["joint_branch_recall_at_32"] == 0.5
    assert summary["oracle_branch_sample_recall_rms_0_05"] == 0.5
    assert summary["oracle_branch_sample_recall_rms_0_10"] == 0.5
    assert summary["oracle_branch_best_sample_rms"]["median"] == pytest.approx(0.07)


def test_slurm_wrapper_requires_versioned_paths():
    from pathlib import Path

    source = (
        Path(__file__).parents[1]
        / "PosteriorV8"
        / "slurm"
        / "holdout_audit_gpu.sbatch"
    ).read_text(encoding="utf-8")
    assert "POSTERIOR_V8_SOURCE_ROOT:?" in source
    assert "POSTERIOR_V8_DATASET_DIR:?" in source
    assert "POSTERIOR_V8_TRAINING_RUN:?" in source
    assert "POSTERIOR_V8_HOLDOUT_OUTPUT:?" in source
    assert "PosteriorV8.proposal_holdout_audit" in source
    assert "--range-selection" in source


def test_selection_priority_is_deterministic_and_seeded():
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8.proposal_holdout_audit import (
        _selection_priority,
    )

    assert _selection_priority(1, 2, 3) == _selection_priority(1, 2, 3)
    assert len({_selection_priority(seed, 2, 3) for seed in range(4)}) == 4
    assert isinstance(np.uint64(_selection_priority(1, 2, 3)), np.uint64)
