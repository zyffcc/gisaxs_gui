"""Content provenance for the complete bounds-conditioned V3 training chain."""

from __future__ import annotations

from pathlib import Path

from .proposal_training_audit import _file_sha256


BOUNDS_TRAINING_SOURCE_PATHS = (
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/proposal_training_v3.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/train_proposal_v3.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/slurm/bounds_proposal_v3_gpu4.sbatch",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/bounds_training_audit.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/bounds_training_data.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/bounds_training_steps.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/bounds_training_sources.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/bounds_model_contract.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/model_v3.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/training_objective_v3.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/training_objective_v2.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/training_objective.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/model.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/local_target.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/bounds_first_contract.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/bounds_first_dataset.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/bounds_first_schedule.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/bounds_first_shards.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/build_bounds_first_shards.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/branch_catalog.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/canonical_branch_catalog.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/canonical_component_slots.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/branch_codec.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/contract.py",
    "utils/ML_Fitting_1D_GISAXS/PosteriorV8/preprocessing.py",
)


def bounds_training_source_hashes() -> dict[str, str]:
    root = next(
        (
            parent
            for parent in Path(__file__).resolve().parents
            if (parent / "src").is_dir() and (parent / "utils").is_dir()
        ),
        None,
    )
    if root is None:
        raise RuntimeError("could not locate Posterior V8 source root")
    return {name: _file_sha256(root / name) for name in BOUNDS_TRAINING_SOURCE_PATHS}


__all__ = ["BOUNDS_TRAINING_SOURCE_PATHS", "bounds_training_source_hashes"]
