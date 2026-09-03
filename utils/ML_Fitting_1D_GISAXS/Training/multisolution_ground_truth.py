"""Read-only access to the V5 ragged multi-solution ground-truth sidecar."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class MultiSolutionGroundTruth:
    """Map a V5 stable sample ID/index to parameter-distinct curve solutions.

    The three stored tiers are *exclusive* labels for the tightest threshold a
    pair satisfies.  ``max_tier`` queries are cumulative, e.g. 0.03 returns
    both the 0.01 and 0.03 pairs.
    """

    VALID_TIERS = (0.01, 0.03, 0.05)

    def __init__(self, sidecar_dir):
        root = Path(sidecar_dir)
        self._pairs_pack = np.load(root / "multi_solution_pairs.npz", allow_pickle=False)
        self._catalog_pack = np.load(root / "solution_parameter_catalog.npz", allow_pickle=False)
        self.target_ids = self._pairs_pack["target_id"].astype(str)
        self.catalog_ids = self._catalog_pack["stable_sample_id"].astype(str)
        if not np.array_equal(self.target_ids, self.catalog_ids):
            self.close()
            raise ValueError("pair and parameter catalogs have different stable sample IDs")
        self.offsets = np.asarray(self._pairs_pack["solution_offsets"], dtype=np.int64)
        if len(self.offsets) != len(self.target_ids) + 1:
            self.close()
            raise ValueError("invalid ragged solution_offsets length")
        self._id_to_index = {sample_id: index for index, sample_id in enumerate(self.target_ids)}

    def close(self):
        for name in ("_pairs_pack", "_catalog_pack"):
            pack = getattr(self, name, None)
            if pack is not None:
                pack.close()
                setattr(self, name, None)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __len__(self):
        return len(self.target_ids)

    def index_for_id(self, stable_sample_id: str) -> int:
        try:
            return self._id_to_index[str(stable_sample_id)]
        except KeyError as exc:
            raise KeyError(f"unknown V5 stable sample ID: {stable_sample_id}") from exc

    @staticmethod
    def _validate_tier(max_tier: float):
        if not any(np.isclose(max_tier, tier, atol=1e-8) for tier in MultiSolutionGroundTruth.VALID_TIERS):
            raise ValueError(f"max_tier must be one of {MultiSolutionGroundTruth.VALID_TIERS}")

    def solutions_for_index(self, target_index: int, max_tier: float = 0.05):
        self._validate_tier(max_tier)
        target_index = int(target_index)
        if not 0 <= target_index < len(self):
            raise IndexError(target_index)
        start, stop = self.offsets[target_index : target_index + 2]
        local = np.arange(start, stop, dtype=np.int64)
        tiers = np.asarray(self._pairs_pack["tier"][local], dtype=np.float32)
        local = local[tiers <= float(max_tier) + 1e-7]
        candidate_indices = np.asarray(self._pairs_pack["candidate_index"][local], dtype=np.int64)
        return {
            "target_index": target_index,
            "target_id": str(self.target_ids[target_index]),
            "candidate_index": candidate_indices,
            "candidate_id": np.asarray(self._pairs_pack["candidate_id"][local]).astype(str),
            "tier": np.asarray(self._pairs_pack["tier"][local], dtype=np.float32),
            "target_window_logrmse": np.asarray(self._pairs_pack["target_window_logrmse"][local], dtype=np.float32),
            "target_window_max_log_error": np.asarray(self._pairs_pack["target_window_max_log_error"][local], dtype=np.float32),
            "active_parameter_distance": np.asarray(self._pairs_pack["active_parameter_distance"][local], dtype=np.float32),
            "branch": np.asarray(self._catalog_pack["branch"][candidate_indices]).astype(str),
            "slot_type": np.asarray(self._catalog_pack["slot_type"][candidate_indices], dtype=np.int32),
            "slot_params_phys": np.asarray(self._catalog_pack["slot_params_phys"][candidate_indices], dtype=np.float32),
            "slot_params_norm": np.asarray(self._catalog_pack["slot_params_norm"][candidate_indices], dtype=np.float32),
            "slot_param_mask": np.asarray(self._catalog_pack["slot_param_mask"][candidate_indices], dtype=np.uint8),
            "slot_weight": np.asarray(self._catalog_pack["slot_weight"][candidate_indices], dtype=np.float32),
            "global_params_phys": np.asarray(self._catalog_pack["global_params_phys"][candidate_indices], dtype=np.float32),
        }

    def target_parameters_for_index(self, target_index: int):
        """Return the original V5 generating label that anchors a solution set."""
        target_index = int(target_index)
        if not 0 <= target_index < len(self):
            raise IndexError(target_index)
        return {
            "target_index": target_index,
            "target_id": str(self.target_ids[target_index]),
            "branch": str(self._catalog_pack["branch"][target_index]),
            "slot_type": np.asarray(self._catalog_pack["slot_type"][target_index], dtype=np.int32),
            "slot_params_phys": np.asarray(self._catalog_pack["slot_params_phys"][target_index], dtype=np.float32),
            "slot_params_norm": np.asarray(self._catalog_pack["slot_params_norm"][target_index], dtype=np.float32),
            "slot_param_mask": np.asarray(self._catalog_pack["slot_param_mask"][target_index], dtype=np.uint8),
            "slot_weight": np.asarray(self._catalog_pack["slot_weight"][target_index], dtype=np.float32),
            "global_params_phys": np.asarray(self._catalog_pack["global_params_phys"][target_index], dtype=np.float32),
        }

    def solutions_for_id(self, stable_sample_id: str, max_tier: float = 0.05):
        return self.solutions_for_index(self.index_for_id(stable_sample_id), max_tier=max_tier)

    def target_parameters_for_id(self, stable_sample_id: str):
        return self.target_parameters_for_index(self.index_for_id(stable_sample_id))
