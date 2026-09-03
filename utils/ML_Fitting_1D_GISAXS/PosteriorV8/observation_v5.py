"""Deterministic V5 observation/data-view construction.

Acquisition and uncertainty-availability choices are made from the clean
recipe seed and view index before the exact physical curve is inspected.  The
result keeps encoder uncertainty separate from acceptance evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Sequence

import numpy as np

from .compatibility_calibration import (
    ACQUISITION_POLICY_ID_VERSION,
    acquisition_policy_payload,
    make_acquisition_policy_id,
)
from .clean_recipe_forward_v5 import (
    V5CleanRecipeLike,
    evaluate_v5_clean_recipe_forward,
    validate_v5_clean_recipe_like,
)
from .preprocessing import (
    DEFAULT_CONTRACT,
    PreprocessedCurve,
    preprocess_curve,
)
from .simulation import (
    NOISE_APPLICATION_VERSION,
    OBSERVATION_SEED_DERIVATION,
    OBSERVATION_STRATUM_VERSION,
    ObservationView,
    apply_observation_noise,
    sample_observation_view,
)
from .uncertainty_provenance_v5 import (
    V5UncertaintyProvenance,
    acceptance_sigma_log,
    prepare_encoder_sigma,
)


V5_OBSERVATION_DATA_VIEW_SCHEMA = "gisaxs.posterior_v8.observation_data_view/v1"
V5_OBSERVATION_DATA_VIEW_VERSION = "posterior_v8_seeded_acquisition_uncertainty_preprocessing_v1"
V5_UNCERTAINTY_VIEW_POLICY_VERSION = (
    "posterior_v8_recipe_seed_view_index_alternating_sigma_availability_v1"
)
V5_ENCODER_PROXY_RELATIVE_SIGMA = 0.015
V5_TRAINING_UNCERTAINTY_KINDS = (
    "simulated_sigma",
    "encoder_proxy_missing_sigma",
)
_UNCERTAINTY_POLICY_NAMESPACE = 0x554E4354
_UINT64_MAX = (1 << 64) - 1


def _uint64(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if not 0 <= result <= _UINT64_MAX:
        raise ValueError(f"{name} must fit in uint64")
    return result


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _nonempty_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    result = value.strip()
    if len(result) > 512:
        raise ValueError(f"{name} is too long")
    return result


def _readonly_float_vector(value, name: str, size: int) -> np.ndarray:
    try:
        result = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError(f"{name} must contain finite positive values")
    result.setflags(write=False)
    return result


def sample_v5_uncertainty_provenance(
    recipe_seed: int,
    view_index: int,
) -> V5UncertaintyProvenance:
    """Choose sigma-present/missing policy without reading physical truth."""

    seed = _uint64(recipe_seed, "recipe_seed")
    index = _uint64(view_index, "view_index")
    state = np.random.SeedSequence([seed, _UNCERTAINTY_POLICY_NAMESPACE]).generate_state(
        1, dtype=np.uint32
    )
    kind = V5_TRAINING_UNCERTAINTY_KINDS[(int(state[0]) + index) % 2]
    if kind == "simulated_sigma":
        return V5UncertaintyProvenance(kind)
    return V5UncertaintyProvenance(
        kind,
        encoder_relative_sigma_proxy=V5_ENCODER_PROXY_RELATIVE_SIGMA,
    )


def v5_sigma_log_source(provenance: V5UncertaintyProvenance) -> str:
    """Return the canonical full-policy identity for one uncertainty source."""

    if not isinstance(provenance, V5UncertaintyProvenance):
        raise TypeError("provenance must be a V5UncertaintyProvenance")
    if provenance.kind == "simulated_sigma":
        detail = "acceptance=measurement_sigma/I"
    else:
        proxy = format(float(provenance.encoder_relative_sigma_proxy), ".12g")
        detail = f"proxy={proxy};acceptance=none"
    return "|".join(
        (
            NOISE_APPLICATION_VERSION,
            V5_UNCERTAINTY_VIEW_POLICY_VERSION,
            provenance.version,
            provenance.kind,
            detail,
        )
    )


def v5_acquisition_policy_id(
    view: ObservationView,
    provenance: V5UncertaintyProvenance,
) -> str:
    """Encode one observation view with its complete uncertainty provenance."""

    if not isinstance(view, ObservationView):
        raise TypeError("view must be an ObservationView")
    if not isinstance(provenance, V5UncertaintyProvenance):
        raise TypeError("provenance must be a V5UncertaintyProvenance")
    q_window_id = f"{OBSERVATION_STRATUM_VERSION}:q-window-{view.q_window_id}"
    noise_id = f"{OBSERVATION_STRATUM_VERSION}:noise-{view.noise_id}"
    return make_acquisition_policy_id(
        grid={
            "kind": view.grid.kind,
            "design_point_count": view.grid.n_points,
            "q_min": view.grid.q_min,
            "q_max": view.grid.q_max,
            "q_window_id": q_window_id,
        },
        mask={
            "mask_id": f"{view.version}:mask-{view.mask_id}",
            "point_keep_probability": view.point_keep_probability,
        },
        crop={
            "crop_id": f"{view.version}:crop-{view.crop_id}",
            "q_min": view.preprocess_q_range[0],
            "q_max": view.preprocess_q_range[1],
        },
        view={
            "observation_view_version": view.version,
            "view_index": view.view_index,
            "observation_seed_derivation": OBSERVATION_SEED_DERIVATION,
        },
        sigma={
            "noise_id": noise_id,
            "poisson_count_scale": view.noise.poisson_count_scale,
            "relative_sigma": view.noise.relative_sigma,
            "sigma_floor_fraction": view.noise.sigma_floor_fraction,
            "sigma_log_source": v5_sigma_log_source(provenance),
        },
    )


@dataclass(frozen=True)
class V5ObservationDataView:
    """One clean recipe rendered as one independently seeded training view.

    ``acceptance_sigma_log`` is aligned with ``preprocessed.valid_arrays()``.
    It is deliberately ``None`` for an encoder-only uncertainty proxy.
    """

    clean_recipe_sha256: str
    recipe_seed: int
    view_index: int
    split_id: str
    observation: ObservationView
    uncertainty: V5UncertaintyProvenance
    q: np.ndarray
    clean_intensity: np.ndarray
    intensity: np.ndarray
    measurement_sigma: np.ndarray | None
    encoder_sigma: np.ndarray
    selection_mask: np.ndarray
    preprocessed: PreprocessedCurve
    acceptance_sigma_log: np.ndarray | None
    uncertainty_provenance: np.ndarray
    design_point_count: int
    effective_valid_point_count: int
    acquisition_policy_id: str
    schema_version: str = V5_OBSERVATION_DATA_VIEW_SCHEMA
    version: str = V5_OBSERVATION_DATA_VIEW_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != V5_OBSERVATION_DATA_VIEW_SCHEMA:
            raise ValueError("unsupported V5 observation data-view schema")
        if self.version != V5_OBSERVATION_DATA_VIEW_VERSION:
            raise ValueError("unsupported V5 observation data-view version")
        recipe_seed = _uint64(self.recipe_seed, "recipe_seed")
        view_index = _uint64(self.view_index, "view_index")
        split_id = _nonempty_text(self.split_id, "split_id")
        digest = self.clean_recipe_sha256
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(value not in "0123456789abcdef" for value in digest)
        ):
            raise ValueError("clean_recipe_sha256 must be a lowercase SHA-256 digest")
        if not isinstance(self.observation, ObservationView):
            raise TypeError("observation must be an ObservationView")
        expected_observation = sample_observation_view(
            recipe_seed,
            view_index,
            max_points=DEFAULT_CONTRACT.max_points,
        )
        if self.observation != expected_observation:
            raise ValueError("observation must derive only from recipe_seed and view_index")
        if not isinstance(self.uncertainty, V5UncertaintyProvenance):
            raise TypeError("uncertainty must be V5UncertaintyProvenance")
        expected_uncertainty = sample_v5_uncertainty_provenance(recipe_seed, view_index)
        if self.uncertainty != expected_uncertainty:
            raise ValueError("uncertainty policy must derive only from recipe_seed and view_index")

        design_n = _positive_integer(self.design_point_count, "design_point_count")
        effective_n = _positive_integer(
            self.effective_valid_point_count,
            "effective_valid_point_count",
        )
        if design_n != self.observation.grid.n_points:
            raise ValueError("design_point_count disagrees with the acquisition grid")
        if effective_n > design_n:
            raise ValueError("effective_valid_point_count cannot exceed design_point_count")
        q = _readonly_float_vector(self.q, "q", design_n)
        if not np.array_equal(q, self.observation.grid.values()):
            raise ValueError("q does not match the deterministic observation grid")
        clean = _readonly_float_vector(
            self.clean_intensity,
            "clean_intensity",
            design_n,
        )
        intensity = _readonly_float_vector(self.intensity, "intensity", design_n)
        encoder_sigma = _readonly_float_vector(
            self.encoder_sigma,
            "encoder_sigma",
            design_n,
        )
        raw_mask = np.asarray(self.selection_mask)
        if raw_mask.shape != (design_n,) or raw_mask.dtype.kind != "b":
            raise ValueError("selection_mask must be a boolean acquisition-length vector")
        selection_mask = np.array(raw_mask, dtype=np.bool_, copy=True)
        selection_mask.setflags(write=False)
        if not np.array_equal(selection_mask, self.observation.selection_mask(q)):
            raise ValueError("selection_mask disagrees with deterministic mask provenance")

        expected_intensity, generated_sigma = apply_observation_noise(
            clean,
            self.observation.noise,
            observation_seed=self.observation.observation_seed,
        )
        if not np.array_equal(intensity, expected_intensity):
            raise ValueError("intensity disagrees with deterministic observation noise")
        if self.uncertainty.measurement_sigma_available:
            measurement_sigma = _readonly_float_vector(
                self.measurement_sigma,
                "measurement_sigma",
                design_n,
            )
            if not np.array_equal(measurement_sigma, generated_sigma):
                raise ValueError("measurement_sigma disagrees with simulated uncertainty")
        else:
            if self.measurement_sigma is not None:
                raise ValueError("missing-sigma views cannot expose simulated sigma as evidence")
            measurement_sigma = None
        expected_encoder_sigma = prepare_encoder_sigma(
            intensity,
            measurement_sigma,
            self.uncertainty,
        )
        if not np.array_equal(encoder_sigma, expected_encoder_sigma):
            raise ValueError("encoder_sigma disagrees with uncertainty provenance")

        if not isinstance(self.preprocessed, PreprocessedCurve):
            raise TypeError("preprocessed must be a PreprocessedCurve")
        expected_preprocessed = preprocess_curve(
            q,
            intensity,
            encoder_sigma,
            mask=selection_mask,
            q_range=self.observation.preprocess_q_range,
            contract=DEFAULT_CONTRACT,
        )
        for name in (
            "x",
            "point_mask",
            "global_features",
            "q",
            "intensity",
            "sigma",
            "source_indices",
        ):
            if not np.array_equal(
                getattr(self.preprocessed, name),
                getattr(expected_preprocessed, name),
            ):
                raise ValueError(f"preprocessed.{name} disagrees with the V5 view policy")
        if dict(self.preprocessed.stats) != dict(expected_preprocessed.stats):
            raise ValueError("preprocessed.stats disagrees with the V5 view policy")
        if dict(self.preprocessed.stats["contract"]) != asdict(DEFAULT_CONTRACT):
            raise ValueError("preprocessed curve does not use the frozen V5 contract")
        expected_effective_n = int(expected_preprocessed.stats["valid_before_downsampling"])
        if effective_n != expected_effective_n:
            raise ValueError("effective_valid_point_count disagrees with preprocessing")

        raw_acceptance = acceptance_sigma_log(
            intensity,
            measurement_sigma,
            self.uncertainty,
        )
        if raw_acceptance is None:
            if self.acceptance_sigma_log is not None:
                raise ValueError("encoder-only sigma proxy cannot become acceptance evidence")
            acceptance = None
        else:
            source_indices = expected_preprocessed.source_indices[expected_preprocessed.point_mask]
            expected_acceptance = raw_acceptance[source_indices]
            acceptance = _readonly_float_vector(
                self.acceptance_sigma_log,
                "acceptance_sigma_log",
                expected_preprocessed.valid_count,
            )
            if not np.array_equal(acceptance, expected_acceptance):
                raise ValueError("acceptance_sigma_log is not aligned with valid points")

        expected_feature = np.asarray(self.uncertainty.feature_vector, dtype=np.float32)
        feature = np.asarray(self.uncertainty_provenance)
        if feature.shape != (3,) or feature.dtype != np.float32:
            raise ValueError("uncertainty_provenance must be a float32 vector with shape (3,)")
        if not np.array_equal(feature, expected_feature):
            raise ValueError("uncertainty_provenance disagrees with its source kind")
        feature = np.array(feature, dtype=np.float32, copy=True)
        feature.setflags(write=False)

        expected_policy_id = v5_acquisition_policy_id(self.observation, self.uncertainty)
        if self.acquisition_policy_id != expected_policy_id:
            raise ValueError("acquisition_policy_id is incomplete or inconsistent")
        acquisition_policy_payload(self.acquisition_policy_id)

        object.__setattr__(self, "recipe_seed", recipe_seed)
        object.__setattr__(self, "view_index", view_index)
        object.__setattr__(self, "split_id", split_id)
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "clean_intensity", clean)
        object.__setattr__(self, "intensity", intensity)
        object.__setattr__(self, "measurement_sigma", measurement_sigma)
        object.__setattr__(self, "encoder_sigma", encoder_sigma)
        object.__setattr__(self, "selection_mask", selection_mask)
        object.__setattr__(self, "acceptance_sigma_log", acceptance)
        object.__setattr__(self, "uncertainty_provenance", feature)
        object.__setattr__(self, "design_point_count", design_n)
        object.__setattr__(self, "effective_valid_point_count", effective_n)

    def encoder_curve_inputs(self, *, add_batch_axis: bool = False) -> dict[str, np.ndarray]:
        """Return the four curve-side tensors consumed by the V5 model."""

        values = self.preprocessed.model_inputs(add_batch_axis=add_batch_axis)
        provenance = self.uncertainty_provenance
        if add_batch_axis:
            provenance = provenance[np.newaxis, ...]
        return {**values, "uncertainty_provenance": provenance}

    def audit_payload(self) -> dict[str, object]:
        """Return compact provenance without duplicating raw curve arrays."""

        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "clean_recipe_sha256": self.clean_recipe_sha256,
            "recipe_seed": self.recipe_seed,
            "view_index": self.view_index,
            "observation_seed": self.observation.observation_seed,
            "split_id": self.split_id,
            "split_assignment_unit": "clean_recipe_all_views_inherit",
            "design_point_count": self.design_point_count,
            "design_point_count_semantics": "pre_mask_pre_crop_acquisition_grid_n",
            "effective_valid_point_count": self.effective_valid_point_count,
            "effective_valid_point_count_semantics": (
                "finite_selected_points_before_any_encoder_downsampling"
            ),
            "encoder_valid_point_count": self.preprocessed.valid_count,
            "acquisition_policy_id_version": ACQUISITION_POLICY_ID_VERSION,
            "acquisition_policy_id": self.acquisition_policy_id,
            "uncertainty_view_policy_version": V5_UNCERTAINTY_VIEW_POLICY_VERSION,
            "uncertainty_provenance": self.uncertainty.audit_payload(),
            "uncertainty_provenance_feature": self.uncertainty_provenance.tolist(),
            "measurement_sigma_available": self.measurement_sigma is not None,
            "acceptance_sigma_log_available": self.acceptance_sigma_log is not None,
            "preprocessing_contract": dict(self.preprocessed.stats["contract"]),
        }


def build_v5_observation_data_view(
    recipe: V5CleanRecipeLike,
    view_index: int,
    *,
    split_id: str,
) -> V5ObservationDataView:
    """Render one exact clean recipe through the frozen V5 view policy."""

    recipe = validate_v5_clean_recipe_like(recipe)
    index = _uint64(view_index, "view_index")
    assigned_split = _nonempty_text(split_id, "split_id")

    # These choices intentionally precede and cannot inspect the physical curve.
    observation = sample_observation_view(
        recipe.recipe_seed,
        index,
        max_points=DEFAULT_CONTRACT.max_points,
    )
    uncertainty = sample_v5_uncertainty_provenance(recipe.recipe_seed, index)

    q = observation.grid.values()
    clean = evaluate_v5_clean_recipe_forward(recipe, q)
    intensity, simulated_sigma = apply_observation_noise(
        clean,
        observation.noise,
        observation_seed=observation.observation_seed,
    )
    measurement_sigma = simulated_sigma if uncertainty.measurement_sigma_available else None
    encoder_sigma = prepare_encoder_sigma(intensity, measurement_sigma, uncertainty)
    selection_mask = observation.selection_mask(q)
    preprocessed = preprocess_curve(
        q,
        intensity,
        encoder_sigma,
        mask=selection_mask,
        q_range=observation.preprocess_q_range,
        contract=DEFAULT_CONTRACT,
    )
    raw_acceptance = acceptance_sigma_log(
        intensity,
        measurement_sigma,
        uncertainty,
    )
    if raw_acceptance is None:
        valid_acceptance = None
    else:
        valid_indices = preprocessed.source_indices[preprocessed.point_mask]
        valid_acceptance = raw_acceptance[valid_indices]
        valid_acceptance.setflags(write=False)
    return V5ObservationDataView(
        clean_recipe_sha256=recipe.sha256,
        recipe_seed=recipe.recipe_seed,
        view_index=index,
        split_id=assigned_split,
        observation=observation,
        uncertainty=uncertainty,
        q=q,
        clean_intensity=clean,
        intensity=intensity,
        measurement_sigma=measurement_sigma,
        encoder_sigma=encoder_sigma,
        selection_mask=selection_mask,
        preprocessed=preprocessed,
        acceptance_sigma_log=valid_acceptance,
        uncertainty_provenance=np.asarray(uncertainty.feature_vector, dtype=np.float32),
        design_point_count=observation.grid.n_points,
        effective_valid_point_count=int(preprocessed.stats["valid_before_downsampling"]),
        acquisition_policy_id=v5_acquisition_policy_id(observation, uncertainty),
    )


def build_v5_observation_data_views(
    recipe: V5CleanRecipeLike,
    view_indices: Sequence[int],
    *,
    split_id: str,
) -> tuple[V5ObservationDataView, ...]:
    """Render distinct views while inheriting one clean-recipe split ID."""

    try:
        indices = tuple(_uint64(value, "view_index") for value in view_indices)
    except TypeError as exc:
        if isinstance(view_indices, (str, bytes)):
            raise TypeError("view_indices must be a sequence of integers") from exc
        raise
    if not indices:
        raise ValueError("view_indices cannot be empty")
    if len(indices) != len(set(indices)):
        raise ValueError("view_indices must be unique within one clean recipe")
    return tuple(
        build_v5_observation_data_view(recipe, index, split_id=split_id) for index in indices
    )


__all__ = [
    "V5_ENCODER_PROXY_RELATIVE_SIGMA",
    "V5_OBSERVATION_DATA_VIEW_SCHEMA",
    "V5_OBSERVATION_DATA_VIEW_VERSION",
    "V5_TRAINING_UNCERTAINTY_KINDS",
    "V5_UNCERTAINTY_VIEW_POLICY_VERSION",
    "V5ObservationDataView",
    "build_v5_observation_data_view",
    "build_v5_observation_data_views",
    "sample_v5_uncertainty_provenance",
    "v5_acquisition_policy_id",
    "v5_sigma_log_source",
]
