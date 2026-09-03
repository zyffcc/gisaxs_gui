"""TFRecord serialization/parsing for fixed-shape 1D GISAXS samples."""

from __future__ import annotations

from typing import Dict

import numpy as np
import tensorflow as tf

from TrainSetBuild import schema

SAMPLE_SPECS = {
    "x": (np.float32, tf.float32, (schema.MAX_POINTS, 3)),
    "point_mask": (np.uint8, tf.uint8, (schema.MAX_POINTS,)),
    "q": (np.float32, tf.float32, (schema.MAX_POINTS,)),
    "I_noisy": (np.float32, tf.float32, (schema.MAX_POINTS,)),
    "sigma": (np.float32, tf.float32, (schema.MAX_POINTS,)),
    "I_clean": (np.float32, tf.float32, (schema.MAX_POINTS,)),
    "global_features": (np.float32, tf.float32, (5,)),
    "slot_type": (np.int32, tf.int32, (schema.MAX_SLOTS,)),
    "slot_exist": (np.float32, tf.float32, (schema.MAX_SLOTS,)),
    "slot_params_phys": (np.float32, tf.float32, (schema.MAX_SLOTS, schema.P_MAX)),
    "slot_params_norm": (np.float32, tf.float32, (schema.MAX_SLOTS, schema.P_MAX)),
    "slot_param_mask": (np.float32, tf.float32, (schema.MAX_SLOTS, schema.P_MAX)),
    "slot_weight": (np.float32, tf.float32, (schema.MAX_SLOTS,)),
    "global_params_phys": (np.float32, tf.float32, (schema.G_MAX,)),
    "global_params_norm": (np.float32, tf.float32, (schema.G_MAX,)),
    "type_allowed": (np.float32, tf.float32, (schema.MAX_SLOTS, schema.NUM_TYPES)),
    "param_low_norm": (np.float32, tf.float32, (schema.MAX_SLOTS, schema.NUM_TYPES, schema.P_MAX)),
    "param_high_norm": (np.float32, tf.float32, (schema.MAX_SLOTS, schema.NUM_TYPES, schema.P_MAX)),
    "param_range_mask": (np.float32, tf.float32, (schema.MAX_SLOTS, schema.NUM_TYPES, schema.P_MAX)),
    "force_exist": (np.float32, tf.float32, (schema.MAX_SLOTS,)),
    "global_low_norm": (np.float32, tf.float32, (schema.G_MAX,)),
    "global_high_norm": (np.float32, tf.float32, (schema.G_MAX,)),
    "global_range_mask": (np.float32, tf.float32, (schema.G_MAX,)),
    "d_allowed": (np.float32, tf.float32, (schema.MAX_SLOTS, 2)),
    "d_spacing_rule": (np.float32, tf.float32, (schema.NUM_D_RULES,)),
}

OPTIONAL_INT_FEATURES = {
    "sampling_mode": (np.int32, tf.int32, ()),
    "schema_version": (np.int32, tf.int32, ()),
    "target_qmin_bin": (np.int32, tf.int32, ()),
    "target_w_class": (np.int32, tf.int32, ()),
    "grid_type": (np.int32, tf.int32, ()),
    "d_pattern": (np.int32, tf.int32, ()),
    "constraint_mode": (np.int32, tf.int32, ()),
    "generation_attempts": (np.int32, tf.int32, ()),
    "floor_points_removed": (np.int32, tf.int32, ()),
    "resolution_stratum": (np.int32, tf.int32, ()),
    "bad_point_count": (np.int32, tf.int32, ()),
    "gap_point_count": (np.int32, tf.int32, ()),
    "q_range_class": (np.int32, tf.int32, ()),
    "particle_dynamic_stratum": (np.int32, tf.int32, ()),
    "noise_count_scale_clip_code": (np.int32, tf.int32, ()),
}

OPTIONAL_FLOAT_FEATURES = {
    "rho_BG", "rho_Res", "rho_Res_actual", "count_scale",
    "relative_noise_scale", "particle_median", "particle_low_max",
    "resolution_low_max", "p99_noisy",
    "particle_log10_drop",
    "noise_count_scale_unclipped", "noise_target_particle_tail_snr",
    "noise_achieved_particle_tail_snr", "noise_particle_reference",
    "noise_particle_tail", "noise_clean_tail", "noise_edge_fraction",
}

OPTIONAL_STRING_FEATURES = {"stable_sample_id"}

OPTIONAL_ARRAY_FEATURES = {
    "q_window_pre": (np.float32, tf.float32, (4,)),
    "q_window_final": (np.float32, tf.float32, (4,)),
    "global_param_mask": (np.float32, tf.float32, (schema.G_MAX,)),
}

INPUT_KEYS = [
    "x",
    "point_mask",
    "global_features",
    "type_allowed",
    "param_low_norm",
    "param_high_norm",
    "param_range_mask",
    "force_exist",
    "global_low_norm",
    "global_high_norm",
    "global_range_mask",
    "d_allowed",
    "d_spacing_rule",
]

LABEL_KEYS = [
    "slot_type",
    "slot_exist",
    "slot_params_norm",
    "slot_param_mask",
    "slot_weight",
    "global_params_norm",
    "global_param_mask",
    "d_spacing_rule",
    "q",
    "I_clean",
    "point_mask",
]


def _bytes_feature(value: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value: int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))


def _float_feature(value: float) -> tf.train.Feature:
    return tf.train.Feature(float_list=tf.train.FloatList(value=[float(value)]))


def serialize_sample(sample: Dict[str, np.ndarray]) -> bytes:
    features = {}
    for key, (np_dtype, _, expected_shape) in SAMPLE_SPECS.items():
        arr = np.asarray(sample[key])
        if key == "point_mask":
            arr = arr.astype(np.uint8)
        else:
            arr = arr.astype(np_dtype)
        if arr.shape != expected_shape:
            raise ValueError(f"{key} has shape {arr.shape}, expected {expected_shape}")
        features[key] = _bytes_feature(np.ascontiguousarray(arr).tobytes())
    for key in OPTIONAL_INT_FEATURES:
        if key in sample:
            features[key] = _int64_feature(int(np.asarray(sample[key]).item()))
    for key in OPTIONAL_FLOAT_FEATURES:
        if key in sample:
            features[key] = _float_feature(float(np.asarray(sample[key]).item()))
    for key in OPTIONAL_STRING_FEATURES:
        if key in sample:
            value = sample[key]
            if isinstance(value, np.ndarray):
                value = value.item()
            if isinstance(value, str):
                value = value.encode("utf-8")
            features[key] = _bytes_feature(bytes(value))
    for key, (np_dtype, _, expected_shape) in OPTIONAL_ARRAY_FEATURES.items():
        if key in sample:
            arr = np.asarray(sample[key], dtype=np_dtype)
            if arr.shape != expected_shape:
                raise ValueError(f"{key} has shape {arr.shape}, expected {expected_shape}")
            features[key] = _bytes_feature(np.ascontiguousarray(arr).tobytes())
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example.SerializeToString()


def parse_example(example_proto):
    feature_spec = {key: tf.io.FixedLenFeature([], tf.string) for key in SAMPLE_SPECS}
    # Backward-compatible defaults for datasets built before relational D constraints.
    feature_spec["d_allowed"] = tf.io.FixedLenFeature(
        [], tf.string, default_value=np.ones((schema.MAX_SLOTS, 2), dtype=np.float32).tobytes()
    )
    feature_spec["d_spacing_rule"] = tf.io.FixedLenFeature(
        [], tf.string, default_value=np.eye(schema.NUM_D_RULES, dtype=np.float32)[schema.D_RULE_FREE].tobytes()
    )
    defaults = {
        "sampling_mode": -1, "schema_version": schema.SCHEMA_VERSION_LEGACY_V3,
        "target_qmin_bin": -1, "target_w_class": -1, "grid_type": -1,
        "d_pattern": -1, "constraint_mode": -1, "generation_attempts": 0,
        "floor_points_removed": 0,
        "resolution_stratum": -1, "bad_point_count": 0, "gap_point_count": 0,
        "q_range_class": -1, "particle_dynamic_stratum": -1,
        "noise_count_scale_clip_code": 0,
    }
    for key in OPTIONAL_INT_FEATURES:
        feature_spec[key] = tf.io.FixedLenFeature([], tf.int64, default_value=defaults[key])
    for key in OPTIONAL_FLOAT_FEATURES:
        feature_spec[key] = tf.io.FixedLenFeature([], tf.float32, default_value=float("nan"))
    for key in OPTIONAL_STRING_FEATURES:
        feature_spec[key] = tf.io.FixedLenFeature([], tf.string, default_value=b"")
    for key, (np_dtype, _, shape) in OPTIONAL_ARRAY_FEATURES.items():
        default = np.ones(shape, dtype=np_dtype) if key == "global_param_mask" else np.zeros(shape, dtype=np_dtype)
        feature_spec[key] = tf.io.FixedLenFeature([], tf.string, default_value=default.tobytes())
    parsed = tf.io.parse_single_example(example_proto, feature_spec)
    out = {}
    for key, (_, tf_dtype, shape) in SAMPLE_SPECS.items():
        value = tf.io.decode_raw(parsed[key], tf_dtype)
        value = tf.reshape(value, shape)
        if key == "point_mask":
            value = tf.cast(value > 0, tf.bool)
        out[key] = value
    params_phys = out["slot_params_phys"]
    param_mask = out["slot_param_mask"]
    optional_d_active = tf.cast(
        tf.logical_and(params_phys[:, 4] > 0.0, params_phys[:, 5] > 0.0),
        tf.float32,
    )
    optional_mask = tf.stack([optional_d_active, optional_d_active], axis=-1)
    out["slot_param_mask"] = tf.concat([param_mask[:, :4], param_mask[:, 4:6] * optional_mask], axis=-1)
    for key, (_, tf_dtype, shape) in OPTIONAL_ARRAY_FEATURES.items():
        out[key] = tf.reshape(tf.io.decode_raw(parsed[key], tf_dtype), shape)
    for key in OPTIONAL_INT_FEATURES:
        out[key] = tf.cast(parsed[key], tf.int32)
    for key in OPTIONAL_FLOAT_FEATURES | OPTIONAL_STRING_FEATURES:
        out[key] = parsed[key]
    return out


def split_inputs_labels(sample):
    inputs = {key: sample[key] for key in INPUT_KEYS}
    labels = {key: sample[key] for key in LABEL_KEYS}
    return inputs, labels
