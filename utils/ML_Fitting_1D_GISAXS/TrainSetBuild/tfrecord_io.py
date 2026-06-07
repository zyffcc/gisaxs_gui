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
}

OPTIONAL_INT_FEATURES = {
    "sampling_mode": (np.int32, tf.int32, ()),
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
]

LABEL_KEYS = [
    "slot_type",
    "slot_exist",
    "slot_params_norm",
    "slot_param_mask",
    "slot_weight",
    "global_params_norm",
]


def _bytes_feature(value: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value: int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))


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
    if "sampling_mode" in sample:
        features["sampling_mode"] = _int64_feature(int(np.asarray(sample["sampling_mode"]).item()))
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example.SerializeToString()


def parse_example(example_proto):
    feature_spec = {key: tf.io.FixedLenFeature([], tf.string) for key in SAMPLE_SPECS}
    feature_spec["sampling_mode"] = tf.io.FixedLenFeature([], tf.int64, default_value=-1)
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
    out["sampling_mode"] = tf.cast(parsed["sampling_mode"], tf.int32)
    return out


def split_inputs_labels(sample):
    inputs = {key: sample[key] for key in INPUT_KEYS}
    labels = {key: sample[key] for key in LABEL_KEYS}
    return inputs, labels
