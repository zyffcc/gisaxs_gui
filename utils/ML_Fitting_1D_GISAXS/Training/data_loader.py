"""tf.data loader for TFRecord shards, with NPZ fallback for debugging."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import json

import numpy as np
import tensorflow as tf

from TrainSetBuild import schema
from TrainSetBuild.tfrecord_io import INPUT_KEYS, LABEL_KEYS, parse_example, split_inputs_labels

EXPECTED_SAMPLE_SHAPES = {
    "x": (schema.MAX_POINTS, 3),
    "point_mask": (schema.MAX_POINTS,),
    "global_features": (5,),
    "type_allowed": (schema.MAX_SLOTS, schema.NUM_TYPES),
    "param_low_norm": (schema.MAX_SLOTS, schema.NUM_TYPES, schema.P_MAX),
    "param_high_norm": (schema.MAX_SLOTS, schema.NUM_TYPES, schema.P_MAX),
    "param_range_mask": (schema.MAX_SLOTS, schema.NUM_TYPES, schema.P_MAX),
    "force_exist": (schema.MAX_SLOTS,),
    "global_low_norm": (schema.G_MAX,),
    "global_high_norm": (schema.G_MAX,),
    "global_range_mask": (schema.G_MAX,),
    "d_allowed": (schema.MAX_SLOTS, 2),
    "d_spacing_rule": (schema.NUM_D_RULES,),
    "slot_type": (schema.MAX_SLOTS,),
    "slot_exist": (schema.MAX_SLOTS,),
    "slot_params_norm": (schema.MAX_SLOTS, schema.P_MAX),
    "slot_param_mask": (schema.MAX_SLOTS, schema.P_MAX),
    "slot_weight": (schema.MAX_SLOTS,),
    "global_params_norm": (schema.G_MAX,),
    "global_param_mask": (schema.G_MAX,),
    "q": (schema.MAX_POINTS,),
    "I_clean": (schema.MAX_POINTS,),
}

MAX_SOLUTION_MODES = 16
MULTISOLUTION_SPECS = {
    "solution_mask": (tf.float32, (MAX_SOLUTION_MODES,)),
    "solution_tier": (tf.float32, (MAX_SOLUTION_MODES,)),
    "solution_logrmse": (tf.float32, (MAX_SOLUTION_MODES,)),
    "solution_label_weight": (tf.float32, (MAX_SOLUTION_MODES,)),
    "solution_slot_type": (tf.int32, (MAX_SOLUTION_MODES, schema.MAX_SLOTS)),
    "solution_slot_exist": (tf.float32, (MAX_SOLUTION_MODES, schema.MAX_SLOTS)),
    "solution_slot_params_norm": (tf.float32, (MAX_SOLUTION_MODES, schema.MAX_SLOTS, schema.P_MAX)),
    "solution_slot_param_mask": (tf.float32, (MAX_SOLUTION_MODES, schema.MAX_SLOTS, schema.P_MAX)),
    "solution_slot_weight": (tf.float32, (MAX_SOLUTION_MODES, schema.MAX_SLOTS)),
}


def parse_multisolution_example(example_proto):
    feature_spec = {"stable_sample_id": tf.io.FixedLenFeature([], tf.string)}
    feature_spec.update({key: tf.io.FixedLenFeature([], tf.string) for key in MULTISOLUTION_SPECS})
    parsed = tf.io.parse_single_example(example_proto, feature_spec)
    out = {"stable_sample_id": parsed["stable_sample_id"]}
    for key, (dtype, shape) in MULTISOLUTION_SPECS.items():
        out[key] = tf.reshape(tf.io.decode_raw(parsed[key], dtype), shape)
    return out


def list_shards(dataset_dir: str | Path, split: str):
    split_dir = Path(dataset_dir) / split
    tfrecord_shards = sorted(split_dir.glob("*.tfrecord"))
    if tfrecord_shards:
        return tfrecord_shards
    return sorted(split_dir.glob("*.npz"))


def validate_shards(shards: Iterable[Path]):
    shard_list = list(shards)
    if not shard_list:
        return
    if shard_list[0].suffix == ".tfrecord":
        for sample in tf.data.TFRecordDataset([str(shard_list[0])]).take(1).map(parse_example):
            for key, expected in EXPECTED_SAMPLE_SHAPES.items():
                actual = tuple(sample[key].shape.as_list())
                if actual != expected:
                    raise ValueError(f"TFRecord {shard_list[0]} array {key} has shape {actual}, expected {expected}")
        return

    # NPZ reading below uses the same all-ones default as TFRecord parsing.
    required = set(INPUT_KEYS + LABEL_KEYS) - {"global_param_mask"}
    for shard in shard_list:
        with np.load(shard) as data:
            missing = sorted(required.difference(data.files))
            if missing:
                raise ValueError(f"Shard {shard} is missing required arrays: {missing}")

    first = shard_list[0]
    with np.load(first) as data:
        for key, expected in EXPECTED_SAMPLE_SHAPES.items():
            if key == "global_param_mask" and key not in data.files:
                continue
            actual = data[key].shape[1:]
            if actual != expected:
                raise ValueError(f"Shard {first} array {key} has sample shape {actual}, expected {expected}")


def prepare_v5_model_input(sample):
    """Build V5 normalized inputs from the raw stored arrays at read time."""
    mask = tf.cast(sample["point_mask"], tf.bool)
    valid_i = tf.boolean_mask(tf.cast(sample["I_noisy"], tf.float32), mask)
    sorted_i = tf.sort(valid_i)
    n = tf.size(sorted_i)
    idx = tf.cast(tf.round(0.99 * tf.cast(tf.maximum(n - 1, 0), tf.float32)), tf.int32)
    p99 = tf.maximum(sorted_i[idx], tf.constant(1e-30, tf.float32))
    log_p99 = tf.math.log(p99)
    q = tf.cast(sample["q"], tf.float32)
    intensity = tf.cast(sample["I_noisy"], tf.float32)
    sigma_i = tf.cast(sample["sigma"], tf.float32)
    log_q_norm = (
        tf.math.log(tf.maximum(q, tf.constant(schema.V5_Q_MIN_GLOBAL, tf.float32)))
        - tf.math.log(tf.constant(schema.V5_Q_MIN_GLOBAL, tf.float32))
    ) / tf.math.log(tf.constant(schema.V5_Q_MAX_GLOBAL / schema.V5_Q_MIN_GLOBAL, tf.float32))
    log_i_norm = tf.math.log(tf.maximum(intensity, 1e-30)) - log_p99
    log_sigma_norm = tf.math.log(tf.maximum(sigma_i, 1e-30)) - log_p99
    x = tf.stack([log_q_norm, log_i_norm, log_sigma_norm], axis=-1)
    sample["x"] = tf.where(mask[:, tf.newaxis], x, tf.zeros_like(x))
    valid_q = tf.boolean_mask(q, mask)
    sample["global_features"] = tf.stack([
        (tf.math.log(valid_q[0]) - tf.math.log(tf.constant(schema.V5_Q_MIN_GLOBAL, tf.float32)))
        / tf.math.log(tf.constant(schema.V5_Q_MAX_GLOBAL / schema.V5_Q_MIN_GLOBAL, tf.float32)),
        (tf.math.log(valid_q[-1]) - tf.math.log(tf.constant(schema.V5_Q_MIN_GLOBAL, tf.float32)))
        / tf.math.log(tf.constant(schema.V5_Q_MAX_GLOBAL / schema.V5_Q_MIN_GLOBAL, tf.float32)),
        tf.cast(n, tf.float32) / float(schema.MAX_POINTS),
        log_p99,
        tf.constant(1.0, tf.float32),
    ])
    return sample


def _signature():
    inputs = {
        "x": tf.TensorSpec((schema.MAX_POINTS, 3), tf.float32),
        "point_mask": tf.TensorSpec((schema.MAX_POINTS,), tf.bool),
        "global_features": tf.TensorSpec((5,), tf.float32),
        "type_allowed": tf.TensorSpec((schema.MAX_SLOTS, schema.NUM_TYPES), tf.float32),
        "param_low_norm": tf.TensorSpec((schema.MAX_SLOTS, schema.NUM_TYPES, schema.P_MAX), tf.float32),
        "param_high_norm": tf.TensorSpec((schema.MAX_SLOTS, schema.NUM_TYPES, schema.P_MAX), tf.float32),
        "param_range_mask": tf.TensorSpec((schema.MAX_SLOTS, schema.NUM_TYPES, schema.P_MAX), tf.float32),
        "force_exist": tf.TensorSpec((schema.MAX_SLOTS,), tf.float32),
        "global_low_norm": tf.TensorSpec((schema.G_MAX,), tf.float32),
        "global_high_norm": tf.TensorSpec((schema.G_MAX,), tf.float32),
        "global_range_mask": tf.TensorSpec((schema.G_MAX,), tf.float32),
        "d_allowed": tf.TensorSpec((schema.MAX_SLOTS, 2), tf.float32),
        "d_spacing_rule": tf.TensorSpec((schema.NUM_D_RULES,), tf.float32),
    }
    labels = {
        "slot_type": tf.TensorSpec((schema.MAX_SLOTS,), tf.int32),
        "slot_exist": tf.TensorSpec((schema.MAX_SLOTS,), tf.float32),
        "slot_params_norm": tf.TensorSpec((schema.MAX_SLOTS, schema.P_MAX), tf.float32),
        "slot_param_mask": tf.TensorSpec((schema.MAX_SLOTS, schema.P_MAX), tf.float32),
        "slot_weight": tf.TensorSpec((schema.MAX_SLOTS,), tf.float32),
        "global_params_norm": tf.TensorSpec((schema.G_MAX,), tf.float32),
        "global_param_mask": tf.TensorSpec((schema.G_MAX,), tf.float32),
        "resolution_present": tf.TensorSpec((), tf.float32),
        "d_spacing_rule": tf.TensorSpec((schema.NUM_D_RULES,), tf.float32),
        "q": tf.TensorSpec((schema.MAX_POINTS,), tf.float32),
        "I_clean": tf.TensorSpec((schema.MAX_POINTS,), tf.float32),
        "point_mask": tf.TensorSpec((schema.MAX_POINTS,), tf.bool),
    }
    return inputs, labels


def sample_generator(shards: Iterable[Path], shuffle_samples: bool = True, seed: int = 0, max_samples: int | None = None):
    """Yield one sample at a time from NPZ shards.

    This simple generator is robust for the first version; for million-scale
    datasets it may become a bottleneck compared with shard-level batching.
    """
    rng = np.random.default_rng(seed)
    emitted = 0
    shard_list = list(shards)
    while True:
        order = np.arange(len(shard_list))
        if shuffle_samples:
            rng.shuffle(order)
        for shard_idx in order:
            with np.load(shard_list[int(shard_idx)]) as data:
                n = data["x"].shape[0]
                sample_order = np.arange(n)
                if shuffle_samples:
                    rng.shuffle(sample_order)
                for i in sample_order:
                    inputs = {k: data[k][i].astype(np.float32) if k != "point_mask" else data[k][i].astype(bool) for k in INPUT_KEYS}
                    slot_type = data["slot_type"][i].astype(np.int32)
                    slot_param_mask = data["slot_param_mask"][i].astype(np.float32).copy()
                    if "slot_params_phys" in data.files:
                        slot_params_phys = data["slot_params_phys"][i].astype(np.float32)
                        for slot in np.where(data["slot_exist"][i] > 0.5)[0]:
                            slot_param_mask[slot] *= schema.effective_param_mask(int(slot_type[slot]), slot_params_phys[slot])
                    labels = {
                        "slot_type": slot_type,
                        "slot_exist": data["slot_exist"][i].astype(np.float32),
                        "slot_params_norm": data["slot_params_norm"][i].astype(np.float32),
                        "slot_param_mask": slot_param_mask,
                        "slot_weight": data["slot_weight"][i].astype(np.float32),
                        "global_params_norm": data["global_params_norm"][i].astype(np.float32),
                        "global_param_mask": data["global_param_mask"][i].astype(np.float32) if "global_param_mask" in data.files else np.ones(schema.G_MAX, np.float32),
                        # A parameter supervision mask is not an existence label.
                        "resolution_present": np.float32(
                            data["global_params_phys"][i, 3] > 0
                            if "global_params_phys" in data.files
                            else data["global_params_norm"][i, 3] > 0
                        ),
                        "d_spacing_rule": data["d_spacing_rule"][i].astype(np.float32),
                        "q": data["q"][i].astype(np.float32),
                        "I_clean": data["I_clean"][i].astype(np.float32),
                        "point_mask": data["point_mask"][i].astype(bool),
                    }
                    yield inputs, labels
                    emitted += 1
                    if max_samples is not None and emitted >= max_samples:
                        return


def make_tfrecord_dataset(
    shards: Iterable[Path],
    batch_size: int,
    shuffle: bool = True,
    seed: int = 0,
    max_samples: int | None = None,
    drop_remainder: bool = False,
    dataset_profile: str = schema.DATASET_PROFILE_LEGACY_V3,
    multisolution_dir: str | Path | None = None,
    split: str | None = None,
    multisolution_fraction: float = 0.0,
):
    files = [str(p) for p in shards]
    if multisolution_dir is not None:
        if split is None:
            raise ValueError("split is required with multisolution_dir")
        mode_files = [
            str(Path(multisolution_dir) / split / Path(path).name.replace(".tfrecord", ".multisolution.tfrecord"))
            for path in files
        ]
        missing = [path for path in mode_files if not Path(path).exists()]
        if missing:
            raise FileNotFoundError(f"missing multi-solution sidecars: {missing[:3]}")
        ds = tf.data.Dataset.from_tensor_slices((files, mode_files))
    else:
        ds = tf.data.Dataset.from_tensor_slices(files)
    if shuffle:
        ds = ds.shuffle(len(files), seed=seed, reshuffle_each_iteration=True)
    if multisolution_dir is not None:
        ds = ds.interleave(
            lambda original, modes: tf.data.Dataset.zip(
                (tf.data.TFRecordDataset(original), tf.data.TFRecordDataset(modes))
            ),
            cycle_length=tf.data.AUTOTUNE,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=not shuffle,
        )
    else:
        ds = ds.interleave(
            tf.data.TFRecordDataset,
            cycle_length=tf.data.AUTOTUNE,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=not shuffle,
        )

    def prepare(record, mode_record=None):
        sample = parse_example(record)
        if dataset_profile == schema.DATASET_PROFILE_UNIVERSAL_V5:
            sample = prepare_v5_model_input(sample)
        inputs, labels = split_inputs_labels(sample)
        labels["resolution_present"] = tf.cast(sample["resolution_stratum"] > 0, tf.float32)
        if mode_record is not None:
            modes = parse_multisolution_example(mode_record)
            assertion = tf.debugging.assert_equal(sample["stable_sample_id"], modes["stable_sample_id"])
            with tf.control_dependencies([assertion]):
                labels.update({key: tf.identity(modes[key]) for key in MULTISOLUTION_SPECS})
        return inputs, labels

    ds = ds.map(prepare, num_parallel_calls=tf.data.AUTOTUNE)
    if multisolution_dir is not None and multisolution_fraction > 0.0:
        if max_samples is None:
            raise ValueError("max_samples is required for balanced multi-solution sampling")
        multi = ds.filter(lambda _, labels: tf.reduce_sum(labels["solution_mask"]) > 1.5).repeat()
        single = ds.filter(lambda _, labels: tf.reduce_sum(labels["solution_mask"]) <= 1.5).repeat()
        ds = tf.data.Dataset.sample_from_datasets(
            [multi, single],
            weights=[float(multisolution_fraction), 1.0 - float(multisolution_fraction)],
            seed=seed,
            stop_on_empty_dataset=False,
        ).take(max_samples)
    elif max_samples is not None:
        ds = ds.take(max_samples)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(8192, max(batch_size * 64, 1024)), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size, drop_remainder=drop_remainder).prefetch(tf.data.AUTOTUNE)


def make_dataset(
    dataset_dir: str | Path,
    split: str,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 0,
    max_samples: int | None = None,
    drop_remainder: bool = False,
    multisolution_dir: str | Path | None = None,
    multisolution_fraction: float = 0.0,
):
    shards = list_shards(dataset_dir, split)
    if not shards:
        raise FileNotFoundError(f"No {split} shards found under {Path(dataset_dir) / split}")
    validate_shards(shards)
    metadata_path = Path(dataset_dir) / "metadata.json"
    dataset_profile = schema.DATASET_PROFILE_LEGACY_V3
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            dataset_profile = json.load(handle).get("dataset_profile", dataset_profile)
    if shards[0].suffix == ".tfrecord":
        return make_tfrecord_dataset(
            shards,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            max_samples=max_samples,
            drop_remainder=drop_remainder,
            dataset_profile=dataset_profile,
            multisolution_dir=multisolution_dir,
            split=split,
            multisolution_fraction=multisolution_fraction,
        )

    ds = tf.data.Dataset.from_generator(
        lambda: sample_generator(shards, shuffle_samples=shuffle, seed=seed, max_samples=max_samples),
        output_signature=_signature(),
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=min(2048, max(batch_size * 16, 128)), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder).prefetch(tf.data.AUTOTUNE)
    return ds
