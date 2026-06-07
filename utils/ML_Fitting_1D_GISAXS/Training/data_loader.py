"""tf.data loader for TFRecord shards, with NPZ fallback for debugging."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

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
    "slot_type": (schema.MAX_SLOTS,),
    "slot_exist": (schema.MAX_SLOTS,),
    "slot_params_norm": (schema.MAX_SLOTS, schema.P_MAX),
    "slot_param_mask": (schema.MAX_SLOTS, schema.P_MAX),
    "slot_weight": (schema.MAX_SLOTS,),
    "global_params_norm": (schema.G_MAX,),
}


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

    required = set(INPUT_KEYS + LABEL_KEYS)
    for shard in shard_list:
        with np.load(shard) as data:
            missing = sorted(required.difference(data.files))
            if missing:
                raise ValueError(f"Shard {shard} is missing required arrays: {missing}")

    first = shard_list[0]
    with np.load(first) as data:
        for key, expected in EXPECTED_SAMPLE_SHAPES.items():
            actual = data[key].shape[1:]
            if actual != expected:
                raise ValueError(f"Shard {first} array {key} has sample shape {actual}, expected {expected}")


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
    }
    labels = {
        "slot_type": tf.TensorSpec((schema.MAX_SLOTS,), tf.int32),
        "slot_exist": tf.TensorSpec((schema.MAX_SLOTS,), tf.float32),
        "slot_params_norm": tf.TensorSpec((schema.MAX_SLOTS, schema.P_MAX), tf.float32),
        "slot_param_mask": tf.TensorSpec((schema.MAX_SLOTS, schema.P_MAX), tf.float32),
        "slot_weight": tf.TensorSpec((schema.MAX_SLOTS,), tf.float32),
        "global_params_norm": tf.TensorSpec((schema.G_MAX,), tf.float32),
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
):
    files = [str(p) for p in shards]
    ds = tf.data.Dataset.from_tensor_slices(files)
    if shuffle:
        ds = ds.shuffle(len(files), seed=seed, reshuffle_each_iteration=True)
    ds = ds.interleave(
        tf.data.TFRecordDataset,
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not shuffle,
    )
    ds = ds.map(lambda record: split_inputs_labels(parse_example(record)), num_parallel_calls=tf.data.AUTOTUNE)
    if max_samples is not None:
        ds = ds.take(max_samples)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(8192, max(batch_size * 64, 1024)), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)


def make_dataset(
    dataset_dir: str | Path,
    split: str,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 0,
    max_samples: int | None = None,
):
    shards = list_shards(dataset_dir, split)
    if not shards:
        raise FileNotFoundError(f"No {split} shards found under {Path(dataset_dir) / split}")
    validate_shards(shards)
    if shards[0].suffix == ".tfrecord":
        return make_tfrecord_dataset(shards, batch_size=batch_size, shuffle=shuffle, seed=seed, max_samples=max_samples)

    ds = tf.data.Dataset.from_generator(
        lambda: sample_generator(shards, shuffle_samples=shuffle, seed=seed, max_samples=max_samples),
        output_signature=_signature(),
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=min(2048, max(batch_size * 16, 128)), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds
