"""TensorFlow index-join adapter for compact V5.1 grouped datasets."""

from __future__ import annotations

import numpy as np

from .candidate_supervision_v5 import CANDIDATE_SUPERVISION_TENSOR_KEYS
from .grouped_amplitude_join_v5 import observation_amplitude_embeddings
from .model_v5_contract import MODEL_V5_INPUT_KEYS


def as_tensorflow_dataset(
    grouped,
    *,
    batch_size: int,
    include_unverified: bool = True,
    shuffle: bool = False,
    seed: int = 0,
):
    """Join by integer gather inside ``tf.data`` without persisting copies."""

    import tensorflow as tf

    if isinstance(batch_size, (bool, np.bool_)) or int(batch_size) != batch_size:
        raise TypeError("batch_size must be an integer")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    observed, candidates = grouped.join_indices(include_unverified=include_unverified)
    arrays = grouped.arrays
    observation_tensors = {
        name: tf.convert_to_tensor(value)
        for name, value in arrays.items()
        if name.startswith("observation__input__")
    }
    candidate_tensors = {
        name: tf.convert_to_tensor(value)
        for name, value in arrays.items()
        if name.startswith("candidate_context__input__")
    }
    label_tensors = {
        name: tf.convert_to_tensor(arrays[f"candidate_label__{name}"])
        for name in CANDIDATE_SUPERVISION_TENSOR_KEYS
    }
    amplitude_by_observation = observation_amplitude_embeddings(
        query_json=arrays["clean__amplitude_query_canonical_json"],
        query_sha256=arrays["clean__amplitude_query_sha256"],
        observation_intensity_reference=arrays["observation__intensity_reference"],
        observation_recipe_index=arrays["observation__recipe_index"],
    )
    amplitude_tensor = tf.convert_to_tensor(amplitude_by_observation)
    dataset = tf.data.Dataset.from_tensor_slices((observed, candidates))
    if shuffle:
        dataset = dataset.shuffle(len(observed), seed=int(seed), reshuffle_each_iteration=True)

    def gather(observation_index, candidate_index):
        inputs = {}
        for name in MODEL_V5_INPUT_KEYS:
            if name == "amplitude_bounds_embedding":
                inputs[name] = tf.gather(amplitude_tensor, observation_index)
                continue
            observation_name = f"observation__input__{name}"
            candidate_name = f"candidate_context__input__{name}"
            if observation_name in observation_tensors:
                inputs[name] = tf.gather(observation_tensors[observation_name], observation_index)
            else:
                inputs[name] = tf.gather(candidate_tensors[candidate_name], candidate_index)
        labels = {name: tf.gather(value, candidate_index) for name, value in label_tensors.items()}
        return inputs, labels

    return dataset.map(gather).batch(int(batch_size))


__all__ = ["as_tensorflow_dataset"]
