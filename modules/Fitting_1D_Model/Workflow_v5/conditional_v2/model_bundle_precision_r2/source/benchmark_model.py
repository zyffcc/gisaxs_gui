"""Shared encoders and conditional proposals for a controlled V5 benchmark.

Mixture scores are proposal scores, not calibrated posterior probabilities.
"""
import itertools

import numpy as np
import tensorflow as tf
from tensorflow import keras


COMBOS = np.array([c + (0,) * (4-k) for k in range(1, 5)
                   for c in itertools.combinations_with_replacement((1, 2, 3), k)], dtype='int32')


class ProposalModel(keras.Model):
    def __init__(self, architecture='cnn_multi', hypotheses=6):
        super().__init__()
        self.architecture = architecture
        self.hypotheses = hypotheses if architecture == 'cnn_multi' else 1
        curve = keras.Input((256, 3))
        context = keras.Input((5,))
        if architecture == 'mlp_single':
            x = keras.layers.Flatten()(curve)
            x = keras.layers.Concatenate()([x, context])
            for width in (512, 512, 256):
                x = keras.layers.Dense(width, activation='swish')(x)
        else:
            x = keras.layers.Conv1D(48, 9, padding='same', activation='swish')(curve)
            for width in (48, 96, 128, 128):
                shortcut = keras.layers.Conv1D(width, 1, strides=2, padding='same')(x)
                z = keras.layers.Conv1D(width, 5, strides=2, padding='same')(x)
                z = keras.layers.LayerNormalization()(z)
                z = keras.layers.Activation('swish')(z)
                z = keras.layers.Conv1D(width, 5, padding='same')(z)
                x = keras.layers.Activation('swish')(keras.layers.Add()([z, shortcut]))
            # Preserve coarse peak positions rather than averaging away all q information.
            x = keras.layers.Flatten()(x)
            x = keras.layers.Concatenate()([x, context])
            x = keras.layers.Dense(256, activation='swish')(x)
        self.encoder = keras.Model([curve, context], x)
        self.classifier = keras.layers.Dense(len(COMBOS))
        self.embedding = keras.layers.Embedding(len(COMBOS), 32)
        self.decoder = keras.Sequential([
            keras.layers.Dense(256, activation='swish'),
            keras.layers.Dense(256, activation='swish'),
            keras.layers.Dense(self.hypotheses * 38)])

    def encode(self, inputs, training=False):
        return self.encoder([inputs['curve'], inputs['context']], training=training)

    def decode(self, feature, combo):
        if self.architecture != 'mlp_single':
            feature = tf.concat([feature, self.embedding(combo)], axis=-1)
        output = tf.reshape(self.decoder(feature), [-1, self.hypotheses, 38])
        return {'params': tf.sigmoid(tf.reshape(output[..., :24], [-1, self.hypotheses, 4, 6])),
                'weight_logits': output[..., 24:28],
                'globals': tf.sigmoid(output[..., 28:32]),
                'd_logits': output[..., 32:36],
                'resolution_logits': output[..., 36], 'mixture_logits': output[..., 37]}

    def call(self, inputs, training=False):
        feature = self.encode(inputs, training)
        result = self.decode(feature, tf.cast(inputs['combo'], tf.int32))
        result['class_logits'] = self.classifier(feature)
        return result


def proposal_cost(output, target):
    mask = target['param_mask'][:, None]
    p_error = tf.reduce_sum(tf.square(output['params'] - target['params'][:, None]) * mask,
                            axis=[2, 3]) / tf.maximum(tf.reduce_sum(mask, axis=[2, 3]), 1.)
    gm = target['global_mask'][:, None]
    g_error = tf.reduce_sum(tf.square(output['globals'] - target['globals'][:, None]) * gm,
                            axis=2) / tf.maximum(tf.reduce_sum(gm, axis=2), 1.)
    active = tf.cast(target['types'] > 0, tf.float32)[:, None]
    weight_logs = tf.nn.log_softmax(output['weight_logits'] + (1-active) * -1e4, axis=-1)
    w_error = -tf.reduce_sum(target['weights'][:, None] * weight_logs, axis=2)
    d_truth = tf.broadcast_to(target['d'][:, None], tf.shape(output['d_logits']))
    d_error = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=d_truth, logits=output['d_logits']) * active, axis=2)
    d_error /= tf.maximum(tf.reduce_sum(active, axis=2), 1.)
    r_truth = tf.broadcast_to(target['resolution'][:, None], tf.shape(output['resolution_logits']))
    r_error = tf.nn.sigmoid_cross_entropy_with_logits(labels=r_truth, logits=output['resolution_logits'])
    cost = 12*p_error + 4*g_error + w_error + .5*d_error + .5*r_error
    return cost, p_error, g_error


def training_loss(output, target):
    classification = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target['combo'], logits=output['class_logits'])
    cost, _, _ = proposal_cost(output, target)
    temperature = .15
    mixture = -temperature * tf.reduce_logsumexp(
        tf.nn.log_softmax(output['mixture_logits'], axis=-1) - cost / temperature, axis=-1)
    return tf.reduce_mean(classification + mixture)


def load_arrays(directory, keys=None):
    from pathlib import Path
    paths = Path(directory).glob('*.npy')
    return {p.stem: np.load(p, mmap_mode='r', allow_pickle=False) for p in paths
            if keys is None or p.stem in keys}


TRAIN_KEYS = ['curve', 'context', 'combo', 'params', 'param_mask', 'weights',
              'globals', 'global_mask', 'd', 'resolution', 'types']
