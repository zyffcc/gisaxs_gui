"""Same V5 Adam refinement, with reusable TensorFlow graphs and explicit checkpoints."""
import time
import numpy as np
import tensorflow as tf
import physics_v5


def project_tf(p):
    radius = tf.exp(p[..., 0]*tf.math.log(100.))
    minimum = tf.math.log(2*radius*1.001/3.)/tf.math.log(500./3.)
    d = tf.clip_by_value(tf.maximum(p[..., 4], minimum), 0., 1.)
    return tf.concat([p[..., :4], d[..., None], p[..., 5:]], axis=-1)


def forward(q, mask, types, p, w, g, d, resolution, straight_through=False):
    return physics_v5.reconstruct_intensity(
        q, types, tf.cast(types > 0, tf.float32), project_tf(p), w,
        tf.pad(g, [[0, 0], [0, 1]]), d, resolution, point_mask=mask > .5,
        hard_presence=straight_through)


class FastRefiner:
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        b = batch_size
        def var(shape, dtype=tf.float32):
            return tf.Variable(tf.zeros(shape, dtype), trainable=False)
        self.q, self.mask, self.observed = [var((b, 1000)) for _ in range(3)]
        self.types = var((b, 4), tf.int32)
        self.d, self.res = var((b, 4)), var((b,))
        self.pv = tf.Variable(tf.zeros((b, 4, 6)))
        self.wv = tf.Variable(tf.zeros((b, 4)))
        self.gv = tf.Variable(tf.zeros((b, 4)))
        self.variables = [self.pv, self.wv, self.gv]
        self.optimizer = tf.keras.optimizers.Adam(.015, clipnorm=10.)
        self.optimizer.build(self.variables)
        self.best_loss = var((b,))
        self.best_curve = var((b, 1000))
        self.best_p = var((b, 4, 6))
        self.best_w, self.best_g = var((b, 4)), var((b, 4))

    @tf.function
    def step(self):
        with tf.GradientTape() as tape:
            p, g = project_tf(tf.sigmoid(self.pv)), tf.sigmoid(self.gv)
            curve = forward(self.q, self.mask, self.types, p, self.wv, g, self.d, self.res)
            error = tf.math.log(curve)-tf.math.log(tf.maximum(self.observed, 1e-30))
            losses = tf.reduce_sum(error**2*self.mask, axis=1)/tf.reduce_sum(self.mask, axis=1)
            total = tf.reduce_sum(losses)
        better = losses < self.best_loss
        self.best_loss.assign(tf.minimum(losses, self.best_loss))
        self.best_curve.assign(tf.where(better[:, None], curve, self.best_curve))
        self.best_p.assign(tf.where(better[:, None, None], p, self.best_p))
        self.best_w.assign(tf.where(better[:, None], self.wv, self.best_w))
        self.best_g.assign(tf.where(better[:, None], g, self.best_g))
        grads = tape.gradient(total, self.variables)
        for grad in grads:
            tf.debugging.assert_all_finite(grad, 'Nonfinite refinement gradient')
        self.optimizer.apply_gradients(zip(grads, self.variables))
        return total

    @tf.function
    def advance(self, steps):
        for _ in tf.range(steps):
            self.step()
        return self.best_loss

    def run(self, q, mask, observed, types, p, w, g, d, resolution, checkpoints=(0, 10, 30, 160)):
        n = len(q)
        assert 0 < n <= self.batch_size
        assert list(checkpoints) == sorted(set(checkpoints)) and checkpoints[0] == 0
        def padded(a):
            return np.concatenate([a, np.repeat(a[-1:], self.batch_size-n, axis=0)])
        def logit(a):
            a = np.clip(a, 1e-5, 1-1e-5)
            return np.log(a/(1-a))
        for dest, source in [(self.q, q), (self.mask, mask), (self.observed, observed),
                (self.types, types), (self.pv, logit(p)), (self.wv, w), (self.gv, logit(g)),
                (self.d, np.where(d > 0, 30., -30.)), (self.res, (resolution > 0).astype('float32'))]:
            dest.assign(padded(np.asarray(source)))
        for state in self.optimizer.variables():
            state.assign(tf.zeros_like(state))
        self.best_loss.assign(tf.fill([self.batch_size], np.inf))
        result, previous = {}, 0
        start = time.monotonic()
        # Observe state 0 while preparing the first update, just as the v1 implementation.
        self.step()
        for count in checkpoints:
            if count > previous:
                self.advance(tf.constant(count-previous))
            values = {k: v.numpy()[:n] for k, v in [('curve', self.best_curve), ('params', self.best_p),
                ('weights', self.best_w), ('globals', self.best_g), ('loss', self.best_loss)]}
            values['seconds'] = time.monotonic()-start
            result[count] = values
            previous = count
        return result
