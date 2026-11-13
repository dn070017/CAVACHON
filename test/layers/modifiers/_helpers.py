import numpy as np
import tensorflow as tf

X = np.array([[0.0, 2.0, 0.5], [3.0, 0.0, 1.0]], dtype=np.float32)


def dense():
    return tf.constant(X)


def sparse():
    return tf.sparse.from_dense(tf.constant(X))


def to_np(t):
    if isinstance(t, tf.SparseTensor):
        t = tf.sparse.to_dense(t)
    return np.asarray(t)
