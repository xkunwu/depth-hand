""" Hand in Depth
    https://github.com/xkunwu/depth-hand
"""
import tensorflow as tf


def unravel_index(indices, shape):
    indices = tf.expand_dims(indices, 0)
    shape = tf.expand_dims(shape, 1)
    shape = tf.cast(shape, tf.float32)
    strides = tf.cumprod(shape, reverse=True)
    strides_shifted = tf.cumprod(shape, exclusive=True, reverse=True)
    strides = tf.cast(strides, tf.int32)
    strides_shifted = tf.cast(strides_shifted, tf.int32)

    def even():
        rem = indices - (indices // strides) * strides
        return rem // strides_shifted

    def odd():
        div = indices // strides_shifted
        return div - (div // strides) * strides
    rank = tf.rank(shape)
    return tf.cond(tf.equal(rank - (rank // 2) * 2, 0), even, odd)
