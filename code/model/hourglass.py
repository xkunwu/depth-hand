import tensorflow as tf
from tensorflow.contrib import slim
from model.incept_resnet import incept_resnet


class hourglass:
    @staticmethod
    def hg_net(net, n, scope=None, reuse=None):
        sc_current = 'hg_net_{}'.format(n)
        with tf.variable_scope(scope, sc_current, [net], reuse=reuse):
            upper0 = incept_resnet.residual3(net)

            lower0 = slim.max_pool2d(net, 3, stride=2)
            lower0 = incept_resnet.residual3(lower0)

            if 1 < n:
                lower1 = hourglass.hg_net(lower0, n - 1)
            else:
                lower1 = lower0

            lower2 = incept_resnet.residual3(lower1)
            upper1 = slim.conv2d_transpose(lower2, int(net.shape[-1].value), 3)
            return upper0 + upper1

    @staticmethod
    def get_model(
            input_tensor, out_dim, is_training, end_point_list,
            block_rep=[2, 2, 1], block_scale=[0.5, 0.5, 0.5],
            scope=None, final_endpoint='stage_out'):
        """ input_tensor: BxHxWxC
        """
        end_points = {}

        def add_and_check_final(name, net):
            end_points[name] = net
            return name == final_endpoint

        with tf.variable_scope(
                scope, 'hourglass_net', [input_tensor]):
            with \
                slim.arg_scope(
                    [slim.batch_norm],
                    is_training=is_training,
                    # # Make sure updates happen automatically
                    # updates_collections=None,
                    # exponential moving average is actually alpha filter in signal processing,
                    # the time to converge is approximately 1/(1-decay) steps of train.
                    # For decay=0.999, you need 1/0.001=1000 steps to converge.
                    # Lower `decay` value (recommend trying `decay`=0.9) if model experiences
                    # reasonably good training performance but poor validation and/or test performance.
                    # Try zero_debias_moving_mean=True for improved stability.
                    # zero_debias_moving_mean=True,
                    decay=0.999), \
                slim.arg_scope(
                    [slim.dropout],
                    is_training=is_training), \
                slim.arg_scope(
                    [slim.fully_connected],
                    weights_regularizer=slim.l2_regularizer(0.00004),
                    biases_regularizer=slim.l2_regularizer(0.00004),
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm), \
                slim.arg_scope(
                    [slim.max_pool2d, slim.avg_pool2d],
                    stride=1, padding='SAME'), \
                slim.arg_scope(
                    [slim.conv2d_transpose],
                    stride=2, padding='SAME',
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm), \
                slim.arg_scope(
                    [slim.conv2d],
                    stride=1, padding='SAME',
                    weights_regularizer=slim.l2_regularizer(0.00004),
                    biases_regularizer=slim.l2_regularizer(0.00004),
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm):
                    pass

if __name__ == "__main__":
    import numpy as np
    xx, yy = np.meshgrid(np.arange(4), np.arange(4)).astype(float)
    xx, yy = xx.astype(float), yy.astype(float)
    xx = np.tile(np.expand_dims(xx, axis=-1), [1, 1, 3])
    yy = np.tile(np.expand_dims(yy, axis=-1), [1, 1, 3])
