import tensorflow as tf
from tensorflow.contrib import slim
from model.incept_resnet import incept_resnet


class hourglass:
    @staticmethod
    def hg_net(net, n, scope=None, reuse=None):
        num = int(net.shape[-1].value)
        sc_current = 'hg_net_{}'.format(n)
        with tf.variable_scope(scope, sc_current, [net], reuse=reuse):
            upper0 = incept_resnet.resnet_k(net)

            lower0 = slim.max_pool2d(net, 3, stride=2)
            lower0 = incept_resnet.resnet_k(lower0)

            lower0 = slim.conv2d(lower0, num * 2, 1, stride=1)

            if 1 < n:
                lower1 = hourglass.hg_net(lower0, n - 1)
            else:
                lower1 = lower0

            lower1 = slim.conv2d(lower1, num, 1, stride=1)

            lower2 = incept_resnet.resnet_k(lower1)
            upper1 = slim.conv2d_transpose(
                lower2, int(net.shape[-1].value), 3, stride=2)
            return upper0 + upper1

    @staticmethod
    def get_net(
            input_tensor, out_dim, is_training, bn_decay, end_point_list,
            hg_repeat=8, scope=None, final_endpoint='stage_out'):
        """ input_tensor: BxHxWxC
        """
        end_points = {}

        def add_and_check_final(name, net):
            end_points[name] = net
            return name == final_endpoint

        with tf.variable_scope(
                scope, 'hourglass_net', [input_tensor]):
            regu_scale = 0.00004
            bn_epsilon = 0.001
            with \
                slim.arg_scope(
                    [slim.batch_norm],
                    is_training=is_training,
                    epsilon=bn_epsilon,
                    # # Make sure updates happen automatically
                    # updates_collections=None,
                    # Try zero_debias_moving_mean=True for improved stability.
                    # zero_debias_moving_mean=True,
                    decay=bn_decay), \
                slim.arg_scope(
                    [slim.dropout],
                    is_training=is_training), \
                slim.arg_scope(
                    [slim.fully_connected],
                    weights_regularizer=slim.l2_regularizer(regu_scale),
                    biases_regularizer=slim.l2_regularizer(regu_scale),
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm), \
                slim.arg_scope(
                    [slim.max_pool2d, slim.avg_pool2d],
                    stride=2, padding='SAME'), \
                slim.arg_scope(
                    [slim.conv2d_transpose],
                    stride=2, padding='SAME',
                    weights_regularizer=slim.l2_regularizer(regu_scale),
                    biases_regularizer=slim.l2_regularizer(regu_scale),
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm), \
                slim.arg_scope(
                    [slim.conv2d],
                    stride=1, padding='SAME',
                    weights_regularizer=slim.l2_regularizer(regu_scale),
                    biases_regularizer=slim.l2_regularizer(regu_scale),
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm):
                with tf.variable_scope('stage128'):
                    sc = 'stage128'
                    net = slim.conv2d(input_tensor, 8, 3)
                    net = incept_resnet.conv_maxpool(net, scope=sc)
                    end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                    sc = 'stage64'
                    net = incept_resnet.reduce_net(net, scope=sc)
                    end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                for hg in range(hg_repeat):
                    sc = 'hourglass_{}'.format(hg)
                    with tf.variable_scope(sc):
                        net = hourglass.hg_net(
                            net, 4, scope=sc + '_hg')
                        net = incept_resnet.resnet_k(
                            net, scope=sc + '_res')
                        end_point_list.append(sc)
                        if add_and_check_final(sc, net):
                            return net, end_points
                with tf.variable_scope('stage32'):
                    sc = 'stage32_1'
                    net = incept_resnet.block32(net, scope=sc)
                    end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                    sc = 'stage32_2'
                    net = incept_resnet.reduce_net(net, scope=sc)
                    end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                with tf.variable_scope('stage16'):
                    sc = 'stage16_1'
                    net = incept_resnet.block16(net, scope=sc)
                    end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                    sc = 'stage16_2'
                    net = incept_resnet.reduce_net(net, scope=sc)
                    end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                with tf.variable_scope('stage8'):
                    sc = 'stage8'
                    net = incept_resnet.block8(net, scope=sc)
                    end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                    sc = 'stage_out'
                    net = incept_resnet.pullout8(
                        net, out_dim, is_training,
                        scope=sc
                    )
                    end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points

        raise ValueError('final_endpoint (%s) not recognized', final_endpoint)

if __name__ == "__main__":
    import numpy as np
    xx, yy = np.meshgrid(np.arange(4), np.arange(4)).astype(float)
    xx, yy = xx.astype(float), yy.astype(float)
    xx = np.tile(np.expand_dims(xx, axis=-1), [1, 1, 3])
    yy = np.tile(np.expand_dims(yy, axis=-1), [1, 1, 3])
