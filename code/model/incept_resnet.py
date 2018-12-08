""" Hand in Depth
    https://github.com/xkunwu/depth-hand
"""
import tensorflow as tf
from tensorflow.contrib import slim


class incept_resnet:
    @staticmethod
    def block32(net, num_out=None,
                scale=1.0, activation_fn=tf.nn.relu,
                scope=None, reuse=None):
        """ 32x32x32 --> 32x32x32 """
        # return slim.conv2d(net, 32, 3)
        num = int(net.shape[-1].value)
        if num_out is None:
            num_out = num
        num2 = (num_out >> 1)
        num4 = (num2 >> 1)
        with tf.variable_scope(scope, 'block32', [net], reuse=reuse):
            with tf.variable_scope('branch0'):
                tower0 = slim.conv2d(net, num4, 1, stride=1)
            with tf.variable_scope('branch1'):
                tower1 = slim.conv2d(net, num4, 1, stride=1)
                tower1 = slim.conv2d(tower1, num4, 3, stride=1)
            with tf.variable_scope('branch2'):  # equavalent to 5x5
                tower2 = slim.conv2d(net, num4, 1, stride=1)
                tower2 = slim.conv2d(tower2, num2, 3, stride=1)
                tower2 = slim.conv2d(tower2, num2, 3, stride=1)
            mixed = tf.concat(axis=-1, values=[tower0, tower1, tower2])
            mixup = slim.conv2d(
                mixed, num_out, 1,
                normalizer_fn=None, activation_fn=None,
                scope='mixup')
            if num != num_out:
                net = slim.conv2d(net, num_out, 1, stride=1)
            net += mixup * scale
        if activation_fn is not None:
            net = activation_fn(net)
        return net

    @staticmethod
    def block16(net, num_out=None,
                scale=1.0, activation_fn=tf.nn.relu,
                scope=None, reuse=None):
        """ 16x16x64 --> 16x16x64 """
        # return slim.conv2d(net, 64, 3)
        num = int(net.shape[-1].value)
        if num_out is None:
            num_out = num
        num2 = (num_out >> 1)
        num4 = (num2 >> 1)
        with tf.variable_scope(scope, 'block16', [net], reuse=reuse):
            with tf.variable_scope('branch0'):
                tower0 = slim.conv2d(net, num2, 1, stride=1)
            with tf.variable_scope('branch1'):  # equavalent to 7x7
                tower1 = slim.conv2d(net, num4, 1, stride=1)
                tower1 = slim.conv2d(tower1, num2, [1, 7], stride=1)
                tower1 = slim.conv2d(tower1, num2, [7, 1], stride=1)
            mixed = tf.concat(axis=-1, values=[tower0, tower1])
            mixup = slim.conv2d(
                mixed, num_out, 1,
                normalizer_fn=None, activation_fn=None,
                scope='mixup')
            if num != num_out:
                net = slim.conv2d(net, num_out, 1, stride=1)
            net += mixup
        if activation_fn is not None:
            net = activation_fn(net)
        return net

    @staticmethod
    def block8(net, num_out=None,
               scale=1.0, activation_fn=tf.nn.relu,
               scope=None, reuse=None):
        """ 8x8x128 --> 8x8x128 """
        # return slim.conv2d(net, 128, 3)
        num = int(net.shape[-1].value)
        if num_out is None:
            num_out = num
        num2 = (num_out >> 1)
        num4 = (num2 >> 1)
        with tf.variable_scope(scope, 'block8', [net], reuse=reuse):
            with tf.variable_scope('branch0'):
                tower0 = slim.conv2d(net, num2, 1, stride=1)
            with tf.variable_scope('branch1'):  # equavalent to 3x3
                tower1 = slim.conv2d(net, num4, 1, stride=1)
                tower1 = slim.conv2d(tower1, num2, [1, 3], stride=1)
                tower1 = slim.conv2d(tower1, num2, [3, 1], stride=1)
            mixed = tf.concat(axis=-1, values=[tower0, tower1])
            mixup = slim.conv2d(
                mixed, num_out, 1,
                normalizer_fn=None, activation_fn=None,
                scope='mixup')
            if num != num_out:
                net = slim.conv2d(net, num_out, 1, stride=1)
            net += mixup * scale
        if activation_fn is not None:
            net = activation_fn(net)
        return net

    @staticmethod
    def resnet_k(
        net, kernel_size=3, num_out=None,
        scale=1.0, activation_fn=tf.nn.relu,
            scope=None, reuse=None):
        """ general residual model """
        num = int(net.shape[-1].value)
        if num_out is None:
            num_out = num
        num2 = (num_out >> 1)
        sc_current = 'residual_{}_{}'.format(kernel_size, num2)
        with tf.variable_scope(scope, sc_current, [net], reuse=reuse):
            with tf.variable_scope('branch0'):
                tower0 = slim.conv2d(net, num2, 1, stride=1)
            with tf.variable_scope('branch1'):  # equavalent to 3x3
                tower1 = slim.conv2d(net, num2, 1, stride=1)
                tower1 = slim.conv2d(
                    tower1, num2, [1, kernel_size], stride=1)
                tower1 = slim.conv2d(
                    tower1, num2, [kernel_size, 1], stride=1)
            mixed = tf.concat(axis=-1, values=[tower0, tower1])
            mixup = slim.conv2d(
                mixed, num_out, 1, stride=1,
                normalizer_fn=None, activation_fn=None,
                scope='mixup')
            if num != num_out:
                net = slim.conv2d(net, num_out, 1, stride=1)
            net += mixup * scale
        if activation_fn is not None:
            net = activation_fn(net)
        return net

    @staticmethod
    def conv_maxpool(net, scope=None, reuse=None):
        """ simple conv + max_pool """
        num = int(net.shape[-1].value)
        sc_current = 'conv_maxpool_{}'.format(num)
        with tf.variable_scope(scope, sc_current, [net], reuse=reuse):
            net = slim.conv2d(net, 2 * num, 3, stride=1)
            net = slim.max_pool2d(net, 3, stride=2)
        return net

    @staticmethod
    def reduce_net(net, num_out=None, scope=None, reuse=None):
        """ reduce scale by one-half, while double feature size """
        # return incept_resnet.conv_maxpool(net, scope, reuse)
        num = int(net.shape[-1].value)
        if num_out is None:
            num_out = num
        num2 = (num_out >> 1)
        num4 = (num2 >> 1)
        sc_current = 'reduce_net_{}'.format(num2)
        with tf.variable_scope(scope, sc_current, [net], reuse=reuse):
            with tf.variable_scope('branch0'):
                tower0 = slim.max_pool2d(net, 3, stride=2)
            with tf.variable_scope('branch1'):
                tower1 = slim.conv2d(net, num4, 1, stride=1)
                tower1 = slim.conv2d(tower1, num2, 3, stride=2)
            with tf.variable_scope('branch2'):
                tower2 = slim.conv2d(net, num4, 1, stride=1)
                tower2 = slim.conv2d(tower2, num4, 3, stride=1)
                tower2 = slim.conv2d(tower2, num2, 3, stride=2)
            net = tf.concat(axis=-1, values=[tower0, tower1, tower2])
        return net

    @staticmethod
    def pullout8(net, out_dim, is_training,
                 scope=None, reuse=None):
        """ supposed to work best with 8x8 input """
        with tf.variable_scope(scope, 'pullout8', [net], reuse=reuse):
            net = incept_resnet.conv_maxpool(net, scope='conv_pool_8')
            print(net.shape)
            net = incept_resnet.conv_maxpool(net, scope='conv_pool_4')
            print(net.shape)
            shape2 = net.get_shape()
            fc_num = shape2[3] * 2
            net = slim.conv2d(
                net, fc_num, shape2[1:3],
                padding='VALID', scope='fullconn4')
            # net = slim.avg_pool2d(  # --> 2x2x?
            #     net, 5, stride=3, padding='VALID',
            #     scope='avgpool8_5x5_3')
            print(net.shape)
            # net = slim.conv2d(net, 64, 1, scope='reduce8')
            # print(net.shape)
            # net = slim.conv2d(  # --> 1x1x?
            #     net, 128, net.get_shape()[1:3],
            #     padding='VALID', scope='fullconn8')
            # print(net.shape)
            net = slim.flatten(net)
            print(net.shape)
            net = slim.dropout(
                net, 0.5, scope='dropout8')
            print(net.shape)
            net = slim.fully_connected(
                net, out_dim,
                activation_fn=None, normalizer_fn=None,
                scope='output8')
        return net

    @staticmethod
    def get_net(
            input_tensor, out_dim, is_training,
            bn_decay, regu_scale, end_point_list,
            block_rep=[4, 4, 4], block_scale=[1.0, 1.0, 1.0],
            scope=None, final_endpoint='stage_out'):
        """ input_tensor: BxHxWxC
            out_dim: Bx(Jx3), where J is number of joints
        """
        end_points = {}

        def add_and_check_final(name, net):
            end_points[name] = net
            return name == final_endpoint

        with tf.variable_scope(
                scope, 'incept_resnet', [input_tensor]):
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
                    net = incept_resnet.reduce_net(net, scope=sc)
                    end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                    sc = 'stage64'
                    net = incept_resnet.reduce_net(net, scope=sc)
                    end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                with tf.variable_scope('stage32'):
                    sc = 'stage32_1'
                    net = slim.repeat(
                        net, block_rep[0], incept_resnet.block32,
                        scale=block_scale[0], scope=sc)
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
                    net = slim.repeat(
                        net, block_rep[1], incept_resnet.block16,
                        scale=block_scale[1], scope=sc)
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
                    net = slim.repeat(
                        net, block_rep[2], incept_resnet.block8,
                        scale=block_scale[2], scope=sc)
                    # net = incept_resnet.block8(
                    #     net, activation_fn=None, scope=sc)
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
