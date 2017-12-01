import tensorflow as tf
from tensorflow.contrib import slim


class incept_resnet:
    @staticmethod
    def block32(net, activation_fn=tf.nn.relu, scope=None, reuse=None):
        """ 32x32x32 --> 32x32x32 """
        # return slim.conv2d(net, 32, 3)
        with tf.variable_scope(scope, 'block32', [net], reuse=reuse):
            with tf.variable_scope('branch0'):
                tower0 = slim.conv2d(net, 8, 1, scope='conv0_1x1')
            with tf.variable_scope('branch1'):
                tower1 = slim.conv2d(net, 8, 1, scope='conv1_1x1')
                tower1 = slim.conv2d(tower1, 8, 3, scope='conv1_3x3')
            with tf.variable_scope('branch2'):
                tower2 = slim.conv2d(net, 8, 1, scope='conv2_1x1')
                tower2 = slim.conv2d(tower2, 12, 3, scope='conv2_3x3_a')
                tower2 = slim.conv2d(tower2, 16, 3, scope='conv2_3x3_b')
            mixed = tf.concat(axis=3, values=[tower0, tower1, tower2])
            mixup = slim.conv2d(mixed, net.get_shape()[3], 1, scope='mixup')
            net += mixup
        if activation_fn is not None:
            net = activation_fn(net)
        return net

    @staticmethod
    def block16(net, activation_fn=tf.nn.relu, scope=None, reuse=None):
        """ 16x16x64 --> 16x16x64 """
        # return slim.conv2d(net, 64, 3)
        with tf.variable_scope(scope, 'block16', [net], reuse=reuse):
            with tf.variable_scope('branch0'):
                tower0 = slim.conv2d(net, 32, 1, scope='conv0_1x1')
            with tf.variable_scope('branch1'):
                tower1 = slim.conv2d(net, 32, 1, scope='conv1_1x1')
                tower1 = slim.conv2d(tower1, 32, [1, 7], scope='conv1_1x7')
                tower1 = slim.conv2d(tower1, 32, [7, 1], scope='conv1_7x1')
            mixed = tf.concat(axis=3, values=[tower0, tower1])
            mixup = slim.conv2d(mixed, net.get_shape()[3], 1, scope='mixup')
            net += mixup
        if activation_fn is not None:
            net = activation_fn(net)
        return net

    @staticmethod
    def block8(net, activation_fn=tf.nn.relu, scope=None, reuse=None):
        """ 8x8x128 --> 8x8x128 """
        # return slim.conv2d(net, 128, 3)
        with tf.variable_scope(scope, 'block8', [net], reuse=reuse):
            with tf.variable_scope('branch0'):
                tower0 = slim.conv2d(net, 64, 1, scope='conv0_1x1')
            with tf.variable_scope('branch1'):
                tower1 = slim.conv2d(net, 64, 1, scope='conv1_1x1')
                tower1 = slim.conv2d(tower1, 64, [1, 3], scope='conv1_1x3')
                tower1 = slim.conv2d(tower1, 64, [3, 1], scope='conv1_3x1')
            mixed = tf.concat(axis=3, values=[tower0, tower1])
            mixup = slim.conv2d(mixed, net.get_shape()[3], 1, scope='mixup')
            net += mixup
        if activation_fn is not None:
            net = activation_fn(net)
        return net

    @staticmethod
    def reduce_net(net, scope=None, reuse=None):
        num = (int(net.shape[3]) >> 1)
        num2 = (num >> 1)
        # net = slim.conv2d(net, 4 * num, 3, stride=2)
        # net = slim.max_pool2d(net, 3)
        # return net
        with tf.variable_scope(scope, 'reduce_net{}'.format(num), [net], reuse=reuse):
            with tf.variable_scope('branch0'):
                tower0 = slim.max_pool2d(net, 3, stride=2, scope='maxpool0_3x3_2')
            with tf.variable_scope('branch1'):
                tower1 = slim.conv2d(net, num2, 1, scope='conv1_1x1_1')
                tower1 = slim.conv2d(tower1, num, 3, stride=2, scope='conv1_3x3_2')
            with tf.variable_scope('branch2'):
                tower2 = slim.conv2d(net, num2, 1, scope='conv2_1x1_1')
                tower2 = slim.conv2d(tower2, num2, 3, scope='conv2_3x3_1')
                tower2 = slim.conv2d(tower2, num, 3, stride=2, scope='conv2_3x3_2')
            net = tf.concat(axis=3, values=[tower0, tower1, tower2])
        return net

    @staticmethod
    def pullout(net, out_dim, is_training,
                activation_fn=tf.nn.relu, scope=None, reuse=None):
        with tf.variable_scope(scope, 'poolout', [net], reuse=reuse):
            sc = 'avgpool4_5x5_3'
            net = slim.avg_pool2d(
                net, 5, stride=3, padding='VALID', scope=sc)
            sc = 'reduce8'
            net = slim.conv2d(net, 256, 1, scope=sc)
            sc = 'fullconn8'
            net = slim.conv2d(
                net, 256, net.get_shape()[1:3], padding='VALID', scope=sc)
            net = slim.flatten(net)
            # sc = 'fullconn4'
            # net = slim.fully_connected(net, 1024, scope=sc)
            # end_point_list.append(sc)
            # if check_final_fn(sc, net):
            #     return net, end_points
            sc = 'dropout8'
            net = slim.dropout(
                net, 0.5, is_training=is_training, scope=sc)
            sc = 'output8'
            net = slim.fully_connected(
                net, out_dim, activation_fn=None, scope=sc)
        return net

    @staticmethod
    def get_net(
            input_tensor, out_dim, is_training, end_point_list,
            block_rep=[2, 2, 1], scope=None, final_endpoint='stage_out'):
        """ input_tensor: BxHxWxC
            out_dim: BxJ, where J is flattened 3D locations
        """
        end_points = {}

        def add_and_check_final(name, net):
            end_points[name] = net
            return name == final_endpoint

        with tf.variable_scope(scope, 'incept_resnet', [input_tensor]):
            with slim.arg_scope(
                    [slim.batch_norm, slim.dropout], is_training=is_training), \
                slim.arg_scope(
                    [slim.fully_connected],
                    activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm), \
                slim.arg_scope(
                    [slim.max_pool2d, slim.avg_pool2d],
                    stride=1, padding='SAME'), \
                slim.arg_scope(
                    [slim.conv2d],
                    stride=1, padding='SAME',
                    activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm):
                with tf.variable_scope('stage128'):
                    sc = 'stage128_1'
                    net = slim.conv2d(input_tensor, 16, 3, scope='conv128_3x3_1')
                    net = slim.conv2d(net, 16, 3, stride=2, scope='conv128_3x3_2')
                    net = slim.max_pool2d(net, 3, scope='maxpool128_3x3_1')
                    end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                    sc = 'stage128_2'
                    net = incept_resnet.reduce_net(net)
                    end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                with tf.variable_scope('stage32'):
                    sc = 'stage32_1'
                    net = slim.repeat(net, block_rep[0], incept_resnet.block32)
                    end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                    sc = 'stage32_2'
                    net = incept_resnet.reduce_net(net)
                    end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                with tf.variable_scope('stage16'):
                    sc = 'stage16_1'
                    net = slim.repeat(net, block_rep[1], incept_resnet.block16)
                    end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                    sc = 'stage16_2'
                    net = incept_resnet.reduce_net(net)
                    end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                with tf.variable_scope('stage8'):
                    sc = 'stage8'
                    net = slim.repeat(net, block_rep[2], incept_resnet.block8)
                    net = incept_resnet.block8(net, activation_fn=None)
                    end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                    sc = 'stage_out'
                    net = incept_resnet.pullout(
                        net, out_dim, is_training
                    )
                    # end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points

        raise ValueError('final_endpoint (%s) not recognized', final_endpoint)
