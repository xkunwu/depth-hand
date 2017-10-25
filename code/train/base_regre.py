import tensorflow as tf
import os
import sys
from importlib import import_module

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
tf_util = import_module('tf_util')


class base_regre():
    """ This class holds baseline training approach using plain regression.
    """
    @staticmethod
    def tweak_args(args):
        """ Tweak algorithm specific parameters """
        args.crop_resize = 96
        args.batch_size = 96

    @staticmethod
    def receive_args(args):
        """ Receive parameters specific to the data """
        args.pose_dim = args.join_num * 3

    @staticmethod
    def placeholder_inputs(batch_size, img_size, joint_num):
        num_out = joint_num * 3
        batch_frame = tf.placeholder(
            tf.float32, shape=(batch_size, img_size, img_size))
        pose_out = tf.placeholder(
            tf.float32, shape=(batch_size, num_out))
        return batch_frame, pose_out

    @staticmethod
    def get_model(batch_frame, pose_dim, is_training, bn_decay=None):
        """ directly predict all joints' location using regression
            batch_frame: BxHxW
            pose_dim: BxJ, where J is flattened 3D locations
        """
        batch_size = batch_frame.get_shape()[0].value
        end_points = {}
        input_image = tf.expand_dims(batch_frame, -1)

        # Point functions (MLP implemented as conv2d)
        net = tf_util.conv2d(
            input_image, 64, [3, 3],
            padding='VALID', stride=[1, 1],
            bn=True, is_training=is_training,
            scope='conv1', bn_decay=bn_decay)
        net = tf_util.max_pool2d(
            net, [1, 1],
            padding='VALID', scope='maxpool1')
        net = tf_util.conv2d(
            net, 256, [3, 3],
            padding='VALID', stride=[1, 1],
            bn=True, is_training=is_training,
            scope='conv2', bn_decay=bn_decay)
        net = tf_util.max_pool2d(
            net, [1, 1],
            padding='VALID', scope='maxpool2')
        net = tf_util.conv2d(
            net, 1024, [3, 3],
            padding='VALID', stride=[1, 1],
            bn=True, is_training=is_training,
            scope='conv3', bn_decay=bn_decay)
        net = tf_util.max_pool2d(
            net, [1, 1],
            padding='VALID', scope='maxpool3')

        # MLP on global point cloud vector
        net = tf.reshape(net, [batch_size, -1])
        net = tf_util.fully_connected(
            net, 512, bn=True, is_training=is_training,
            scope='fc1', bn_decay=bn_decay)
        net = tf_util.dropout(
            net, keep_prob=0.5, is_training=is_training,
            scope='dp1')
        net = tf_util.fully_connected(
            net, pose_dim, activation_fn=None, scope='fc2')

        return net, end_points

    @staticmethod
    def get_loss(pred, anno, end_points):
        """ simple sum-of-squares loss
            pred: BxJ
            anno: BxJ
        """
        # loss = tf.reduce_sum(tf.pow(tf.subtract(pred, anno), 2)) / 2
        # loss = tf.nn.l2_loss(pred - anno)  # already divided by 2
        loss = tf.reduce_mean(tf.squared_difference(pred, anno)) / 2
        return loss
