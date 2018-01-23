import tensorflow as tf
from tensorflow.contrib import slim
from pyquaternion import Quaternion


class affine_net:
    # @staticmethod
    # def get_net(
    #         input_tensor, out_dim, is_training, end_point_list,
    #         scope=None, final_endpoint='stage_out'):
    #     """ input_tensor: BxHxWxC
    #         out_dim: BxJ, where J is flattened 3D locations
    #     """
    #     end_points = {}
    #
    #     def add_and_check_final(name, net):
    #         end_points[name] = net
    #         return name == final_endpoint
    #
    #     with tf.variable_scope(scope, 'incept_resnet', [input_tensor]):
    #         with slim.arg_scope(
    #                 [slim.batch_norm, slim.dropout], is_training=is_training), \
    #             slim.arg_scope(
    #                 [slim.fully_connected],
    #                 activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm), \
    #             slim.arg_scope(
    #                 [slim.max_pool2d, slim.avg_pool2d],
    #                 stride=1, padding='SAME'), \
    #             slim.arg_scope(
    #                 [slim.conv2d],
                    # stride=1, padding='SAME',
                    # activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm):
