import os
from importlib import import_module
import numpy as np
import tensorflow as tf
import progressbar
import h5py
from .voxel_detect import voxel_detect
from utils.iso_boxes import iso_cube
from utils.regu_grid import regu_grid


class voxel_regre(voxel_detect):
    """ 3d offset detection based method
    """
    @staticmethod
    def get_trainer(args, new_log):
        from train.train_voxel_regre import train_voxel_regre
        return train_voxel_regre(args, new_log)

    def __init__(self, args):
        super(voxel_regre, self).__init__(args)
        self.batch_allot = getattr(
            import_module('model.batch_allot'),
            'batch_allot_vxdir'
        )

    def provider_worker(self, line, image_dir, caminfo):
        img_name, pose_raw = self.data_module.io.parse_line_annot(line)
        img = self.data_module.io.read_image(os.path.join(image_dir, img_name))
        pcnt, resce = self.data_module.ops.voxel_hit(
            img, pose_raw, caminfo.crop_size, caminfo)
        resce3 = resce[0:4]
        cube = iso_cube()
        cube.load(resce3)
        pose_pca = self.data_module.ops.raw_to_pca(pose_raw, resce3)
        vxhit = self.data_module.ops.raw_to_vxlabel(
            pose_raw, cube, self.hmap_size, caminfo
        )
        index = self.data_module.io.imagename2index(img_name)
        return (index, np.expand_dims(pcnt, axis=-1),
                pose_pca.flatten().T, vxhit, resce)

    def yanker(self, pose_local, resce):
        resce3 = resce[0:4]
        return self.data_module.ops.pca_to_raw(pose_local, resce3)

    def yanker_hmap(self, resce, voxhig, hmap_size, caminfo):
        resce3 = resce[0:4]
        cube = iso_cube()
        cube.load(resce3)
        return self.data_module.ops.vxlabel_to_raw(
            voxhig, cube, hmap_size, caminfo)

    def get_model(
            self, input_tensor, is_training, bn_decay,
            hg_repeat=1, scope=None):
        """ input_tensor: BxHxWxDxC
            out_dim: BxHxWxDxJ, where J is number of joints
        """
        end_points = {}
        self.end_point_list = []
        final_endpoint = 'hourglass_{}'.format(hg_repeat - 1)
        num_joint = self.out_dim
        num_feature = 32

        def add_and_check_final(name, net):
            end_points[name] = net
            return name == final_endpoint

        from tensorflow.contrib import slim
        from inresnet3d import inresnet3d
        # ~/anaconda2/lib/python2.7/site-packages/tensorflow/contrib/layers/
        with tf.variable_scope(
                scope, self.name_desc, [input_tensor]):
            weight_decay = 0.00004
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
                    weights_regularizer=slim.l2_regularizer(weight_decay),
                    biases_regularizer=slim.l2_regularizer(weight_decay),
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm), \
                slim.arg_scope(
                    [slim.max_pool3d, slim.avg_pool3d],
                    stride=2, padding='SAME'), \
                slim.arg_scope(
                    [slim.conv3d_transpose],
                    stride=2, padding='SAME',
                    weights_regularizer=slim.l2_regularizer(weight_decay),
                    biases_regularizer=slim.l2_regularizer(weight_decay),
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm), \
                slim.arg_scope(
                    [slim.conv3d],
                    stride=1, padding='SAME',
                    weights_regularizer=slim.l2_regularizer(weight_decay),
                    biases_regularizer=slim.l2_regularizer(weight_decay),
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm):
                with tf.variable_scope('stage64'):
                    sc = 'stage64'
                    net = slim.conv3d(input_tensor, 16, 3)
                    net = inresnet3d.conv_maxpool(net, scope=sc)
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                    sc = 'stage32_image'
                    net = inresnet3d.resnet_k(
                        net, scope='stage32_residual')
                    net = slim.conv2d(
                        net, num_feature, 1, scope='stage32_out')
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                for hg in range(hg_repeat):
                    sc = 'hourglass_{}'.format(hg)
                    with tf.variable_scope(sc):
                        branch0 = inresnet3d.hourglass3d(
                            net, 2, scope=sc + '_hg')
                        branch0 = inresnet3d.resnet_k(
                            branch0, scope='_res')
                        branch_det = slim.conv3d(
                            branch0, num_joint, 1,
                            # normalizer_fn=None, activation_fn=tf.nn.softmax)
                            normalizer_fn=None, activation_fn=None)
                        self.end_point_list.append(sc)
                        if add_and_check_final(sc, branch_det):
                            return branch_det, end_points
                        branch1 = slim.conv2d(
                            branch_det, num_feature, 1)
                        net = net + branch0 + branch1
        raise ValueError('final_endpoint (%s) not recognized', final_endpoint)
