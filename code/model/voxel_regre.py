import os
from importlib import import_module
import numpy as np
import tensorflow as tf
import progressbar
import h5py
from .voxel_offset import voxel_offset
from utils.iso_boxes import iso_cube
from utils.regu_grid import regu_grid


class voxel_regre(voxel_offset):
    """ 3d regression with offset supervisation
    """
    @staticmethod
    def get_trainer(args, new_log):
        from train.train_voxel_regre import train_voxel_regre
        return train_voxel_regre(args, new_log)

    def __init__(self, args):
        super(voxel_regre, self).__init__(args)
        self.crop_size = 64
        self.hmap_size = 16
        self.map_scale = self.crop_size / self.hmap_size

    def fetch_batch(self, fetch_size=None):
        if fetch_size is None:
            fetch_size = self.batch_size
        batch_end = self.batch_beg + fetch_size
        # if batch_end >= self.store_size:
        #     self.batch_beg = batch_end
        #     batch_end = self.batch_beg + fetch_size
        #     self.split_end -= self.store_size
        # # print(self.batch_beg, batch_end, self.split_end)
        if batch_end >= self.split_end:
            return None
        self.batch_data['batch_frame'] = np.expand_dims(
            self.store_handle['pcnt3'][self.batch_beg:batch_end, ...],
            axis=-1)
        self.batch_data['batch_poses'] = \
            self.store_handle['pose_c1'][self.batch_beg:batch_end, ...]
        # self.batch_data['batch_poses'] = \
        #     self.store_handle['pose_c1'][self.batch_beg:batch_end, ...]
        self.batch_data['batch_vxudir'] = \
            self.store_handle['vxudir'][self.batch_beg:batch_end, ...]
        self.batch_data['batch_index'] = \
            self.store_handle['index'][self.batch_beg:batch_end, ...]
        self.batch_data['batch_resce'] = \
            self.store_handle['resce'][self.batch_beg:batch_end, ...]
        self.batch_beg = batch_end
        return self.batch_data

    def receive_data(self, thedata, args):
        """ Receive parameters specific to the data """
        super(voxel_offset, self).receive_data(thedata, args)
        thedata.hmap_size = self.hmap_size
        self.out_dim = self.join_num * 3
        self.store_name = {
            'index': self.train_file,
            'poses': self.train_file,
            'resce': self.train_file,
            'pcnt3': os.path.join(
                self.prepare_dir, 'pcnt3_{}'.format(self.crop_size)),
            'vxudir': os.path.join(
                self.prepare_dir, 'vxudir_{}'.format(self.hmap_size)),
            'pose_c1': os.path.join(self.prepare_dir, 'pose_c1'),
        }
        self.store_precon = {
            'index': [],
            'poses': [],
            'resce': [],
            'pcnt3': ['index', 'resce'],
            'vxudir': ['pcnt3', 'poses', 'resce'],
            'pose_c1': ['poses', 'resce'],
        }

    def yanker(self, pose_local, resce, caminfo):
        cube = iso_cube()
        cube.load(resce)
        # return cube.transform_add_center(pose_local)
        return cube.transform_expand_move(pose_local)

    def evaluate_batch(self, pred_val):
        batch_index = self.batch_data['batch_index']
        batch_resce = self.batch_data['batch_resce']
        batch_poses = pred_val
        num_elem = batch_index.shape[0]
        poses_out = np.empty(batch_poses.shape)
        for ei, resce, poses in zip(range(num_elem), batch_resce, batch_poses):
            pose_local = poses.reshape(-1, 3)
            pose_raw = self.yanker(pose_local, resce, self.caminfo)
            poses_out[ei] = pose_raw.flatten()
        self.eval_pred.append(poses_out)

    def get_model(
            self, input_tensor, is_training,
            bn_decay, regu_scale,
            hg_repeat=2, scope=None):
        """ input_tensor: BxHxWxDxC
            out_dim: BxHxWxDx(J*5), where J is number of joints
        """
        end_points = {}
        self.end_point_list = []
        final_endpoint = 'stage_out'
        num_joint = self.join_num
        num_feature = 128

        def add_and_check_final(name, net):
            end_points[name] = net
            return name == final_endpoint

        from tensorflow.contrib import slim
        from inresnet3d import inresnet3d
        # ~/anaconda2/lib/python2.7/site-packages/tensorflow/contrib/layers/
        with tf.variable_scope(
                scope, self.name_desc, [input_tensor]):
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
                    [slim.max_pool3d, slim.avg_pool3d],
                    stride=2, padding='SAME'), \
                slim.arg_scope(
                    [slim.conv3d_transpose],
                    stride=2, padding='SAME',
                    weights_regularizer=slim.l2_regularizer(regu_scale),
                    biases_regularizer=slim.l2_regularizer(regu_scale),
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm), \
                slim.arg_scope(
                    [slim.conv3d],
                    stride=1, padding='SAME',
                    weights_regularizer=slim.l2_regularizer(regu_scale),
                    biases_regularizer=slim.l2_regularizer(regu_scale),
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm):
                with tf.variable_scope('stage64'):
                    sc = 'stage64'
                    net = slim.conv3d(input_tensor, 8, 3)
                    net = inresnet3d.conv_maxpool(net, scope=sc)
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                    sc = 'stage32'
                    # net = inresnet3d.resnet_k(
                    #     net, scope='stage32_res')
                    net = inresnet3d.conv_maxpool(net, scope=sc)
                    net = slim.conv3d(
                        net, num_feature, 1, scope='stage32_out')
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                for hg in range(hg_repeat):
                    sc = 'hourglass_{}'.format(hg)
                    with tf.variable_scope(sc):
                        # branch0 = inresnet3d.hourglass3d(
                        #     net, 2, scope=sc + '_hg')
                        branch0 = inresnet3d.resnet_k(
                            net, scope='_res')
                        branch_olm = slim.conv3d(
                            branch0, num_joint, 1,
                            # normalizer_fn=None, activation_fn=tf.nn.relu)
                            normalizer_fn=None, activation_fn=None)
                        branch_uom = slim.conv3d(
                            branch0, num_joint * 3, 1,
                            # normalizer_fn=None, activation_fn=tf.nn.sigmoid)
                            normalizer_fn=None, activation_fn=None)
                        net_maps = tf.concat(
                            [branch_olm, branch_uom],
                            axis=-1)
                        self.end_point_list.append(sc)
                        if add_and_check_final(sc, net_maps):
                            return net_maps, end_points
                        branch1 = slim.conv3d(
                            net_maps, num_feature, 1)
                        net = net + branch0 + branch1
                with tf.variable_scope('stage16'):
                    sc = 'stage16'
                    net = slim.max_pool3d(net, 3, scope=sc)
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                with tf.variable_scope('stage8'):
                    sc = 'stage_out'
                    net = inresnet3d.pullout8(
                        net, self.out_dim, is_training,
                        scope=sc)
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
        raise ValueError('final_endpoint (%s) not recognized', final_endpoint)

    def placeholder_inputs(self, batch_size=None):
        frames_tf = tf.placeholder(
            tf.float32, shape=(
                batch_size,
                self.crop_size, self.crop_size, self.crop_size,
                1))
        poses_tf = tf.placeholder(
            tf.float32, shape=(
                batch_size,
                self.out_dim))
        vxudir_tf = tf.placeholder(
            tf.float32, shape=(
                batch_size,
                self.hmap_size, self.hmap_size, self.hmap_size,
                self.join_num * 4))
        return frames_tf, poses_tf, vxudir_tf

    @staticmethod
    def smooth_l1(xa):
        return tf.where(
            1 < xa,
            xa - 0.5,
            0.5 * (xa ** 2)
        )

    def get_loss(self, pred, echt, vxudir, end_points):
        """ simple sum-of-squares loss
            pred: BxHxWxDxJ
            echt: BxJ
        """
        num_j = self.join_num
        loss_l2 = tf.nn.l2_loss(pred - echt)
        loss_udir = 0
        loss_unit = 0
        for name, net in end_points.items():
            if not name.startswith('hourglass_'):
                continue
            # loss_udir += tf.nn.l2_loss(net - vxudir)
            loss_udir += tf.reduce_mean(
                self.smooth_l1(tf.abs(net - vxudir)))
            vxunit_pred = tf.reshape(
                net[..., num_j:],
                (-1, 3))
            loss_unit += tf.reduce_mean(
                self.smooth_l1(tf.abs(
                    1 - tf.reduce_sum(vxunit_pred ** 2, axis=-1))))
        loss_reg = tf.add_n(tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES))
        return loss_l2, loss_udir, loss_unit, loss_reg
