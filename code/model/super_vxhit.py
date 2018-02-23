import os
from importlib import import_module
import numpy as np
import tensorflow as tf
from .voxel_detect import voxel_detect
from utils.iso_boxes import iso_cube


class super_vxhit(voxel_detect):
    """ this might not be a good idea: intermediate layers -> 0
    """
    @staticmethod
    def get_trainer(args, new_log):
        from train.train_super_vxhit import train_super_vxhit
        return train_super_vxhit(args, new_log)

    def __init__(self, args):
        super(super_vxhit, self).__init__(args)
        self.batch_allot = getattr(
            import_module('model.batch_allot'),
            'batch_vxhit'
        )
        self.crop_size = 64
        self.hmap_size = 32
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
            self.store_handle['vxhit'][self.batch_beg:batch_end, ...],
            axis=-1)
        self.batch_data['batch_poses'] = \
            self.store_handle['pose_c1'][self.batch_beg:batch_end, ...]
        self.batch_data['batch_vxhit'] = self.store_handle['pose_lab'][
            self.batch_beg:batch_end, ..., :self.join_num]
        self.batch_data['batch_index'] = \
            self.store_handle['index'][self.batch_beg:batch_end, ...]
        self.batch_data['batch_resce'] = \
            self.store_handle['resce'][self.batch_beg:batch_end, ...]
        self.batch_beg = batch_end
        return self.batch_data

    def receive_data(self, thedata, args):
        """ Receive parameters specific to the data """
        super(super_vxhit, self).receive_data(thedata, args)
        thedata.hmap_size = self.hmap_size
        self.out_dim = self.join_num * 3
        self.store_name = {
            'index': self.train_file,
            'poses': self.train_file,
            'resce': self.train_file,
            'pose_lab': os.path.join(
                self.prepare_dir, 'pose_lab_{}'.format(self.hmap_size)),
            'vxhit': os.path.join(
                self.prepare_dir, 'vxhit_{}'.format(self.crop_size)),
            # 'pose_c1': os.path.join(self.prepare_dir, 'pose_c1'),
            'pose_c1': os.path.join(self.prepare_dir, 'pose_c1'),
        }
        self.store_precon = {
            'index': [],
            'poses': [],
            'resce': [],
            'pose_lab': ['poses', 'resce'],
            'vxhit': ['index', 'resce'],
            # 'pose_c1': ['poses', 'resce'],
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
        poses_out = np.empty((num_elem, self.join_num * 3))
        for ei, resce, poses in zip(range(num_elem), batch_resce, batch_poses):
            pose_local = poses.reshape(-1, 3)
            pose_raw = self.yanker(pose_local, resce, self.caminfo)
            poses_out[ei] = pose_raw.reshape(1, -1)
        self.eval_pred.append(poses_out)

    def get_model(
            self, input_tensor, is_training,
            bn_decay, regu_scale,
            hg_repeat=2, scope=None):
        """ input_tensor: BxHxWxDxC
            out_dim: BxHxWxDx(J*4), where J is number of joints
        """
        end_points = {}
        self.end_point_list = []
        final_endpoint = 'stage_out'
        num_joint = self.join_num
        num_feature = 32
        num_vol = self.hmap_size * self.hmap_size * self.hmap_size

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
                    sc = 'stage32_pre'
                    net = inresnet3d.resnet_k(
                        net, scope='stage32_res')
                    net = slim.conv3d(
                        net, num_feature, 1, scope='stage32_out')
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                for hg in range(hg_repeat):  # 32x32x32
                    sc = 'hourglass_{}'.format(hg)
                    with tf.variable_scope(sc):
                        # branch0 = inresnet3d.hourglass3d(
                        #     net, 2, scope=sc + '_hg')
                        branch0 = inresnet3d.resnet_k(
                            net, scope='_res')
                        branch_det = slim.conv3d(
                            branch0, num_joint, 1,
                            # normalizer_fn=None, activation_fn=tf.nn.softmax)
                            normalizer_fn=None, activation_fn=None)
                        branch_flat = tf.reshape(
                            branch_det,
                            [-1, num_vol, num_joint])
                        self.end_point_list.append(sc)
                        if add_and_check_final(sc, branch_flat):
                            return branch_flat, end_points
                        branch1 = slim.conv3d(
                            branch_det, num_feature, 1)
                        net = net + branch0 + branch1
                with tf.variable_scope('stage32'):
                    sc = 'stage32_post'
                    net = inresnet3d.conv_maxpool(net, scope=sc)
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                with tf.variable_scope('stage16'):
                    sc = 'stage16'
                    net = inresnet3d.conv_maxpool(net, scope=sc)
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
        vxhit_tf = tf.placeholder(
            tf.int32, shape=(
                batch_size,
                # self.hmap_size, self.hmap_size, self.hmap_size,
                self.join_num))
        return frames_tf, poses_tf, vxhit_tf

    @staticmethod
    def smooth_l1(xa):
        return tf.where(
            1 < xa,
            xa - 0.5,
            0.5 * (xa ** 2)
        )

    def get_loss(self, pred, echt, vxhit, end_points):
        """ simple sum-of-squares loss
            pred: BxHxWxDxJ
            echt: BxJ
        """
        loss_l2 = tf.nn.l2_loss(pred - echt)
        loss_ce = 0
        for name, net in end_points.items():
            if not name.startswith('hourglass_'):
                continue
            echt_l = tf.unstack(vxhit, axis=-1)
            pred_l = tf.unstack(net, axis=-1)
            losses_vxhit = [
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=e, logits=p) for e, p in zip(echt_l, pred_l)]
            loss_ce += tf.reduce_mean(tf.add_n(losses_vxhit))
        loss_reg = tf.add_n(tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES))
        return loss_l2, loss_ce, loss_reg
