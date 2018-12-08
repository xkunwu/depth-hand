""" Hand in Depth
    https://github.com/xkunwu/depth-hand
"""
import os
from importlib import import_module
import numpy as np
from collections import namedtuple
import tensorflow as tf
import matplotlib.pyplot as mpplot
from cv2 import resize as cv2resize
from model.base_clean import base_clean
from utils.iso_boxes import iso_cube


class dense_regre(base_clean):
    @staticmethod
    def get_trainer(args, new_log):
        from train.train_dense_regre import train_dense_regre
        return train_dense_regre(args, new_log)

    def __init__(self, args):
        super(dense_regre, self).__init__(args)
        self.batch_allot = getattr(
            import_module('model.batch_allot'),
            'batch_udir2'
        )
        # self.num_appen = 4
        self.crop_size = 128
        self.hmap_size = 32
        self.map_scale = self.crop_size / self.hmap_size

    def fetch_batch(self, mode='train', fetch_size=None):
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
        store_handle = self.store_handle[mode]
        self.batch_data['batch_frame'] = np.expand_dims(
            store_handle['clean'][self.batch_beg:batch_end, ...],
            axis=-1)
        hmap2 = store_handle['hmap2'][self.batch_beg:batch_end, ...]
        udir2 = store_handle['udir2'][self.batch_beg:batch_end, ...]
        batch_hmaps = np.empty((fetch_size, self.hmap_size, self.hmap_size, self.join_num * 5))
        batch_hmaps[..., :self.join_num] = hmap2
        batch_hmaps[..., self.join_num:] = udir2
        self.batch_data['batch_poses'] = batch_hmaps
        self.batch_data['batch_index'] = \
            store_handle['index'][self.batch_beg:batch_end, ...]
        self.batch_data['batch_resce'] = \
            store_handle['resce'][self.batch_beg:batch_end, ...]
        self.batch_beg = batch_end
        return self.batch_data

    def receive_data(self, thedata, args):
        """ Receive parameters specific to the data """
        super(dense_regre, self).receive_data(thedata, args)
        thedata.hmap_size = self.hmap_size
        self.out_dim = self.join_num * 5
        self.store_name = {
            'index': thedata.annotation,
            'poses': thedata.annotation,
            'resce': thedata.annotation,
            'clean': 'clean_{}'.format(self.crop_size),
            'hmap2': 'hmap2_{}'.format(self.hmap_size),
            'udir2': 'udir2_{}'.format(self.hmap_size),
        }
        self.frame_type = 'clean'

    def yanker_hmap(self, resce, hmap2, olmap, uomap, depth, caminfo):
        cube = iso_cube()
        cube.load(resce)
        # return self.data_module.ops.offset_to_raw(  # often sum weight to 0
        #     hmap2, olmap, uomap, depth, cube, caminfo)
        return self.data_module.ops.udir2_to_raw(
            olmap, uomap, depth, cube, caminfo)

    def evaluate_batch(self, pred_val):
        batch_index = self.batch_data['batch_index']
        batch_resce = self.batch_data['batch_resce']
        batch_frame = self.batch_data['batch_frame']
        num_j = self.join_num
        num_elem = batch_index.shape[0]
        poses_out = np.empty((num_elem, self.join_num * 3))
        for ei in range(num_elem):
            resce = batch_resce[ei, :]
            depth = np.squeeze(batch_frame[ei, ...])
            hmap2 = pred_val[ei, ..., 0 * num_j:1 * num_j]
            olmap = pred_val[ei, ..., 1 * num_j:2 * num_j]
            uomap = pred_val[ei, ..., 2 * num_j:]
            pose_raw = self.yanker_hmap(
                resce, hmap2, olmap, uomap, depth, self.caminfo)
            poses_out[ei] = pose_raw.reshape(1, -1)
        self.eval_pred.append(poses_out)

    # def write_pred(self, fanno, caminfo,
    #                batch_index, batch_resce,
    #                batch_frame, batch_poses):
    #     num_j = self.join_num
    #     for ii in range(batch_index.shape[0]):
    #         img_name = self.args.data_io.index2imagename(batch_index[ii, 0])
    #         resce = batch_resce[ii, :]
    #         depth = np.squeeze(batch_frame[ii, ...])
    #         hmap2 = batch_poses[ii, ..., 0 * num_j:1 * num_j]
    #         olmap = batch_poses[ii, ..., 1 * num_j:2 * num_j]
    #         uomap = batch_poses[ii, ..., 2 * num_j:]
    #         pose_raw = self.yanker_hmap(
    #             resce, hmap2, olmap, uomap, depth,
    #             self.hmap_size, caminfo)
    #         fanno.write(
    #             img_name +
    #             '\t' + '\t'.join("%.4f" % x for x in pose_raw.flatten()) +
    #             '\n')

    # def fetch_batch(self, fetch_size=None):
    #     if fetch_size is None:
    #         fetch_size = self.batch_size
    #     batch_end = self.batch_beg + fetch_size
    #     if batch_end >= self.store_size:
    #         self.batch_beg = batch_end
    #         batch_end = self.batch_beg + fetch_size
    #         self.split_end -= self.store_size
    #     # print(self.batch_beg, batch_end, self.split_end)
    #     if batch_end >= self.split_end:
    #         return None
    #     # self.batch_data = {
    #     #     'batch_index': self.store_file['index'][self.batch_beg:batch_end, ...],
    #     #     'batch_frame': self.store_file['frame'][self.batch_beg:batch_end, ...],
    #     #     'batch_poses': self.store_file['poses'][self.batch_beg:batch_end, ...],
    #     #     'batch_hmap2': self.store_file['hmap2'][self.batch_beg:batch_end, ...],
    #     #     'batch_olmap': self.store_file['olmap'][self.batch_beg:batch_end, ...],
    #     #     'batch_uomap': self.store_file['uomap'][self.batch_beg:batch_end, ...],
    #     #     'batch_resce': self.store_file['resce'][self.batch_beg:batch_end, ...]
    #     # }
    #     batch_hmaps = []
    #     batch_hmaps.append(self.store_file['hmap2'][self.batch_beg:batch_end, ...])
    #     batch_hmaps.append(self.store_file['olmap'][self.batch_beg:batch_end, ...])
    #     batch_hmaps.append(self.store_file['uomap'][self.batch_beg:batch_end, ...])
    #     self.batch_data = {
    #         'batch_index': self.store_file['index'][self.batch_beg:batch_end, ...],
    #         'batch_frame': self.store_file['frame'][self.batch_beg:batch_end, ...],
    #         'batch_poses': np.concatenate(batch_hmaps, axis=-1),
    #         'batch_resce': self.store_file['resce'][self.batch_beg:batch_end, ...]
    #     }
    #     self.batch_beg = batch_end
    #     return self.batch_data

    def draw_random(self, thedata, args):
        # mode = 'train'
        mode = 'test'
        store_handle = self.store_handle[mode]
        index_h5 = store_handle['index']
        store_size = index_h5.shape[0]
        frame_id = np.random.choice(store_size)
        # frame_id = 0  # frame_id = img_id - 1
        frame_id = 239
        img_id = index_h5[frame_id, ...]
        frame_h5 = store_handle['clean'][frame_id, ...]
        poses_h5 = store_handle['poses'][frame_id, ...].reshape(-1, 3)
        resce_h5 = store_handle['resce'][frame_id, ...]
        hmap2_h5 = store_handle['hmap2'][frame_id, ...]
        udir2_h5 = store_handle['udir2'][frame_id, ...]
        print(store_handle['udir2'])

        print('[{}] drawing image #{:d} ...'.format(self.name_desc, img_id))
        print(np.min(frame_h5), np.max(frame_h5))
        print(np.histogram(frame_h5, range=(1e-4, np.max(frame_h5))))
        print(np.min(poses_h5, axis=0), np.max(poses_h5, axis=0))
        print(resce_h5)
        resce3 = resce_h5[0:4]
        cube = iso_cube()
        cube.load(resce3)
        cube.show_dims()
        from colour import Color
        colors = [Color('orange').rgb, Color('red').rgb, Color('lime').rgb]
        fig, _ = mpplot.subplots(nrows=2, ncols=3, figsize=(3 * 5, 2 * 5))
        pose_raw = poses_h5
        joint_id = self.join_num - 1
        olmap_h5 = udir2_h5[..., :self.join_num]
        uomap_h5 = udir2_h5[..., self.join_num:]
        hmap2 = hmap2_h5[..., joint_id]
        olmap = olmap_h5[..., joint_id]
        uomap = uomap_h5[..., 3 * joint_id:3 * (joint_id + 1)]
        depth_crop = frame_h5

        # ax = mpplot.subplot(2, 3, 2)
        # ax.imshow(depth_crop, cmap=mpplot.cm.bone_r)
        # pose3d = cube.transform_center_shrink(poses_h5)
        # pose2d, _ = cube.project_ortho(pose3d, roll=0, sort=False)
        # pose2d *= sizel
        # args.data_draw.draw_pose2d(
        #     ax, thedata,
        #     pose2d,
        # )

        from utils.image_ops import draw_hmap2, draw_olmap, draw_uomap, draw_udir2
        ax = mpplot.subplot(2, 3, 3)
        draw_hmap2(fig, ax, frame_h5, hmap2)

        ax = mpplot.subplot(2, 3, 4)
        draw_udir2(fig, ax, olmap, uomap)

        ax = mpplot.subplot(2, 3, 5)
        draw_uomap(fig, ax, frame_h5, uomap)

        ax = mpplot.subplot(2, 3, 6)
        draw_olmap(fig, ax, frame_h5, olmap)

        ax = mpplot.subplot(2, 3, 2)
        pose_out = self.yanker_hmap(
            resce_h5, hmap2_h5, olmap_h5, uomap_h5,
            frame_h5, self.caminfo)
        err_re = np.sum(np.abs(pose_out - pose_raw))
        if 1e-2 < err_re:
            print('ERROR: reprojection error: {}'.format(err_re))
        else:
            print('reprojection looks good: {}'.format(err_re))
        ax.imshow(depth_crop, cmap=mpplot.cm.bone_r)
        pose3d = cube.transform_center_shrink(pose_out)
        pose2d, _ = cube.project_ortho(pose3d, roll=0, sort=False)
        pose2d *= self.crop_size
        args.data_draw.draw_pose2d(
            ax, thedata,
            pose2d,
        )

        ax = mpplot.subplot(2, 3, 1)
        mpplot.gca().set_title('test image - {:d}'.format(img_id))
        img_name = args.data_io.index2imagename(img_id)
        img = args.data_io.read_image(self.data_inst.images_join(img_name, mode))
        ax.imshow(img, cmap=mpplot.cm.bone_r)
        args.data_draw.draw_pose2d(
            ax, thedata,
            args.data_ops.raw_to_2d(pose_raw, thedata)
        )
        rects = cube.proj_rects_3(
            args.data_ops.raw_to_2d, self.caminfo
        )
        for ii, rect in enumerate(rects):
            rect.draw(ax, colors[ii])

        # hmap2 = self.data_module.ops.raw_to_heatmap2(
        #     pose_raw, cube, self.hmap_size, self.caminfo
        # )
        # offset, olmap, uomap = self.data_module.ops.raw_to_udir2(
        #     frame_h5, pose_raw, cube, self.hmap_size, self.caminfo
        # )

        fig.tight_layout()
        mpplot.savefig(os.path.join(
            self.predict_dir,
            'draw_{}_{}.png'.format(self.name_desc, img_id)))
        if self.args.show_draw:
            mpplot.show()
        print('[{}] drawing image #{:d} - done.'.format(
            self.name_desc, img_id))

    def get_model(
            self, input_tensor, is_training,
            bn_decay, regu_scale,
            hg_repeat=2, scope=None):
        """ input_tensor: BxHxWxC
            out_dim: BxHxWx(J*5), where J is number of joints
        """
        end_points = {}
        self.end_point_list = []
        final_endpoint = 'hourglass_{}'.format(hg_repeat - 1)
        num_joint = self.join_num
        # num_out_map = num_joint * 5  # hmap2, olmap, uomap
        num_feature = 128

        def add_and_check_final(name, net):
            end_points[name] = net
            return name == final_endpoint

        from tensorflow.contrib import slim
        from model.incept_resnet import incept_resnet
        from model.hourglass import hourglass
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
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                    sc = 'stage64'
                    net = incept_resnet.conv_maxpool(net, scope=sc)
                    # net = incept_resnet.resnet_k(
                    #     net, scope='stage64_residual')
                    # net = incept_resnet.reduce_net(
                    #     net, scope='stage64_reduce')
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                    sc = 'stage32_pre'
                    net = incept_resnet.resnet_k(
                        net, scope='stage32_res')
                    net = slim.conv2d(
                        net, num_feature, 1, scope='stage32_out')
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                for hg in range(hg_repeat):
                    sc = 'hourglass_{}'.format(hg)
                    with tf.variable_scope(sc):
                        # branch0 = hourglass.hg_net(
                        #     net, 2, scope=sc + '_hg')
                        branch0 = incept_resnet.resnet_k(
                            net, scope='_res')
                        branch_hm2 = slim.conv2d(
                            branch0, num_joint, 1,
                            # normalizer_fn=None, activation_fn=tf.nn.softmax)
                            normalizer_fn=None, activation_fn=None)
                        branch_olm = slim.conv2d(
                            branch0, num_joint, 1,
                            # normalizer_fn=None, activation_fn=tf.nn.relu)
                            normalizer_fn=None, activation_fn=None)
                        branch_uom = slim.conv2d(
                            branch0, num_joint * 3, 1,
                            # normalizer_fn=None, activation_fn=tf.nn.sigmoid)
                            normalizer_fn=None, activation_fn=None)
                        net_maps = tf.concat(
                            [branch_hm2, branch_olm, branch_uom],
                            axis=-1)
                        self.end_point_list.append(sc)
                        if add_and_check_final(sc, net_maps):
                            # flat_soft = tf.reshape(branch_hit, [-1, num_vol, num_joint])
                            # flat_soft = tf.nn.softmax(flat_soft, dim=1)
                            # branch_hit = tf.reshape(flat_soft, [-1, vol_shape, num_joint])
                            # net_maps = tf.concat(
                            #     [branch_hit, branch_olm, branch_uom],
                            #     axis=-1)
                            return net_maps, end_points
                        branch1 = slim.conv2d(
                            net_maps, num_feature, 1)
                        net = net + branch0 + branch1
        raise ValueError('final_endpoint (%s) not recognized', final_endpoint)

    def placeholder_inputs(self, batch_size=None):
        frames_tf = tf.placeholder(
            tf.float32, shape=(
                batch_size,
                self.crop_size, self.crop_size,
                1))
        # hmap2_tf = tf.placeholder(
        #     tf.float32, shape=(
        #         batch_size,
        #         self.hmap_size, self.hmap_size,
        #         self.join_num))
        # olmap_tf = tf.placeholder(
        #     tf.float32, shape=(
        #         batch_size,
        #         self.hmap_size, self.hmap_size,
        #         self.join_num))
        # uomap_tf = tf.placeholder(
        #     tf.float32, shape=(
        #         batch_size,
        #         self.hmap_size, self.hmap_size,
        #         self.join_num * 3))
        poses_tf = tf.placeholder(
            tf.float32, shape=(
                batch_size,
                self.hmap_size, self.hmap_size,
                self.join_num * 5))
        Placeholders = namedtuple("Placeholders", "frames_tf poses_tf")
        return Placeholders(frames_tf, poses_tf)

    @staticmethod
    def smooth_l1(xa):
        return tf.where(
            1 < xa,
            xa - 0.5,
            0.5 * (xa ** 2)
        )

    def get_loss(self, pred, echt, end_points):
        """ simple sum-of-squares loss
            pred: BxHxWx(J*5)
            echt: BxHxWx(J*5)
        """
        num_j = self.join_num
        loss_udir = 0
        loss_unit = 0
        for name, net in end_points.items():
            if not name.startswith('hourglass_'):
                continue
            # loss_hm2 = tf.reduce_mean(
            #     tf.nn.cross_entropy_with_logits(
            #         targets=echt[:, :num_j],
            #         logits=pred[:, :num_j])
            # )
            loss_udir += tf.reduce_mean(
                self.smooth_l1(tf.abs(net - echt)))
            uomap_pred = tf.reshape(
                net[..., (2 * num_j):],
                (-1, 3))
            loss_unit += tf.reduce_mean(
                self.smooth_l1(tf.abs(
                    1 - tf.reduce_sum(uomap_pred ** 2, axis=-1))))
        loss_reg = tf.add_n(tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES))
        return loss_udir, loss_unit, loss_reg
