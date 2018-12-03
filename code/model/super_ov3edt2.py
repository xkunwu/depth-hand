import os
from importlib import import_module
import numpy as np
from collections import namedtuple
import tensorflow as tf
import matplotlib.pyplot as mpplot
from model.super_edt2 import super_edt2
from utils.iso_boxes import iso_cube


class super_ov3edt2(super_edt2):
    @staticmethod
    def get_trainer(args, new_log):
        from train.train_super_edt2 import train_super_edt2
        return train_super_edt2(args, new_log)

    def __init__(self, args):
        super(super_ov3edt2, self).__init__(args)
        self.batch_allot = getattr(
            import_module('model.batch_allot'),
            'batch_ov3edt2'
        )
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
        self.batch_data['batch_frame'] = \
            store_handle['ortho3'][self.batch_beg:batch_end, ...]
        self.batch_data['batch_poses'] = \
            store_handle['pose_c'][self.batch_beg:batch_end, ...]
        self.batch_data['batch_edt2'] = \
            store_handle['ov3edt2'][self.batch_beg:batch_end, ...]
        self.batch_data['batch_index'] = \
            store_handle['index'][self.batch_beg:batch_end, ...]
        self.batch_data['batch_resce'] = \
            store_handle['resce'][self.batch_beg:batch_end, ...]
        self.batch_beg = batch_end
        return self.batch_data

    def receive_data(self, thedata, args):
        """ Receive parameters specific to the data """
        super(super_ov3edt2, self).receive_data(thedata, args)
        thedata.hmap_size = self.hmap_size
        self.out_dim = self.join_num * 3
        self.store_name = {
            'index': thedata.annotation,
            'poses': thedata.annotation,
            'resce': thedata.annotation,
            'pose_c': 'pose_c',
            'clean': 'clean_{}'.format(self.crop_size),
            'ortho3': 'ortho3_{}'.format(self.crop_size),
            'ov3edt2': 'ov3edt2_{}'.format(self.hmap_size),
        }
        self.frame_type = 'clean'

    def yanker(self, pose_local, resce, caminfo):
        cube = iso_cube()
        cube.load(resce)
        return cube.transform_add_center(pose_local)
        # return cube.transform_expand_move(pose_local)

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

    def draw_random(self, thedata, args):
        # mode = 'train'
        mode = 'test'
        store_handle = self.store_handle[mode]
        index_h5 = store_handle['index']
        store_size = index_h5.shape[0]
        frame_id = np.random.choice(store_size)
        # frame_id = 886  # frame_id = img_id - 1
        # frame_id = 125  # frame_id = img_id - 1
        # frame_id = 218  # palm
        frame_id = 598
        frame_id = 239
        img_id = index_h5[frame_id, ...]
        frame_h5 = store_handle['ortho3'][frame_id, ...]
        poses_h5 = store_handle['pose_c'][frame_id, ...].reshape(-1, 3)
        resce_h5 = store_handle['resce'][frame_id, ...]
        ov3edt2_h5 = store_handle['ov3edt2'][frame_id, ...]

        print('[{}] drawing image #{:d} ...'.format(self.name_desc, img_id))
        print(np.min(frame_h5), np.max(frame_h5))
        print(np.histogram(frame_h5, range=(1e-4, np.max(frame_h5))))
        print(np.min(poses_h5, axis=0), np.max(poses_h5, axis=0))
        from colour import Color
        colors = [Color('orange').rgb, Color('red').rgb, Color('lime').rgb]
        fig, _ = mpplot.subplots(nrows=3, ncols=4, figsize=(4 * 5, 3 * 5))
        resce3 = resce_h5[0:4]
        cube = iso_cube()
        cube.load(resce3)

        ax = mpplot.subplot(3, 4, 1)
        mpplot.gca().set_title('test image - {:d}'.format(img_id))
        img_name = args.data_io.index2imagename(img_id)
        img = args.data_io.read_image(self.data_inst.images_join(img_name, mode))
        ax.imshow(img, cmap=mpplot.cm.bone_r)
        pose_raw = self.yanker(poses_h5, resce_h5, self.caminfo)
        args.data_draw.draw_pose2d(
            ax, thedata,
            args.data_ops.raw_to_2d(pose_raw, thedata)
        )
        rects = cube.proj_rects_3(
            args.data_ops.raw_to_2d, self.caminfo
        )
        for ii, rect in enumerate(rects):
            rect.draw(ax, colors[ii])

        for spi in range(3):
            ax = mpplot.subplot(3, 4, spi + 2)
            img = frame_h5[..., spi]
            ax.imshow(img, cmap=mpplot.cm.bone_r)
            # pose3d = poses_h5
            pose3d = cube.trans_scale_to(poses_h5)
            pose2d, _ = cube.project_ortho(pose3d, roll=spi, sort=False)
            pose2d *= self.crop_size
            args.data_draw.draw_pose2d(
                ax, thedata,
                pose2d,
            )

        from utils.image_ops import draw_edt2
        joint_id = self.join_num - 1
        for spi in range(3):
            ax = mpplot.subplot(3, 4, spi + 6)
            edt2 = ov3edt2_h5[..., spi * self.join_num + joint_id]
            draw_edt2(fig, ax, edt2)

        joint_id = self.join_num - 1 - 9
        for spi in range(3):
            ax = mpplot.subplot(3, 4, spi + 10)
            edt2 = ov3edt2_h5[..., spi * self.join_num + joint_id]
            draw_edt2(fig, ax, edt2)

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
        final_endpoint = 'stage_out'
        num_joint = self.join_num
        # num_out_map = num_joint * 5  # hmap2, olmap, uomap
        num_feature = 64

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
                        branch0 = hourglass.hg_net(
                            net, 2, scope=sc + '_hg')
                        branch0 = incept_resnet.resnet_k(
                            branch0, scope='_res')
                        net_maps = slim.conv2d(
                            branch0, num_joint * 3, 1,
                            # normalizer_fn=None, activation_fn=tf.nn.softmax)
                            normalizer_fn=None, activation_fn=None)
                        self.end_point_list.append(sc)
                        if add_and_check_final(sc, net_maps):
                            return net_maps, end_points
                        branch1 = slim.conv2d(
                            net_maps, num_feature, 1)
                        net = net + branch0 + branch1
                with tf.variable_scope('stage32'):
                    sc = 'stage32_post'
                    # net = incept_resnet.conv_maxpool(net, scope=sc)
                    net = slim.max_pool2d(net, 3, scope=sc)
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                with tf.variable_scope('stage16'):
                    sc = 'stage16'
                    net = incept_resnet.conv_maxpool(net, scope=sc)
                    # net = slim.max_pool2d(net, 3, scope=sc)
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                with tf.variable_scope('stage8'):
                    sc = 'stage_out'
                    net = incept_resnet.pullout8(
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
                self.crop_size, self.crop_size,
                3))
        poses_tf = tf.placeholder(
            tf.float32, shape=(
                batch_size,
                self.out_dim))
        ov3edt2_tf = tf.placeholder(
            tf.float32, shape=(
                batch_size,
                self.hmap_size, self.hmap_size,
                self.join_num * 3))
        Placeholders = namedtuple("Placeholders", "frames_tf poses_tf ext_tf")
        return Placeholders(frames_tf, poses_tf, ov3edt2_tf)
