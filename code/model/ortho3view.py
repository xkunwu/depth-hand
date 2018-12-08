""" Hand in Depth
    https://github.com/xkunwu/depth-hand
"""
import os
from importlib import import_module
import numpy as np
from collections import namedtuple
import tensorflow as tf
from tensorflow.contrib import slim
import matplotlib.pyplot as mpplot
from model.base_clean import base_clean
from utils.iso_boxes import iso_cube


class ortho3view(base_clean):
    def __init__(self, args):
        super(ortho3view, self).__init__(args)
        # self.num_channel = 3
        # self.num_appen = 4
        self.batch_allot = getattr(
            import_module('model.batch_allot'),
            'batch_ortho3'
        )

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
        self.batch_data['batch_index'] = \
            store_handle['index'][self.batch_beg:batch_end, ...]
        self.batch_data['batch_resce'] = \
            store_handle['resce'][self.batch_beg:batch_end, ...]
        self.batch_beg = batch_end
        return self.batch_data

    def receive_data(self, thedata, args):
        """ Receive parameters specific to the data """
        super(ortho3view, self).receive_data(thedata, args)
        self.store_name = {
            'index': thedata.annotation,
            'poses': thedata.annotation,
            'resce': thedata.annotation,
            'pose_c': 'pose_c',
            'clean': 'clean_{}'.format(self.crop_size),
            'ortho3': 'ortho3_{}'.format(self.crop_size),
        }
        self.frame_type = 'clean'

    def draw_random(self, thedata, args):
        # mode = 'train'
        mode = 'test'
        store_handle = self.store_handle[mode]
        index_h5 = store_handle['index']
        store_size = index_h5.shape[0]
        frame_id = np.random.choice(store_size)
        # frame_id = 0  # frame_id = img_id - 1
        img_id = index_h5[frame_id, ...]
        frame_h5 = store_handle['ortho3'][frame_id, ...]
        poses_h5 = store_handle['pose_c'][frame_id, ...].reshape(-1, 3)
        resce_h5 = store_handle['resce'][frame_id, ...]

        print('[{}] drawing image #{:d} ...'.format(self.name_desc, img_id))
        from colour import Color
        colors = [Color('orange').rgb, Color('red').rgb, Color('lime').rgb]
        fig, _ = mpplot.subplots(nrows=2, ncols=3, figsize=(3 * 5, 2 * 5))

        resce3 = resce_h5[0:4]
        cube = iso_cube()
        cube.load(resce3)
        # need to maintain both image and poses at the same scale
        for spi in range(3):
            ax = mpplot.subplot(2, 3, spi + 4)
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
            # ax.axis('off')

        ax = mpplot.subplot(2, 3, 1)
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
            scope=None, final_endpoint='stage_out'):
        """ frames_tf: BxHxWxC
            out_dim: Bx(Jx3), where J is number of joints
        """
        end_points = {}
        self.end_point_list = []

        def add_and_check_final(name, net):
            end_points[name] = net
            return name == final_endpoint

        from incept_resnet import incept_resnet
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
                    sc = 'stage128_image'
                    net = slim.conv2d(input_tensor, 8, 3)
                    net = incept_resnet.conv_maxpool(net, scope=sc)
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                    sc = 'stage64_image'
                    net = incept_resnet.conv_maxpool(net, scope=sc)
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                with tf.variable_scope('stage32'):
                    sc = 'stage32_image'
                    net = incept_resnet.conv_maxpool(net, scope=sc)
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                with tf.variable_scope('stage16'):
                    sc = 'stage16_image'
                    net = incept_resnet.conv_maxpool(net, scope=sc)
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                with tf.variable_scope('stage8'):
                    sc = 'stage_out'
                    net = slim.flatten(net)
                    net = slim.fully_connected(
                        net, 2592,
                        scope='fullconn8')
                    net = slim.dropout(
                        net, 0.5, scope='dropout8')
                    net = slim.fully_connected(
                        net, self.out_dim,
                        activation_fn=None,
                        normalizer_fn=None,
                        scope='output8')
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
        raise ValueError('final_endpoint (%s) not recognized', final_endpoint)

    def placeholder_inputs(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        frames_tf = tf.placeholder(
            tf.float32, shape=(
                batch_size,
                self.crop_size, self.crop_size,
                3))
        poses_tf = tf.placeholder(
            tf.float32, shape=(batch_size, self.out_dim))
        Placeholders = namedtuple("Placeholders", "frames_tf poses_tf")
        return Placeholders(frames_tf, poses_tf)
