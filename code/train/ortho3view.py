import os
import sys
from importlib import import_module
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import h5py
import matplotlib.pyplot as mpplot
from base_regre import base_regre

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
sys.path.append(BASE_DIR)
file_pack = getattr(
    import_module('utils.coder'),
    'file_pack'
)
iso_cube = getattr(
    import_module('utils.iso_boxes'),
    'iso_cube'
)


class ortho3view(base_regre):
    def __init__(self, args):
        super(ortho3view, self).__init__(args)
        self.num_channel = 3
        self.num_appen = 4

    def receive_data(self, thedata, args):
        """ Receive parameters specific to the data """
        super(ortho3view, self).receive_data(thedata, args)
        self.provider_worker = args.data_provider.prow_ortho3v
        self.yanker = self.provider.yank_ortho3v

    def draw_random(self, thedata, args):
        from cv2 import resize as cv2resize
        with h5py.File(self.appen_train, 'r') as h5file:
            store_size = h5file['index'].shape[0]
            frame_id = np.random.choice(store_size)
            img_id = h5file['index'][frame_id, 0]
            frame_h5 = h5file['frame'][frame_id, ...]
            poses_h5 = h5file['poses'][frame_id, ...].reshape(-1, 3)
            resce_h5 = h5file['resce'][frame_id, ...]
            print(np.min(frame_h5), np.max(frame_h5))
            print(np.histogram(frame_h5, range=(1e-4, np.max(frame_h5))))

        print('[{}] drawing image #{:d}'.format(self.name_desc, img_id))
        mpplot.subplots(nrows=2, ncols=2, figsize=(3 * 5, 3 * 5))

        resce3 = resce_h5[0:4]
        cube = iso_cube()
        cube.load(resce3)
        sizel = np.floor(resce3[0]).astype(int)
        for spi in range(3):
            mpplot.subplot(3, 3, spi + 7)
            img = frame_h5[..., spi]
            mpplot.imshow(
                cv2resize(img, (sizel, sizel)),
                cmap='bone')
            pose2d, _ = cube.project_pca(poses_h5, roll=spi, sort=False)
            pose2d *= sizel
            args.data_draw.draw_pose2d(
                thedata,
                pose2d,
            )
            mpplot.gca().axis('off')

        mpplot.subplot(3, 3, 3)
        img_name = args.data_io.index2imagename(img_id)
        img = args.data_io.read_image(os.path.join(self.image_dir, img_name))
        mpplot.imshow(img, cmap='bone')
        pose_raw = self.yanker(poses_h5, resce_h5, self.caminfo)
        args.data_draw.draw_pose2d(
            thedata,
            args.data_ops.raw_to_2d(pose_raw, thedata)
        )

        mpplot.subplot(3, 3, 1)
        annot_line = args.data_io.get_line(
            thedata.training_annot_cleaned, img_id)
        img_name, pose_raw = args.data_io.parse_line_annot(annot_line)
        img = args.data_io.read_image(os.path.join(self.image_dir, img_name))
        mpplot.imshow(img, cmap='bone')
        args.data_draw.draw_pose2d(
            thedata,
            args.data_ops.raw_to_2d(pose_raw, thedata))

        img_name, frame, poses, resce = self.provider_worker(
            annot_line, self.image_dir, thedata)
        poses = poses.reshape(-1, 3)
        if (
                (1e-4 < np.linalg.norm(frame_h5 - frame)) or
                (1e-4 < np.linalg.norm(poses_h5 - poses))
        ):
            print(np.linalg.norm(frame_h5 - frame))
            print(np.linalg.norm(poses_h5 - poses))
            _, frame_1, _, _ = self.provider_worker(
                annot_line, self.image_dir, thedata)
            print(np.linalg.norm(frame_1 - frame))
            with h5py.File('/tmp/111', 'w') as h5file:
                h5file.create_dataset(
                    'frame', data=frame_1, dtype=np.float32
                )
            with h5py.File('/tmp/111', 'r') as h5file:
                frame_2 = h5file['frame'][:]
                print(np.linalg.norm(frame_1 - frame_2))
            print('ERROR - h5 storage corrupted!')
        resce3 = resce[0:4]
        cube = iso_cube()
        cube.load(resce3)
        sizel = np.floor(resce3[0]).astype(int)
        for spi in range(3):
            mpplot.subplot(3, 3, spi + 4)
            img = frame[..., spi]
            mpplot.imshow(
                cv2resize(img, (sizel, sizel)),
                cmap='bone')
            pose2d, _ = cube.project_pca(poses, roll=spi, sort=False)
            pose2d *= sizel
            args.data_draw.draw_pose2d(
                thedata,
                pose2d,
            )
            mpplot.gca().axis('off')

        mpplot.savefig(os.path.join(
            args.predict_dir,
            'draw_{}.png'.format(self.__class__.__name__)))
        mpplot.show()

    def get_model(
            self, input_tensor, is_training,
            scope=None, final_endpoint='stage_out'):
        """ frames_tf: BxHxWxC
            out_dim: BxJ, where J is flattened 3D locations
        """
        end_points = {}
        self.end_point_list = []

        def add_and_check_final(name, net):
            end_points[name] = net
            return name == final_endpoint

        with tf.variable_scope(
                scope, self.name_desc, [input_tensor]):
            with slim.arg_scope(
                    [slim.batch_norm, slim.dropout],
                    is_training=is_training), \
                slim.arg_scope(
                    [slim.fully_connected],
                    weights_regularizer=slim.l2_regularizer(0.00004),
                    biases_regularizer=slim.l2_regularizer(0.00004),
                    activation_fn=None, normalizer_fn=None), \
                slim.arg_scope(
                    [slim.max_pool2d, slim.avg_pool2d],
                    stride=1, padding='SAME'), \
                slim.arg_scope(
                    [slim.conv2d],
                    stride=1, padding='SAME',
                    activation_fn=tf.nn.relu,
                    weights_regularizer=slim.l2_regularizer(0.00004),
                    biases_regularizer=slim.l2_regularizer(0.00004),
                    normalizer_fn=slim.batch_norm):
                with tf.variable_scope('stage128'):
                    sc = 'stage128_image'
                    net = slim.conv2d(
                        input_tensor, 16, 3, scope='conv128_3x3_1')
                    net = slim.conv2d(
                        net, 16, 3, stride=2, scope='conv128_3x3_2')
                    net = slim.max_pool2d(
                        net, 3, scope='maxpool128_3x3_1')
                    net = slim.conv2d(
                        net, 32, 3, scope='conv64_3x3_1')
                    net = slim.max_pool2d(
                        net, 3, stride=2, scope='maxpool64_3x3_2')
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                with tf.variable_scope('stage32'):
                    sc = 'stage32_image'
                    net = slim.conv2d(
                        net, 64, 3, scope='conv32_3x3_1')
                    net = slim.max_pool2d(
                        net, 3, stride=2, scope='maxpool32_3x3_2')
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                with tf.variable_scope('stage16'):
                    sc = 'stage16_image'
                    net = slim.conv2d(
                        net, 128, 3, scope='conv16_3x3_1')
                    net = slim.max_pool2d(
                        net, 3, stride=2, scope='maxpool16_3x3_2')
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                with tf.variable_scope('stage8'):
                    sc = 'stage_out'
                    net = slim.flatten(net)
                    net = slim.fully_connected(
                        net, 2592,
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        scope='fullconn8')
                    net = slim.dropout(
                        net, 0.5, scope='dropout8')
                    net = slim.fully_connected(
                        net, self.out_dim, scope='output8')
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
        return frames_tf, poses_tf
