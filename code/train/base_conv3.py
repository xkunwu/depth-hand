import tensorflow as tf
import os
import sys
from importlib import import_module
import numpy as np
import h5py
from base_regre import base_regre

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
tf_util = import_module('tf_util')
file_pack = getattr(
    import_module('coder'),
    'file_pack'
)


class base_conv3(base_regre):
    """ This class holds baseline training approach using 3d CNN.
    """
    def __init__(self, out_dir):
        super(base_conv3, self).__init__(out_dir)
        self.crop_size = 32
        self.train_dir = os.path.join(out_dir, 'conv3d')

    class batch_allot:
        def __init__(self, store_size, image_size, pose_dim, batch_size=1):
            self.store_size = store_size
            self.batch_size = batch_size
            self.image_size = image_size
            self.pose_dim = pose_dim
            self.batch_beg = 0
            self.batch_end = self.batch_beg + self.batch_size

        def allot(self, num_channel, num_appen):
            self.batch_index = np.empty(
                shape=(self.batch_size, 1), dtype=np.uint32)
            self.batch_frame = np.empty(
                shape=(self.batch_size, self.image_size, self.image_size, self.image_size, num_channel),
                dtype=int)
            self.batch_poses = np.empty(
                shape=(self.batch_size, self.pose_dim), dtype=np.float32)
            self.batch_resce = np.empty(
                shape=(self.batch_size, num_appen), dtype=np.float32)
            self.batch_bytes = \
                self.batch_index.nbytes + self.batch_frame.nbytes + \
                self.batch_poses.nbytes + self.batch_resce.nbytes

        def assign(self, batch_index, batch_frame, batch_poses, batch_resce):
            self.batch_index = batch_index
            self.batch_frame = batch_frame
            self.batch_poses = batch_poses
            self.batch_resce = batch_resce
            self.batch_bytes = \
                self.batch_index.nbytes + self.batch_frame.nbytes + \
                self.batch_poses.nbytes + self.batch_resce.nbytes

        def fetch_batch(self):
            if self.batch_end >= self.store_size:
                return None
            batch_data = {
                'batch_index': self.batch_index[self.batch_beg:self.batch_end, ...],
                'batch_frame': self.batch_frame[self.batch_beg:self.batch_end, ...],
                'batch_poses': self.batch_poses[self.batch_beg:self.batch_end, ...],
                'batch_resce': self.batch_resce[self.batch_beg:self.batch_end, ...]
            }
            self.batch_beg = self.batch_end
            self.batch_end = self.batch_beg + self.batch_size
            return batch_data

    def check_dir(self, thedata, args):
        first_run = False
        if not os.path.exists(self.train_dir):
            first_run = True
            os.makedirs(self.train_dir)
        if args.rebuild_data:
            first_run = True
        if not first_run:
            return
        batchallot = self.batch_allot(
            args.store_level, self.crop_size, self.pose_dim, args.store_level)
        batchallot.allot(1, 4)
        with file_pack() as filepack:
            file_annot = filepack.push_file(thedata.training_annot_train)
            self.prepare_data(thedata, batchallot, file_annot, self.appen_train)
        with file_pack() as filepack:
            file_annot = filepack.push_file(thedata.training_annot_test)
            self.prepare_data(thedata, batchallot, file_annot, self.appen_test)
        print('data prepared: {}'.format(self.train_dir))

    def receive_data(self, thedata, args):
        """ Receive parameters specific to the data """
        self.pose_dim = thedata.join_num * 3
        self.image_dir = thedata.training_images
        self.provider = args.data_provider
        self.provider_worker = args.data_provider.prow_conv3d
        self.check_dir(thedata, args)

    def draw_random(self, thedata, args):
        import random
        filelist = [f for f in os.listdir(self.train_dir)
                    if os.path.isfile(os.path.join(self.train_dir, f))]
        filename = os.path.join(self.train_dir, random.choice(filelist))
        with h5py.File(filename, 'r') as h5file:
            store_size = h5file['index'][:].shape[0]
            batchallot = self.batch_allot(
                store_size, self.crop_size, self.pose_dim, self.batch_size)
            batchallot.assign(
                h5file['index'][:],
                h5file['frame'][:],
                h5file['poses'][:],
                h5file['resce'][:]
            )
            frame_id = random.randrange(store_size)
            img_id = batchallot.batch_index[frame_id, 0]
            img_crop_resize = batchallot.batch_frame[frame_id, ...]
            pose_raw = batchallot.batch_poses[frame_id, ...].reshape(-1, 3)
            resce = batchallot.batch_resce[frame_id, ...]

        import matplotlib.pyplot as mpplot
        print('[{}] drawing pose #{:d}'.format(self.__class__.__name__, img_id))
        fig_size = (2 * 5, 2 * 5)
        mpplot.subplots(nrows=2, ncols=2, figsize=fig_size)
        mpplot.subplot(1, 4, 1)
        mpplot.show()

    @staticmethod
    def placeholder_inputs(batch_size, image_size, pose_dim):
        frames_tf = tf.placeholder(
            tf.float32,
            shape=(batch_size, image_size, image_size, image_size, 1))
        poses_tf = tf.placeholder(
            tf.float32, shape=(batch_size, pose_dim))
        return frames_tf, poses_tf

    @staticmethod
    def get_model(frames_tf, pose_dim, is_training, bn_decay=None):
        """ directly predict all joints' location using regression
            frames_tf: BxHxWxDx1
            pose_dim: BxJ, where J is flattened 3D locations
        """
        batch_size = frames_tf.get_shape()[0].value
        end_points = {}
        input_image = frames_tf

        net = tf_util.conv3d(
            input_image, 32, [5, 5, 5],
            padding='VALID', stride=[1, 1, 1],
            bn=True, is_training=is_training,
            scope='conv1', bn_decay=bn_decay)
        net = tf_util.max_pool3d(
            net, [2, 2, 2],
            padding='VALID', scope='maxpool1')
        net = tf_util.conv3d(
            net, 64, [3, 3, 3],
            padding='VALID', stride=[1, 1, 1],
            bn=True, is_training=is_training,
            scope='conv2', bn_decay=bn_decay)
        net = tf_util.max_pool3d(
            net, [2, 2, 2],
            padding='VALID', scope='maxpool2')
        net = tf_util.conv3d(
            net, 128, [3, 3, 3],
            padding='VALID', stride=[1, 1, 1],
            bn=True, is_training=is_training,
            scope='conv3', bn_decay=bn_decay)
        # net = tf_util.max_pool3d(
        #     net, [2, 2, 2],
        #     padding='VALID', scope='maxpool3')
        # print(net.shape)

        net = tf.reshape(net, [batch_size, -1])
        net = tf_util.fully_connected(
            net, 4096, bn=True, is_training=is_training,
            scope='fc1', bn_decay=bn_decay)
        net = tf_util.dropout(
            net, keep_prob=0.5, is_training=is_training,
            scope='dp1')
        net = tf_util.fully_connected(
            net, 1024, bn=True, is_training=is_training,
            scope='fc2', bn_decay=bn_decay)
        net = tf_util.dropout(
            net, keep_prob=0.5, is_training=is_training,
            scope='dp2')
        net = tf_util.fully_connected(
            net, pose_dim, activation_fn=None, scope='fc3')

        return net, end_points
