import tensorflow as tf
import os
import sys
from importlib import import_module
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


class ortho3view(base_regre):
    def __init__(self, out_dir):
        super(ortho3view, self).__init__(out_dir)
        self.train_dir = os.path.join(out_dir, 'ortho3v')

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
        batchallot.allot(3, 13)
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
        self.provider_worker = args.data_provider.prow_ortho3v
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
            # pose_raw = batchallot.batch_poses[frame_id, ...].reshape(-1, 3)
            # resce = batchallot.batch_resce[frame_id, ...]

        import matplotlib.pyplot as mpplot
        print('drawing pose #{:d}'.format(img_id))
        fig_size = (4 * 5, 5)
        mpplot.subplots(nrows=1, ncols=4, figsize=fig_size)
        for spi in range(3):
            mpplot.subplot(1, 4, spi + 2)
            mpplot.imshow(img_crop_resize[..., spi], cmap='bone')
            mpplot.gcf().gca().axis('off')
            mpplot.tight_layout()
        mpplot.subplot(1, 4, 1)
        args.data_draw.draw_pose_raw_random(
            thedata,
            thedata.training_images,
            thedata.training_annot_cleaned,
            img_id
        )
        mpplot.show()

    @staticmethod
    def placeholder_inputs(batch_size, image_size, pose_dim):
        frames_tf = tf.placeholder(
            tf.float32, shape=(batch_size, image_size, image_size, 3))
        poses_tf = tf.placeholder(
            tf.float32, shape=(batch_size, pose_dim))
        return frames_tf, poses_tf

    @staticmethod
    def get_model(frames_tf, pose_dim, is_training, bn_decay=None):
        """ directly predict all joints' location using regression
            frames_tf: BxHxWx3
            pose_dim: BxJ, where J is flattened 3D locations
        """
        batch_size = frames_tf.get_shape()[0].value
        end_points = {}
        input_image = frames_tf

        net = tf_util.conv2d(
            input_image, 16, [5, 5],
            padding='VALID', stride=[1, 1],
            bn=True, is_training=is_training,
            scope='conv1', bn_decay=bn_decay)
        net = tf_util.max_pool2d(
            net, [4, 4],
            padding='VALID', scope='maxpool1')
        net = tf_util.conv2d(
            net, 32, [3, 3],
            padding='VALID', stride=[1, 1],
            bn=True, is_training=is_training,
            scope='conv2', bn_decay=bn_decay)
        net = tf_util.max_pool2d(
            net, [2, 2],
            padding='VALID', scope='maxpool2')
        net = tf_util.conv2d(
            net, 64, [3, 3],
            padding='VALID', stride=[1, 1],
            bn=True, is_training=is_training,
            scope='conv3', bn_decay=bn_decay)
        net = tf_util.max_pool2d(
            net, [2, 2],
            padding='VALID', scope='maxpool3')
        # print(net.shape)

        net = tf.reshape(net, [batch_size, -1])
        net = tf_util.fully_connected(
            net, 2592, bn=True, is_training=is_training,
            scope='fc1', bn_decay=bn_decay)
        net = tf_util.dropout(
            net, keep_prob=0.5, is_training=is_training,
            scope='dp1')
        net = tf_util.fully_connected(
            net, pose_dim, activation_fn=None, scope='fc2')

        return net, end_points
