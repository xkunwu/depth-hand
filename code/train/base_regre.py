import tensorflow as tf
import os
import sys
from importlib import import_module
import numpy as np
from timeit import default_timer as timer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
tf_util = import_module('tf_util')
file_pack = getattr(
    import_module('coder'),
    'file_pack'
)


class base_regre():
    """ This class holds baseline training approach using plain regression.
    """
    def __init__(self, out_dir):
        self.train_dir = os.path.join(out_dir, 'cropped')
        self.appen_train = os.path.join(self.train_dir, 'appen_train.h5')
        self.appen_test = os.path.join(self.train_dir, 'appen_test.h5')
        self.crop_size = 96

    def preallocate(self, batch_size, image_size, pose_dim):
        self.batch_index = np.empty(shape=(batch_size, 1))
        self.batch_frame = np.empty(shape=(batch_size, image_size, image_size))
        self.batch_poses = np.empty(shape=(batch_size, pose_dim))
        self.batch_resce = np.empty(shape=(batch_size, 3))

    def start_train(self, filepack):
        pass

    def fetch_batch(self, filepack, provider):
        pass

    def tweak_args(self, args):
        """ Tweak algorithm specific parameters """
        args.crop_size = self.crop_size

    def prepare_data(self, thedata, file_annot, file_appen):
        batch_size = self.batch_size
        num_line = int(sum(1 for line in file_annot))
        file_annot.seek(0)
        num_batches = int(float(num_line) // batch_size)
        print('preparing data: {:d} lines, subdivided into {:d} batches (batch size {:d}) ...'.format(
            num_line, num_batches, batch_size
        ))
        all_index = np.empty(shape=(num_batches, batch_size, 1))
        all_frame = np.empty(shape=(num_batches, batch_size, self.crop_size, self.crop_size))
        all_poses = np.empty(shape=(num_batches, batch_size, self.pose_dim))
        all_resce = np.empty(shape=(num_batches, batch_size, 3))
        # for bi in range(num_batches):
        #     batch_0 = batch_size * bi
        #     batch_1 = batch_size * (bi + 1)
        bi = 0
        while True:
            res = self.provider.put2d(
                thedata, file_annot, self.image_dir, self.batch_size,
                self.batch_index, self.batch_frame, self.batch_poses, self.batch_resce
            )
            if 0 > res:
                break
            all_index[bi, :] = self.batch_index
            all_frame[bi, :, :, :] = self.batch_frame
            all_poses[bi, :, :] = self.batch_poses
            all_resce[bi, :] = self.batch_resce
            bi += 1
            sys.stdout.write('.')
            sys.stdout.flush()
        sys.stdout.write('\n')
        file_appen.create_dataset(
            'frame', data=all_frame, dtype=float
        )
        file_appen.create_dataset(
            'poses', data=all_poses, dtype=float
        )
        file_appen.create_dataset(
            'resce', data=all_resce, dtype=float
        )

    def receive_data(self, thedata, args):
        """ Receive parameters specific to the data """
        self.batch_size = args.batch_size
        self.pose_dim = thedata.join_num * 3
        self.image_dir = thedata.training_images
        self.provider = args.data_provider
        first_run = False
        if not os.path.exists(self.train_dir):
            first_run = True
            os.makedirs(self.train_dir)
        if ((not os.path.exists(self.appen_train)) or
                (not os.path.exists(self.appen_test))):
            first_run = True
        if args.rebuild_data:
            first_run = True
        if not first_run:
            return
        self.preallocate(self.batch_size, self.crop_size, self.pose_dim)
        time_s = timer()
        with file_pack() as filepack:
            file_annot = filepack.push_file(thedata.training_annot_train)
            file_appen = filepack.push_h5(self.appen_train, "w")
            self.prepare_data(thedata, file_annot, file_appen)
        print('training data prepared: {:.4f}'.format(timer() - time_s))
        time_s = timer()
        with file_pack() as filepack:
            file_annot = filepack.push_file(thedata.training_annot_test)
            file_appen = filepack.push_h5(self.appen_test, "w")
            self.prepare_data(thedata, file_annot, file_appen)
        print('testing data prepared: {:.4f}'.format(timer() - time_s))

    @staticmethod
    def placeholder_inputs(batch_size, image_size, pose_dim):
        frames_tf = tf.placeholder(
            tf.float32, shape=(batch_size, image_size, image_size))
        poses_tf = tf.placeholder(
            tf.float32, shape=(batch_size, pose_dim))
        return frames_tf, poses_tf

    @staticmethod
    def get_model(frames_tf, pose_dim, is_training, bn_decay=None):
        """ directly predict all joints' location using regression
            frames_tf: BxHxW
            pose_dim: BxJ, where J is flattened 3D locations
        """
        batch_size = frames_tf.get_shape()[0].value
        end_points = {}
        input_image = tf.expand_dims(frames_tf, -1)

        # Point functions (MLP implemented as conv2d)
        net = tf_util.conv2d(
            input_image, 64, [3, 3],
            padding='VALID', stride=[1, 1],
            bn=True, is_training=is_training,
            scope='conv1', bn_decay=bn_decay)
        net = tf_util.max_pool2d(
            net, [1, 1],
            padding='VALID', scope='maxpool1')
        net = tf_util.conv2d(
            net, 256, [3, 3],
            padding='VALID', stride=[1, 1],
            bn=True, is_training=is_training,
            scope='conv2', bn_decay=bn_decay)
        net = tf_util.max_pool2d(
            net, [1, 1],
            padding='VALID', scope='maxpool2')
        net = tf_util.conv2d(
            net, 1024, [3, 3],
            padding='VALID', stride=[1, 1],
            bn=True, is_training=is_training,
            scope='conv3', bn_decay=bn_decay)
        net = tf_util.max_pool2d(
            net, [1, 1],
            padding='VALID', scope='maxpool3')

        # MLP on global point cloud vector
        net = tf.reshape(net, [batch_size, -1])
        net = tf_util.fully_connected(
            net, 512, bn=True, is_training=is_training,
            scope='fc1', bn_decay=bn_decay)
        net = tf_util.dropout(
            net, keep_prob=0.5, is_training=is_training,
            scope='dp1')
        net = tf_util.fully_connected(
            net, pose_dim, activation_fn=None, scope='fc2')

        return net, end_points

    @staticmethod
    def get_loss(pred, anno, end_points):
        """ simple sum-of-squares loss
            pred: BxJ
            anno: BxJ
        """
        # loss = tf.reduce_sum(tf.pow(tf.subtract(pred, anno), 2)) / 2
        # loss = tf.nn.l2_loss(pred - anno)  # already divided by 2
        loss = tf.reduce_mean(tf.squared_difference(pred, anno)) / 2
        return loss
