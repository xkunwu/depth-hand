import tensorflow as tf
import os
import sys
import numpy as np
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
iso_cube = getattr(
    import_module('iso_boxes'),
    'iso_cube'
)


class ortho3view(base_regre):
    def __init__(self):
        super(ortho3view, self).__init__()
        self.num_channel = 3
        self.num_appen = 11

    def receive_data(self, thedata, args):
        """ Receive parameters specific to the data """
        super(ortho3view, self).receive_data(thedata, args)
        self.provider_worker = args.data_provider.prow_ortho3v
        self.yanker = self.provider.yank_ortho3v

    def draw_random(self, thedata, args):
        import matplotlib.pyplot as mpplot

        with h5py.File(os.path.join(self.prep_dir, self.appen_train), 'r') as h5file:
            store_size = h5file['index'].shape[0]
            frame_id = np.random.choice(store_size)
            img_id = h5file['index'][frame_id, 0]
            frame_h5 = h5file['frame'][frame_id, ...]
            poses_h5 = h5file['poses'][frame_id, ...].reshape(-1, 3)
            resce_h5 = h5file['resce'][frame_id, ...]

        print('[{}] drawing pose #{:d}'.format(self.__class__.__name__, img_id))
        fig_size = (3 * 5, 3 * 5)
        mpplot.subplots(nrows=3, ncols=3, figsize=fig_size)
        resce2 = resce_h5[0:3]
        resce3 = resce_h5[3:11]
        cube = iso_cube()
        cube.load(resce3)
        pose_pca = poses_h5 * resce3[0]  # still in the transformed coordinates
        for spi in range(3):
            mpplot.subplot(3, 3, spi + 7)
            img = frame_h5[..., spi]
            mpplot.imshow(img, cmap='bone')
            pose2d, _ = cube.project_pca(pose_pca, roll=spi, sort=False)
            pose2d = pose2d * resce2[0]
            args.data_draw.draw_pose2d(
                thedata,
                pose2d,
            )
            mpplot.gcf().gca().axis('off')
            # mpplot.tight_layout()

        mpplot.subplot(3, 3, 3)
        img_name = args.data_io.index2imagename(img_id)
        img = args.data_io.read_image(os.path.join(self.image_dir, img_name))
        mpplot.imshow(img, cmap='bone')
        pose_raw = self.yanker(poses_h5, resce_h5)
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
        poses = poses.reshape(-1, 3)
        resce2 = resce[0:3]
        resce3 = resce[3:11]
        cube = iso_cube()
        cube.load(resce3)
        pose_pca = poses * resce3[0]  # still in the transformed coordinates
        for spi in range(3):
            mpplot.subplot(3, 3, spi + 4)
            img = frame[..., spi]
            mpplot.imshow(img, cmap='bone')
            pose2d, _ = cube.project_pca(pose_pca, roll=spi, sort=False)
            pose2d = pose2d * resce2[0]
            args.data_draw.draw_pose2d(
                thedata,
                pose2d,
            )
            mpplot.gcf().gca().axis('off')
            # mpplot.tight_layout()
        mpplot.savefig(os.path.join(
            args.data_inst.predict_dir,
            'draw_{}.png'.format(self.__class__.__name__)))
        mpplot.show()

    def placeholder_inputs(self):
        frames_tf = tf.placeholder(
            tf.float32, shape=(
                self.batch_size,
                self.crop_size, self.crop_size,
                3))
        poses_tf = tf.placeholder(
            tf.float32, shape=(self.batch_size, self.pose_dim))
        return frames_tf, poses_tf

    def get_model(self, frames_tf, is_training, bn_decay=None):
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
            net, self.pose_dim, activation_fn=None, scope='fc2')

        return net, end_points
