import tensorflow as tf
import os
import sys
from importlib import import_module
import numpy as np
import h5py
from base_conv3 import base_conv3

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


class direc_tsdf(base_conv3):
    """ This class holds baseline training approach using 3d CNN.
    """
    def __init__(self, args):
        super(direc_tsdf, self).__init__(args)
        self.num_channel = 3

    def receive_data(self, thedata, args):
        """ Receive parameters specific to the data """
        super(direc_tsdf, self).receive_data(thedata, args)
        self.provider_worker = args.data_provider.prow_dirtsdf
        self.yanker = self.provider.yank_dirtsdf

    def draw_random(self, thedata, args):
        from mayavi import mlab

        # pcnt = np.zeros((6, 6, 6))
        # pcnt[2:4, 1:5, 3] = 1
        # print(pcnt[..., 3])
        # for spi in range(3):
        #     mlab.figure(size=(800, 800))
        #     befs = args.data_ops.trunc_belief(pcnt)
        #     volume3 = befs[..., spi]
        #     mlab.pipeline.volume(mlab.pipeline.scalar_field(volume3))
        #     mlab.pipeline.image_plane_widget(
        #         mlab.pipeline.scalar_field(volume3),
        #         plane_orientation='z_axes',
        #         slice_index=self.crop_size / 2)
        #     print(volume3[..., 3])
        #     print(volume3[0, 0, 3], type(volume3[0, 0, 3]))
        #     mlab.outline()
        # mlab.show()
        # sys.exit()

        with h5py.File(self.appen_train, 'r') as h5file:
            store_size = h5file['index'].shape[0]
            frame_id = np.random.choice(store_size)
            img_id = h5file['index'][frame_id, 0]
            frame_h5 = h5file['frame'][frame_id, ...]
            poses_h5 = h5file['poses'][frame_id, ...].reshape(-1, 3)
            # resce_h5 = h5file['resce'][frame_id, ...]

        print('[{}] drawing image #{:d}'.format(self.__class__.__name__, img_id))
        annot_line = args.data_io.get_line(
            thedata.training_annot_cleaned, img_id)
        img_name, frame, poses, resce = self.provider_worker(
            annot_line, self.image_dir, thedata)
        poses = poses.reshape(-1, 3)
        if (
                (1e-4 < np.linalg.norm(frame_h5 - frame)) or
                (1e-4 < np.linalg.norm(poses_h5 - poses))
        ):
            print(np.linalg.norm(frame_h5 - frame))
            print(np.linalg.norm(poses_h5 - poses))
            print('ERROR - h5 storage corrupted!')
        # for spi in range(3):
        #     mlab.figure(size=(800, 800))
        #     volume3 = frame_h5[..., spi]
        #     mlab.pipeline.volume(mlab.pipeline.scalar_field(volume3))
        #     mlab.pipeline.image_plane_widget(
        #         mlab.pipeline.scalar_field(volume3),
        #         plane_orientation='z_axes',
        #         slice_index=self.crop_size / 2)
        #     np.set_printoptions(precision=4)
        #     # print(volume3[12:20, 12:20, 16])
        #     mlab.outline()
        # mlab.show()

    def placeholder_inputs(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        frames_tf = tf.placeholder(
            tf.float32, shape=(
                batch_size,
                self.crop_size, self.crop_size, self.crop_size,
                3))
        poses_tf = tf.placeholder(
            tf.float32, shape=(batch_size, self.out_dim))
        return frames_tf, poses_tf
