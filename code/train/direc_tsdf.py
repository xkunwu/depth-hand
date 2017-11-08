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


class direc_tsdf(base_conv3):
    """ This class holds baseline training approach using 3d CNN.
    """
    def __init__(self, out_dir):
        super(direc_tsdf, self).__init__(out_dir)
        self.train_dir = os.path.join(out_dir, 'dirtsdf')

    def receive_data(self, thedata, args):
        """ Receive parameters specific to the data """
        self.pose_dim = thedata.join_num * 3
        self.image_dir = thedata.training_images
        self.caminfo = thedata
        self.provider = args.data_provider
        self.provider_worker = args.data_provider.prow_dirtsdf
        self.yanker = self.provider.yank_dirtsdf
        self.check_dir(thedata, args)

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
        batchallot.allot(3, 9)
        with file_pack() as filepack:
            file_annot = filepack.push_file(thedata.training_annot_train)
            self.prepare_data(thedata, batchallot, file_annot, self.appen_train)
        with file_pack() as filepack:
            file_annot = filepack.push_file(thedata.training_annot_test)
            self.prepare_data(thedata, batchallot, file_annot, self.appen_test)
        print('data prepared: {}'.format(self.train_dir))

    def draw_random(self, thedata, args):
        import random
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
            frame_h5 = batchallot.batch_frame[frame_id, ...]
            # poses_h5 = batchallot.batch_poses[frame_id, ...].reshape(-1, 3)
            # resce_h5 = batchallot.batch_resce[frame_id, ...]

        print('[{}] drawing pose #{:d}'.format(self.__class__.__name__, img_id))
        for spi in range(3):
            mlab.figure(size=(800, 800))
            volume3 = frame_h5[..., spi]
            mlab.pipeline.volume(mlab.pipeline.scalar_field(volume3))
            mlab.pipeline.image_plane_widget(
                mlab.pipeline.scalar_field(volume3),
                plane_orientation='z_axes',
                slice_index=self.crop_size / 2)
            np.set_printoptions(precision=4)
            # print(volume3[12:20, 12:20, 16])
            mlab.outline()
        mlab.show()

    @staticmethod
    def placeholder_inputs(batch_size, image_size, pose_dim):
        frames_tf = tf.placeholder(
            tf.float32,
            shape=(batch_size, image_size, image_size, image_size, 3))
        poses_tf = tf.placeholder(
            tf.float32, shape=(batch_size, pose_dim))
        return frames_tf, poses_tf
