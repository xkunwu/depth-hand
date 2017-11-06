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
iso_rect = getattr(
    import_module('iso_boxes'),
    'iso_rect'
)
iso_cube = getattr(
    import_module('iso_boxes'),
    'iso_cube'
)


class base_clean(base_regre):
    """ This class use cleaned data from 3D PCA bounding box.
    """
    def __init__(self, out_dir):
        super(base_clean, self).__init__(out_dir)
        self.train_dir = os.path.join(out_dir, 'cleaned')

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
        batchallot.allot(1, 11)
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
        self.caminfo = thedata
        self.provider = args.data_provider
        self.provider_worker = args.data_provider.prow_cleaned
        self.yanker = self.provider.yank_cleaned
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
            img_crop_resize = np.squeeze(batchallot.batch_frame[frame_id, ...], -1)
            pose_local = batchallot.batch_poses[frame_id, ...].reshape(-1, 3)
            resce = batchallot.batch_resce[frame_id, ...]

        import matplotlib.pyplot as mpplot
        print('[{}] drawing pose #{:d}'.format(self.__class__.__name__, img_id))
        rect = iso_rect(resce[1:3], self.crop_size / resce[0])
        resce2 = resce[0:3]
        resce3 = resce[3:11]
        fig_size = (3 * 5, 5)
        mpplot.subplots(nrows=1, ncols=2, figsize=fig_size)
        mpplot.subplot(1, 3, 3)
        mpplot.imshow(img_crop_resize, cmap='bone')
        pose_raw = args.data_ops.pca_to_raw(pose_local, resce3)
        args.data_draw.draw_pose2d(
            thedata, img_crop_resize,
            args.data_ops.raw_to_2d(pose_raw, thedata, resce2)
        )
        mpplot.subplot(1, 3, 1)
        annot_line = args.data_io.get_line(
            thedata.training_annot_cleaned, img_id)
        img_name, pose_raw = args.data_io.parse_line_annot(annot_line)
        img = args.data_io.read_image(os.path.join(self.image_dir, img_name))
        mpplot.imshow(img, cmap='bone')
        rect.draw()
        args.data_draw.draw_pose2d(
            thedata, img,
            args.data_ops.raw_to_2d(pose_raw, thedata))
        mpplot.subplot(1, 3, 2)
        img_name, frame, poses, resce = self.provider_worker(
            annot_line, self.image_dir, thedata)
        mpplot.imshow(np.squeeze(frame, axis=2), cmap='bone')
        pose_raw = args.data_ops.pca_to_raw(pose_local, resce3)
        args.data_draw.draw_pose2d(
            thedata, img_crop_resize,
            args.data_ops.raw_to_2d(pose_raw, thedata, resce2)
        )
        mpplot.show()
