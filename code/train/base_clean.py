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
    """ This class use cleaned data from 3D PCA bounding cube.
    """
    def __init__(self):
        super(base_clean, self).__init__()
        self.num_appen = 11

    def receive_data(self, thedata, args):
        """ Receive parameters specific to the data """
        super(base_clean, self).receive_data(thedata, args)
        self.provider_worker = args.data_provider.prow_cleaned
        self.yanker = self.provider.yank_cleaned

    def draw_random(self, thedata, args):
        import matplotlib.pyplot as mpplot

        # from colour import Color
        # points3 = np.random.rand(1000, 3)
        # points3[:, 1] *= 2
        # points3[:, 2] *= 4
        # cube = iso_cube()
        # cube.build(points3, 0)
        # corners = cube.get_corners()
        # ax = mpplot.subplot(projection='3d')
        # cube.draw_cube_wire(corners)
        # pose_trans = cube.transform(points3)
        # ax.scatter(
        #     pose_trans[:, 0], pose_trans[:, 1], pose_trans[:, 2],
        #     color=Color('lightsteelblue').rgb)
        # mpplot.show()
        # sys.exit()

        with h5py.File(os.path.join(self.prep_dir, self.appen_train), 'r') as h5file:
            store_size = h5file['index'].shape[0]
            frame_id = np.random.choice(store_size)
            img_id = h5file['index'][frame_id, 0]
            frame_h5 = np.squeeze(h5file['frame'][frame_id, ...], -1)
            poses_h5 = h5file['poses'][frame_id, ...].reshape(-1, 3)
            resce_h5 = h5file['resce'][frame_id, ...]
            # print(np.histogram(frame_h5))
            # print(poses_h5)

        print('[{}] drawing pose #{:d}'.format(self.__class__.__name__, img_id))
        resce2 = resce_h5[0:3]
        resce3 = resce_h5[3:11]
        fig_size = (2 * 5, 2 * 5)
        mpplot.subplots(nrows=2, ncols=2, figsize=fig_size)

        mpplot.subplot(2, 2, 3)
        mpplot.imshow(frame_h5, cmap='bone')
        pose_raw = args.data_ops.pca_to_raw(poses_h5, resce3)
        args.data_draw.draw_pose2d(
            thedata,
            args.data_ops.raw_to_2d(pose_raw, thedata, resce2)
        )

        mpplot.subplot(2, 2, 4)
        img_name = args.data_io.index2imagename(img_id)
        img = args.data_io.read_image(os.path.join(self.image_dir, img_name))
        mpplot.imshow(img, cmap='bone')
        pose_raw = self.yanker(poses_h5, resce_h5)
        args.data_draw.draw_pose2d(
            thedata,
            args.data_ops.raw_to_2d(pose_raw, thedata)
        )

        mpplot.subplot(2, 2, 1)
        annot_line = args.data_io.get_line(
            thedata.training_annot_cleaned, img_id)
        img_name, pose_raw = args.data_io.parse_line_annot(annot_line)
        img = args.data_io.read_image(os.path.join(self.image_dir, img_name))
        mpplot.imshow(img, cmap='bone')
        rect = iso_rect(resce_h5[1:3], self.crop_size / resce_h5[0])
        rect.draw()
        args.data_draw.draw_pose2d(
            thedata,
            args.data_ops.raw_to_2d(pose_raw, thedata))

        mpplot.subplot(2, 2, 2)
        img_name, frame, poses, resce = self.provider_worker(
            annot_line, self.image_dir, thedata)
        frame = np.squeeze(frame, axis=-1)
        poses = poses.reshape(-1, 3)
        if (
                (1e-4 < np.linalg.norm(frame_h5 - frame)) or
                (1e-4 < np.linalg.norm(poses_h5 - poses))
        ):
            print(np.linalg.norm(frame_h5 - frame))
            print(np.linalg.norm(poses_h5 - poses))
            print('ERROR - h5 storage corrupted!')
        resce2 = resce[0:3]
        resce3 = resce[3:11]
        mpplot.imshow(frame, cmap='bone')
        pose_raw = args.data_ops.pca_to_raw(poses, resce3)
        args.data_draw.draw_pose2d(
            thedata,
            args.data_ops.raw_to_2d(pose_raw, thedata, resce2)
        )

        mpplot.savefig(os.path.join(
            args.data_inst.predict_dir,
            'draw_{}.png'.format(self.__class__.__name__)))
        mpplot.show()
