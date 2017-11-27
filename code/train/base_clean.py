import os
import sys
from importlib import import_module
import numpy as np
import h5py
from .base_regre import base_regre

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
        # self.num_appen = 11
        self.num_appen = 8

    def receive_data(self, thedata, args):
        """ Receive parameters specific to the data """
        super(base_clean, self).receive_data(thedata, args)
        self.provider_worker = args.data_provider.prow_cleaned
        self.yanker = self.provider.yank_cleaned

    def draw_random(self, thedata, args):
        import matplotlib.pyplot as mpplot
        from cv2 import resize as cv2resize

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

        with h5py.File(self.appen_train, 'r') as h5file:
            store_size = h5file['index'].shape[0]
            frame_id = np.random.choice(store_size)
            img_id = h5file['index'][frame_id, 0]
            frame_h5 = np.squeeze(h5file['frame'][frame_id, ...], -1)
            poses_h5 = h5file['poses'][frame_id, ...].reshape(-1, 3)
            resce_h5 = h5file['resce'][frame_id, ...]
            print(np.min(frame_h5), np.max(frame_h5))
            print(np.histogram(frame_h5, range=(1e-4, np.max(frame_h5))))
            print(np.min(poses_h5, axis=0), np.max(poses_h5, axis=0))

        print('[{}] drawing pose #{:d}'.format(self.__class__.__name__, img_id))
        # resce3 = resce_h5[3:11]
        resce3 = resce_h5[0:8]
        mpplot.subplots(nrows=2, ncols=2, figsize=(2 * 5, 2 * 5))

        mpplot.subplot(2, 2, 3)
        cube = iso_cube()
        cube.load(resce3)
        sizel = np.floor(resce3[0]).astype(int)
        mpplot.imshow(
            cv2resize(frame_h5, (sizel, sizel)),
            cmap='bone')
        pose2d, _ = cube.project_pca(poses_h5, roll=0, sort=False)
        pose2d *= sizel
        args.data_draw.draw_pose2d(
            thedata,
            pose2d,
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
        # resce3 = resce[3:11]
        resce3 = resce[0:8]
        cube = iso_cube()
        cube.load(resce3)
        sizel = np.floor(resce3[0]).astype(int)
        mpplot.imshow(
            cv2resize(frame, (sizel, sizel)),
            cmap='bone')
        pose2d, _ = cube.project_pca(poses, roll=0, sort=False)
        pose2d *= sizel
        args.data_draw.draw_pose2d(
            thedata,
            pose2d,
        )

        mpplot.savefig(os.path.join(
            args.predict_dir,
            'draw_{}.png'.format(self.__class__.__name__)))
        mpplot.show()
