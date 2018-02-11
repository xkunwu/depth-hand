import os
# import sys
# from importlib import import_module
import numpy as np
import h5py
from .base_regre import base_regre
# from utils.coder import file_pack
# from utils.iso_boxes import iso_rect
from utils.iso_boxes import iso_cube


class base_clean(base_regre):
    """ This class use cleaned data from 3D PCA bounding cube.
    """
    def __init__(self, args):
        super(base_clean, self).__init__(args)
        self.num_appen = 4

    def provider_worker(self, line, image_dir, caminfo):
        img_name, pose_raw = self.data_module.io.parse_line_annot(line)
        img = self.data_module.io.read_image(os.path.join(image_dir, img_name))
        img_crop_resize, resce = self.data_module.ops.crop_resize_pca(
            img, pose_raw, caminfo)
        resce3 = resce[0:4]
        pose_pca = self.data_module.ops.raw_to_pca(pose_raw, resce3)
        index = self.data_module.io.imagename2index(img_name)
        return (index, np.expand_dims(img_crop_resize, axis=-1),
                pose_pca.flatten().T, resce)

    def yanker(self, pose_local, resce, caminfo):
        resce3 = resce[0:4]
        return self.data_module.ops.pca_to_raw(pose_local, resce3)

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
        # pose_trans = cube.transform_to_center(points3)
        # ax.scatter(
        #     pose_trans[:, 0], pose_trans[:, 1], pose_trans[:, 2],
        #     color=Color('lightsteelblue').rgb)
        # mpplot.show()
        # sys.exit()

        with h5py.File(self.appen_train, 'r') as h5file:
            store_size = h5file['index'].shape[0]
            frame_id = np.random.choice(store_size)
            # frame_id = 651  # palm
            img_id = h5file['index'][frame_id, 0]
            frame_h5 = np.squeeze(h5file['frame'][frame_id, ...], -1)
            poses_h5 = h5file['poses'][frame_id, ...].reshape(-1, 3)
            resce_h5 = h5file['resce'][frame_id, ...]

        print('[{}] drawing image #{:d} ...'.format(self.name_desc, img_id))
        print(np.min(frame_h5), np.max(frame_h5))
        print(np.histogram(frame_h5, range=(1e-4, np.max(frame_h5))))
        print(np.min(poses_h5, axis=0), np.max(poses_h5, axis=0))
        from colour import Color
        colors = [Color('orange').rgb, Color('red').rgb, Color('lime').rgb]
        mpplot.subplots(nrows=2, ncols=2, figsize=(2 * 5, 2 * 5))

        ax = mpplot.subplot(2, 2, 3)
        mpplot.gca().set_title('test storage read')
        resce3 = resce_h5[0:4]
        cube = iso_cube()
        cube.load(resce3)
        # need to maintain both image and poses at the same scale
        sizel = np.floor(resce3[0]).astype(int)
        ax.imshow(
            cv2resize(frame_h5, (sizel, sizel)),
            cmap='bone')
        pose3d = cube.trans_scale_to(poses_h5)
        pose2d, _ = cube.project_pca(pose3d, roll=0, sort=False)
        pose2d *= sizel
        args.data_draw.draw_pose2d(
            thedata,
            pose2d,
        )

        ax = mpplot.subplot(2, 2, 4)
        mpplot.gca().set_title('test output')
        img_name = args.data_io.index2imagename(img_id)
        img = args.data_io.read_image(os.path.join(self.image_dir, img_name))
        ax.imshow(img, cmap='bone')
        pose_raw = self.yanker(poses_h5, resce_h5, self.caminfo)
        args.data_draw.draw_pose2d(
            thedata,
            args.data_ops.raw_to_2d(pose_raw, thedata)
        )
        rects = cube.proj_rects_3(
            args.data_ops.raw_to_2d, self.caminfo
        )
        for ii, rect in enumerate(rects):
            rect.draw(colors[ii])

        ax = mpplot.subplot(2, 2, 1)
        mpplot.gca().set_title('test input')
        annot_line = args.data_io.get_line(
            thedata.training_annot_cleaned, img_id)
        img_name, pose_raw = args.data_io.parse_line_annot(annot_line)
        img = args.data_io.read_image(os.path.join(self.image_dir, img_name))
        ax.imshow(img, cmap='bone')
        args.data_draw.draw_pose2d(
            thedata,
            args.data_ops.raw_to_2d(pose_raw, thedata))

        ax = mpplot.subplot(2, 2, 2)
        mpplot.gca().set_title('test storage write')
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
        resce3 = resce[0:4]
        cube = iso_cube()
        cube.load(resce3)
        sizel = np.floor(resce3[0]).astype(int)
        ax.imshow(
            cv2resize(frame, (sizel, sizel)),
            cmap='bone')
        pose3d = cube.trans_scale_to(poses)
        pose2d, _ = cube.project_pca(pose3d, roll=0, sort=False)
        pose2d *= sizel
        args.data_draw.draw_pose2d(
            thedata,
            pose2d,
        )

        mpplot.savefig(os.path.join(
            args.predict_dir,
            'draw_{}.png'.format(self.name_desc)))
        if self.args.show_draw:
            mpplot.show()
        print('[{}] drawing image #{:d} - done.'.format(
            self.name_desc, img_id))
