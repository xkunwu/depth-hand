import os
from importlib import import_module
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as mpplot
from model.super_edt2 import super_edt2
from utils.iso_boxes import iso_cube


class super_edt2m(super_edt2):
    def __init__(self, args):
        super(super_edt2m, self).__init__(args)

    def fetch_batch(self, mode='train', fetch_size=None):
        if fetch_size is None:
            fetch_size = self.batch_size
        batch_end = self.batch_beg + fetch_size
        # if batch_end >= self.store_size:
        #     self.batch_beg = batch_end
        #     batch_end = self.batch_beg + fetch_size
        #     self.split_end -= self.store_size
        # # print(self.batch_beg, batch_end, self.split_end)
        if batch_end >= self.split_end:
            return None
        store_handle = self.store_handle[mode]
        self.batch_data['batch_frame'] = np.expand_dims(
            store_handle['clean'][self.batch_beg:batch_end, ...],
            axis=-1)
        self.batch_data['batch_poses'] = \
            store_handle['pose_c'][self.batch_beg:batch_end, ...]
        self.batch_data['batch_edt2'] = \
            store_handle['edt2m'][self.batch_beg:batch_end, ...]
        self.batch_data['batch_index'] = \
            store_handle['index'][self.batch_beg:batch_end, ...]
        self.batch_data['batch_resce'] = \
            store_handle['resce'][self.batch_beg:batch_end, ...]
        self.batch_beg = batch_end
        return self.batch_data

    def receive_data(self, thedata, args):
        """ Receive parameters specific to the data """
        super(super_edt2m, self).receive_data(thedata, args)
        thedata.hmap_size = self.hmap_size
        self.out_dim = self.join_num * 3
        self.store_name = {
            'index': thedata.annotation,
            'poses': thedata.annotation,
            'resce': thedata.annotation,
            'pose_c': 'pose_c',
            'clean': 'clean_{}'.format(self.crop_size),
            'edt2': 'edt2_{}'.format(self.hmap_size),
            'udir2': 'udir2_{}'.format(self.hmap_size),
            'edt2m': 'edt2m_{}'.format(self.hmap_size),
        }
        self.frame_type = 'clean'

    def draw_random(self, thedata, args):
        # mode = 'train'
        mode = 'test'
        store_handle = self.store_handle[mode]
        index_h5 = store_handle['index']
        store_size = index_h5.shape[0]
        frame_id = np.random.choice(store_size)
        # frame_id = 892  # frame_id = img_id - 1
        # frame_id = 886  # frame_id = img_id - 1
        # frame_id = 218  # palm
        # frame_id = 30
        # frame_id = 848
        # frame_id = 239
        # frame_id = 26
        img_id = index_h5[frame_id, ...]
        frame_h5 = store_handle['clean'][frame_id, ...]
        poses_h5 = store_handle['pose_c'][frame_id, ...].reshape(-1, 3)
        resce_h5 = store_handle['resce'][frame_id, ...]
        edt2_h5 = store_handle['edt2m'][frame_id, ...]

        print('[{}] drawing image #{:d} ...'.format(self.name_desc, img_id))
        print(np.min(frame_h5), np.max(frame_h5))
        print(np.histogram(frame_h5, range=(1e-4, np.max(frame_h5))))
        print(np.min(poses_h5, axis=0), np.max(poses_h5, axis=0))
        from colour import Color
        colors = [Color('orange').rgb, Color('red').rgb, Color('lime').rgb]
        fig, _ = mpplot.subplots(nrows=2, ncols=2, figsize=(2 * 5, 2 * 5))

        ax = mpplot.subplot(2, 2, 2)
        mpplot.gca().set_title('test storage read')
        resce3 = resce_h5[0:4]
        cube = iso_cube()
        cube.load(resce3)
        # draw_pose_pred(
        #     fig, ax, frame_h5, poses_h5, resce_h5,
        #     args.data_draw.draw_pose2d, thedata)
        ax.imshow(frame_h5, cmap=mpplot.cm.bone_r)
        # pose3d = poses_h5
        pose3d = cube.trans_scale_to(poses_h5)
        pose2d, _ = cube.project_ortho(pose3d, roll=0, sort=False)
        pose2d *= self.crop_size
        args.data_draw.draw_pose2d(
            ax, thedata,
            pose2d,
        )

        ax = mpplot.subplot(2, 2, 1)
        mpplot.gca().set_title('test image - {:d}'.format(img_id))
        img_name = args.data_io.index2imagename(img_id)
        img = args.data_io.read_image(self.data_inst.images_join(img_name, mode))
        ax.imshow(img, cmap=mpplot.cm.bone_r)
        pose_raw = self.yanker(poses_h5, resce_h5, self.caminfo)
        args.data_draw.draw_pose2d(
            ax, thedata,
            args.data_ops.raw_to_2d(pose_raw, thedata)
        )
        rects = cube.proj_rects_3(
            args.data_ops.raw_to_2d, self.caminfo
        )
        for ii, rect in enumerate(rects):
            rect.draw(ax, colors[ii])

        from utils.image_ops import draw_edt2
        ax = mpplot.subplot(2, 2, 3)
        joint_id = self.join_num - 1
        edt2 = edt2_h5[..., joint_id]
        draw_edt2(fig, ax, edt2)

        ax = mpplot.subplot(2, 2, 4)
        joint_id = self.join_num - 1 - 9
        edt2 = edt2_h5[..., joint_id]
        draw_edt2(fig, ax, edt2)

        fig.tight_layout()
        mpplot.savefig(os.path.join(
            self.predict_dir,
            'draw_{}_{}.png'.format(self.name_desc, img_id)))
        if self.args.show_draw:
            mpplot.show()
        print('[{}] drawing image #{:d} - done.'.format(
            self.name_desc, img_id))
