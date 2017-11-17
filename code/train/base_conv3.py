import tensorflow as tf
import os
import sys
from importlib import import_module
from psutil import virtual_memory
import numpy as np
import progressbar
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
regu_grid = getattr(
    import_module('regu_grid'),
    'regu_grid'
)


class base_conv3(base_regre):
    """ This class holds baseline training approach using 3d CNN.
    """
    def __init__(self):
        super(base_conv3, self).__init__()
        self.crop_size = 32
        self.num_appen = 9

    class batch_allot:
        def __init__(self, batch_size, image_size, pose_dim, num_channel, num_appen):
            self.batch_size = batch_size
            self.image_size = image_size
            self.pose_dim = pose_dim
            self.num_channel = num_channel
            self.num_appen = num_appen
            batch_data = {
                'batch_index': np.empty(
                    shape=(batch_size, 1), dtype=np.int32),
                'batch_frame': np.empty(
                    shape=(
                        batch_size,
                        image_size, image_size, image_size,
                        num_channel),
                    dtype=np.float32),
                'batch_poses': np.empty(
                    shape=(batch_size, pose_dim),
                    dtype=np.float32),
                'batch_resce': np.empty(
                    shape=(batch_size, num_appen),
                    dtype=np.float32)
            }
            self.batch_bytes = \
                batch_data['batch_index'].nbytes + batch_data['batch_frame'].nbytes + \
                batch_data['batch_poses'].nbytes + batch_data['batch_resce'].nbytes
            self.batch_beg = 0

        def allot(self, store_size=-1):
            store_cap_mult = (virtual_memory().total >> 2) // self.batch_bytes
            store_cap = store_cap_mult * self.batch_size
            if 0 > store_size:
                self.store_size = store_cap
            else:
                self.store_size = min(store_cap, store_size)
            self.store_bytes = self.store_size * self.batch_bytes / self.batch_size
            self.store_beg = 0
            self.batch_index = np.empty(
                shape=(self.store_size, 1), dtype=np.int32)
            self.batch_frame = np.empty(
                shape=(
                    self.store_size,
                    self.image_size, self.image_size, self.image_size,
                    self.num_channel),
                dtype=np.float32)
            self.batch_poses = np.empty(
                shape=(self.store_size, self.pose_dim), dtype=np.float32)
            self.batch_resce = np.empty(
                shape=(self.store_size, self.num_appen), dtype=np.float32)

        def fetch_store(self):
            if self.store_beg >= self.file_size:
                return False
            store_end = min(
                self.store_beg + self.store_size,
                self.file_size
            )
            self.store_size = store_end - self.store_beg
            self.batch_index = self.store_file['index'][self.store_beg:store_end, ...]
            self.batch_frame = self.store_file['frame'][self.store_beg:store_end, ...]
            self.batch_poses = self.store_file['poses'][self.store_beg:store_end, ...]
            self.batch_resce = self.store_file['resce'][self.store_beg:store_end, ...]
            self.store_beg = store_end
            self.batch_beg = 0
            return True

        def assign(self, store_file):
            self.store_file = store_file
            self.file_size = self.store_file['index'].shape[0]
            self.store_size = min(
                self.file_size,
                ((virtual_memory().total >> 1) // self.batch_bytes) * self.batch_size
            )
            self.store_beg = 0
            self.fetch_store()

        def fetch_batch(self):
            batch_end = self.batch_beg + self.batch_size
            if batch_end >= self.store_size:
                if not self.fetch_store():
                    return None
                batch_end = self.batch_beg + self.batch_size
                if batch_end >= self.store_size:
                    return None
            batch_data = {
                'batch_index': self.batch_index[self.batch_beg:batch_end, ...],
                'batch_frame': self.batch_frame[self.batch_beg:batch_end, ...],
                'batch_poses': self.batch_poses[self.batch_beg:batch_end, ...],
                'batch_resce': self.batch_resce[self.batch_beg:batch_end, ...]
            }
            self.batch_beg = batch_end
            return batch_data

    def prepare_data(self, thedata, batchallot, file_annot, name_appen):
        num_line = int(sum(1 for line in file_annot))
        file_annot.seek(0)
        batchallot.allot(num_line)
        store_size = batchallot.store_size
        num_stores = int(np.ceil(float(num_line) / store_size))
        print('[{}] preparing data: {:d} lines \n\
              (producing {:.4f} GB for store size {:d}) ...'.format(
            self.__class__.__name__, num_line,
            float(batchallot.store_bytes) / (2 << 30), store_size
        ))
        timerbar = progressbar.ProgressBar(
            maxval=num_stores,
            widgets=[
                progressbar.Percentage(),
                ' ', progressbar.Bar('=', '[', ']'),
                ' ', progressbar.ETA()]
        ).start()
        image_size = self.crop_size
        pose_dim = batchallot.pose_dim
        num_channel = batchallot.num_channel
        num_appen = batchallot.num_appen
        with h5py.File(os.path.join(self.prep_dir, name_appen), 'w') as h5file:
            h5file.create_dataset(
                'index',
                (num_line, 1),
                compression='lzf',
                dtype=np.int32
            )
            h5file.create_dataset(
                'frame',
                (num_line,
                    image_size, image_size, image_size,
                    num_channel),
                chunks=(1,
                        image_size, image_size, image_size,
                        num_channel),
                compression='lzf',
                dtype=np.float32
            )
            h5file.create_dataset(
                'poses',
                (num_line, pose_dim),
                compression='lzf',
                dtype=np.float32
            )
            h5file.create_dataset(
                'resce',
                (num_line, num_appen),
                compression='lzf',
                dtype=np.float32
            )
            bi = 0
            store_beg = 0
            while True:
                resline = self.provider.puttensor_mt(
                    file_annot, self.provider_worker,
                    self.image_dir, thedata, batchallot
                )
                if 0 > resline:
                    break
                h5file['index'][store_beg:store_beg + resline, ...] = \
                    batchallot.batch_index[0:resline, ...]
                h5file['frame'][store_beg:store_beg + resline, ...] = \
                    batchallot.batch_frame[0:resline, ...]
                h5file['poses'][store_beg:store_beg + resline, ...] = \
                    batchallot.batch_poses[0:resline, ...]
                h5file['resce'][store_beg:store_beg + resline, ...] = \
                    batchallot.batch_resce[0:resline, ...]
                timerbar.update(bi)
                bi += 1
                store_beg += resline
        timerbar.finish()

    def receive_data(self, thedata, args):
        """ Receive parameters specific to the data """
        super(base_conv3, self).receive_data(thedata, args)
        self.provider_worker = args.data_provider.prow_conv3d
        self.yanker = self.provider.yank_conv3d

    def draw_random(self, thedata, args):
        import matplotlib.pyplot as mpplot
        from colour import Color
        from mayavi import mlab

        # mlab.figure(size=(800, 800))
        # # cube = iso_cube()
        # # points3_trans = np.hstack(
        # #     (np.zeros((10, 2)), np.arange(-1, 1, 0.2).reshape(10, 1)))
        # # grid = regu_grid()
        # # grid.from_cube(cube, 6)
        # # grid.fill(points3_trans)
        # # pcnt = grid.pcnt
        #
        # pcnt = np.zeros((6, 6, 6))
        # pcnt[2:4, 2:4, 3] = 1
        # frame = args.data_ops.prop_dist(pcnt)
        # mlab.pipeline.volume(mlab.pipeline.scalar_field(frame))
        # mlab.pipeline.image_plane_widget(
        #     mlab.pipeline.scalar_field(frame),
        #     plane_orientation='z_axes',
        #     slice_index=self.crop_size / 2)
        # print(pcnt[..., 3])
        # print(frame[..., 3])
        # print(frame[0, 0, 3], type(frame[0, 0, 3]))
        # mlab.outline()
        # mlab.show()
        # sys.exit()

        with h5py.File(os.path.join(self.prep_dir, self.appen_train), 'r') as h5file:
            store_size = h5file['index'].shape[0]
            frame_id = np.random.choice(store_size)
            img_id = h5file['index'][frame_id, 0]
            frame_h5 = np.squeeze(h5file['frame'][frame_id, ...], -1)
            poses_h5 = h5file['poses'][frame_id, ...].reshape(-1, 3)
            resce_h5 = h5file['resce'][frame_id, ...]

        print('[{}] drawing pose #{:d}'.format(self.__class__.__name__, img_id))
        fig_size = (2 * 5, 2 * 5)
        resce3 = resce_h5[0:8]
        cube = iso_cube()
        cube.load(resce3)
        mpplot.subplots(nrows=2, ncols=2, figsize=fig_size)
        mpplot.subplot(2, 2, 1)
        annot_line = args.data_io.get_line(
            thedata.training_annot_cleaned, img_id)
        img_name, pose_raw = args.data_io.parse_line_annot(annot_line)
        img = args.data_io.read_image(os.path.join(self.image_dir, img_name))
        mpplot.imshow(img, cmap='bone')
        args.data_draw.draw_pose2d(
            thedata, img,
            args.data_ops.raw_to_2d(pose_raw, thedata))

        ax = mpplot.subplot(2, 2, 2, projection='3d')
        annot_line = args.data_io.get_line(
            thedata.training_annot_cleaned, img_id)
        img_name, pose_raw = args.data_io.parse_line_annot(annot_line)
        img = args.data_io.read_image(os.path.join(self.image_dir, img_name))
        points3 = args.data_ops.img_to_raw(img, thedata)
        numpts = points3.shape[0]
        if 1000 < numpts:
            samid = np.random.choice(numpts, 1000, replace=False)
            points3_sam = points3[samid, :]
        else:
            points3_sam = points3
        ax.scatter(
            points3_sam[:, 0], points3_sam[:, 1], points3_sam[:, 2],
            color=Color('lightsteelblue').rgb)
        ax.view_init(azim=-90, elev=-60)
        ax.set_zlabel('depth (mm)', labelpad=15)
        args.data_draw.draw_raw3d_pose(thedata, pose_raw)
        corners = cube.get_corners()
        corners = cube.transform_inv(corners)
        iso_cube.draw_cube_wire(corners)

        # mlab.figure(size=(800, 800))
        # points3_trans = cube.transform(points3_sam)
        # mlab.points3d(
        #     points3_trans[:, 0], points3_trans[:, 1], points3_trans[:, 2],
        #     scale_factor=8,
        #     color=Color('lightsteelblue').rgb)
        # mlab.outline()

        ax = mpplot.subplot(2, 2, 3, projection='3d')
        _, points3_trans = cube.pick(points3)
        numpts = points3_trans.shape[0]
        if 1000 < numpts:
            points3_trans = points3_trans[np.random.choice(numpts, 1000, replace=False), :]
        pose_trans = cube.transform(pose_raw)
        ax.scatter(
            points3_trans[:, 0], points3_trans[:, 1], points3_trans[:, 2],
            color=Color('lightsteelblue').rgb)
        args.data_draw.draw_raw3d_pose(thedata, pose_trans)
        corners = cube.get_corners()
        cube.draw_cube_wire(corners)
        ax.view_init(azim=-120, elev=-150)

        ax = mpplot.subplot(2, 2, 4)
        img_name = args.data_io.index2imagename(img_id)
        img = args.data_io.read_image(os.path.join(self.image_dir, img_name))
        mpplot.imshow(img, cmap='bone')
        pose_raw = self.yanker(poses_h5, resce_h5)
        args.data_draw.draw_pose2d(
            thedata, img,
            args.data_ops.raw_to_2d(pose_raw, thedata)
        )

        mlab.figure(size=(800, 800))
        img_name, frame, poses, resce = self.provider_worker(
            annot_line, self.image_dir, thedata)
        frame = np.squeeze(frame, axis=-1)
        poses = poses.reshape(-1, 3)
        if (
                # (1e-4 < np.linalg.norm(frame_h5 - frame)) or
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
        resce3 = resce_h5[0:8]
        cube = iso_cube()
        cube.load(resce3)
        # mlab.contour3d(frame)
        mlab.pipeline.volume(mlab.pipeline.scalar_field(frame))
        mlab.pipeline.image_plane_widget(
            mlab.pipeline.scalar_field(frame),
            plane_orientation='z_axes',
            slice_index=self.crop_size / 2)
        np.set_printoptions(precision=4)
        # print(frame[12:20, 12:20, 16])
        mlab.outline()
        mpplot.savefig(os.path.join(
            args.data_inst.predict_dir,
            'draw_{}.png'.format(self.__class__.__name__)))
        mpplot.show()

    def placeholder_inputs(self):
        frames_tf = tf.placeholder(
            tf.float32, shape=(
                self.batch_size,
                self.crop_size, self.crop_size, self.crop_size,
                1))
        poses_tf = tf.placeholder(
            tf.float32, shape=(self.batch_size, self.pose_dim))
        return frames_tf, poses_tf

    def get_model(self, frames_tf, is_training, bn_decay=None):
        """ directly predict all joints' location using regression
            frames_tf: BxHxWxDx1
            pose_dim: BxJ, where J is flattened 3D locations
        """
        batch_size = frames_tf.get_shape()[0].value
        end_points = {}
        input_image = frames_tf

        net = tf_util.conv3d(
            input_image, 32, [5, 5, 5],
            padding='VALID', stride=[1, 1, 1],
            bn=True, is_training=is_training,
            scope='conv1', bn_decay=bn_decay)
        net = tf_util.max_pool3d(
            net, [2, 2, 2],
            padding='VALID', scope='maxpool1')
        net = tf_util.conv3d(
            net, 64, [3, 3, 3],
            padding='VALID', stride=[1, 1, 1],
            bn=True, is_training=is_training,
            scope='conv2', bn_decay=bn_decay)
        net = tf_util.max_pool3d(
            net, [2, 2, 2],
            padding='VALID', scope='maxpool2')
        net = tf_util.conv3d(
            net, 128, [3, 3, 3],
            padding='VALID', stride=[1, 1, 1],
            bn=True, is_training=is_training,
            scope='conv3', bn_decay=bn_decay)
        # net = tf_util.max_pool3d(
        #     net, [2, 2, 2],
        #     padding='VALID', scope='maxpool3')
        # print(net.shape)

        net = tf.reshape(net, [batch_size, -1])
        net = tf_util.fully_connected(
            net, 2048, bn=True, is_training=is_training,
            scope='fc1', bn_decay=bn_decay)
        net = tf_util.dropout(
            net, keep_prob=0.5, is_training=is_training,
            scope='dp1')
        # net = tf_util.fully_connected(
        #     net, 1024, bn=True, is_training=is_training,
        #     scope='fc2', bn_decay=bn_decay)
        # net = tf_util.dropout(
        #     net, keep_prob=0.5, is_training=is_training,
        #     scope='dp2')
        net = tf_util.fully_connected(
            net, self.pose_dim, activation_fn=None, scope='fc3')

        return net, end_points
