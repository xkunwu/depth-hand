import os
import sys
from importlib import import_module
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import progressbar
import h5py
from model.base_regre import base_regre
from model.batch_allot import batch_allot_conv3

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
regu_grid = getattr(
    import_module('utils.regu_grid'),
    'regu_grid'
)


class base_conv3(base_regre):
    """ This class holds baseline training approach using 3d CNN.
    """
    def __init__(self, args):
        super(base_conv3, self).__init__(args)
        self.batch_allot = batch_allot_conv3
        self.net_rank = 3
        self.crop_size = 32
        self.num_appen = 4

    def prepare_data(self, thedata, args,
                     batchallot, file_annot, name_appen):
        num_line = int(sum(1 for line in file_annot))
        file_annot.seek(0)
        batchallot.allot(num_line)
        store_size = batchallot.store_size
        num_stores = int(np.ceil(float(num_line) / store_size))
        args.logger.debug(
            'preparing data [{}]: {:d} lines (producing {:.4f} GB for store size {:d}) ...'.format(
                self.__class__.__name__, num_line,
                float(batchallot.store_bytes) / (2 << 30),
                store_size))
        timerbar = progressbar.ProgressBar(
            maxval=num_stores,
            widgets=[
                progressbar.Percentage(),
                ' ', progressbar.Bar('=', '[', ']'),
                ' ', progressbar.ETA()]
        ).start()
        image_size = self.crop_size
        out_dim = self.out_dim
        num_channel = self.num_channel
        num_appen = self.num_appen
        with h5py.File(os.path.join(self.prepare_dir, name_appen), 'w') as h5file:
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
                # dtype=np.float32)
                dtype=float)
            h5file.create_dataset(
                'poses',
                (num_line, out_dim),
                compression='lzf',
                # dtype=np.float32)
                dtype=float)
            h5file.create_dataset(
                'resce',
                (num_line, num_appen),
                compression='lzf',
                # dtype=np.float32)
                dtype=float)
            bi = 0
            store_beg = 0
            while True:
                resline = self.data_module.provider.puttensor_mt(
                    file_annot, self.put_worker, self.image_dir,
                    self, thedata, self.data_module, batchallot
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

    def provider_worker(self, line, image_dir, caminfo):
        img_name, pose_raw = self.data_module.io.parse_line_annot(line)
        img = self.data_module.io.read_image(os.path.join(image_dir, img_name))
        pcnt, resce = self.data_module.ops.fill_grid(
            img, pose_raw, caminfo.crop_size, caminfo)
        resce3 = resce[0:4]
        pose_pca = self.data_module.ops.raw_to_pca(pose_raw, resce3)
        index = self.data_module.io.imagename2index(img_name)
        return (index, np.expand_dims(pcnt, axis=-1),
                pose_pca.flatten().T, resce)

    def yanker(self, pose_local, resce, caminfo):
        resce3 = resce[0:4]
        return self.data_module.ops.pca_to_raw(pose_local, resce3)

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

        with h5py.File(self.appen_train, 'r') as h5file:
            store_size = h5file['index'].shape[0]
            frame_id = np.random.choice(store_size)
            img_id = h5file['index'][frame_id, 0]
            frame_h5 = np.squeeze(h5file['frame'][frame_id, ...], -1)
            poses_h5 = h5file['poses'][frame_id, ...].reshape(-1, 3)
            resce_h5 = h5file['resce'][frame_id, ...]

        print('[{}] drawing image #{:d} ...'.format(self.name_desc, img_id))
        print(np.min(frame_h5), np.max(frame_h5))
        print(np.histogram(frame_h5, range=(1e-4, np.max(frame_h5))))
        colors = [Color('orange').rgb, Color('red').rgb, Color('lime').rgb]
        mpplot.subplots(nrows=2, ncols=2, figsize=(2 * 5, 2 * 5))
        mpplot.subplot(2, 2, 1)
        annot_line = args.data_io.get_line(
            thedata.training_annot_cleaned, img_id)
        img_name, pose_raw = args.data_io.parse_line_annot(annot_line)
        img = args.data_io.read_image(os.path.join(self.image_dir, img_name))
        mpplot.imshow(img, cmap='bone')
        args.data_draw.draw_pose2d(
            thedata,
            args.data_ops.raw_to_2d(pose_raw, self.caminfo))

        ax = mpplot.subplot(2, 2, 3, projection='3d')
        resce3 = resce_h5[0:4]
        cube = iso_cube()
        cube.load(resce3)
        cube.show_dims()
        points3 = args.data_ops.img_to_raw(img, self.caminfo)
        points3_trans = cube.pick(points3)
        points3_trans = cube.transform_to_center(points3_trans)
        numpts = points3_trans.shape[0]
        if 1000 < numpts:
            points3_trans = points3_trans[
                np.random.choice(numpts, 1000, replace=False), :]
        ax.scatter(
            points3_trans[:, 0], points3_trans[:, 1], points3_trans[:, 2],
            color=Color('lightsteelblue').rgb)
        args.data_draw.draw_raw3d_pose(thedata, poses_h5)
        corners = cube.transform_to_center(cube.get_corners())
        cube.draw_cube_wire(corners)
        ax.view_init(azim=-120, elev=-150)

        ax = mpplot.subplot(2, 2, 4)
        img_name = args.data_io.index2imagename(img_id)
        img = args.data_io.read_image(os.path.join(self.image_dir, img_name))
        mpplot.imshow(img, cmap='bone')
        pose_raw = self.yanker(poses_h5, resce_h5, self.caminfo)
        args.data_draw.draw_pose2d(
            thedata,
            args.data_ops.raw_to_2d(pose_raw, self.caminfo)
        )
        rects = cube.proj_rects_3(
            args.data_ops.raw_to_2d, self.caminfo
        )
        for ii, rect in enumerate(rects):
            rect.draw(colors[ii])

        ax = mpplot.subplot(2, 2, 2, projection='3d')
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
        iso_cube.draw_cube_wire(corners)

        # mlab.figure(size=(800, 800))
        # points3_trans = cube.transform_to_center(points3_sam)
        # mlab.points3d(
        #     points3_trans[:, 0], points3_trans[:, 1], points3_trans[:, 2],
        #     scale_factor=8,
        #     color=Color('lightsteelblue').rgb)
        # mlab.outline()

        img_name, frame, poses, resce = self.provider_worker(
            annot_line, self.image_dir, self.caminfo)
        frame = np.squeeze(frame, axis=-1)
        poses = poses.reshape(-1, 3)
        if (
                (1e-4 < np.linalg.norm(frame_h5 - frame)) or
                (1e-4 < np.linalg.norm(poses_h5 - poses))
        ):
            print(np.linalg.norm(frame_h5 - frame))
            print(np.linalg.norm(poses_h5 - poses))
            _, frame_1, _, _ = self.provider_worker(
                annot_line, self.image_dir, self.caminfo)
            print(np.linalg.norm(frame_1 - frame))
            with h5py.File('/tmp/111', 'w') as h5file:
                h5file.create_dataset(
                    'frame', data=frame_1, dtype=np.float32
                )
            with h5py.File('/tmp/111', 'r') as h5file:
                frame_2 = h5file['frame'][:]
                print(np.linalg.norm(frame_1 - frame_2))
            print('ERROR - h5 storage corrupted!')
        resce3 = resce_h5[0:4]
        cube = iso_cube()
        cube.load(resce3)
        cube.show_dims()

        # mlab.figure(size=(800, 800))
        # # mlab.contour3d(frame)
        # mlab.pipeline.volume(mlab.pipeline.scalar_field(frame))
        # mlab.pipeline.image_plane_widget(
        #     mlab.pipeline.scalar_field(frame),
        #     plane_orientation='z_axes',
        #     slice_index=self.crop_size / 2)
        # np.set_printoptions(precision=4)
        # # print(frame[12:20, 12:20, 16])
        # mlab.outline()

        mpplot.savefig(os.path.join(
            args.predict_dir,
            'draw_{}.png'.format(self.name_desc)))
        mpplot.show()
        print('[{}] drawing image #{:d} - done.'.format(
            self.name_desc, img_id))

    def get_model(
            self, input_tensor, is_training, bn_decay,
            scope=None, final_endpoint='stage_out'):
        """ frames_tf: BxHxWxDx1
            out_dim: BxJ, where J is flattened 3D locations
        """
        end_points = {}
        self.end_point_list = []

        def add_and_check_final(name, net):
            end_points[name] = net
            return name == final_endpoint

        with tf.variable_scope(
                scope, self.name_desc, [input_tensor]):
            with slim.arg_scope(
                    [slim.batch_norm, slim.dropout],
                    is_training=is_training), \
                slim.arg_scope(
                    [slim.fully_connected],
                    weights_regularizer=slim.l2_regularizer(0.00004),
                    biases_regularizer=slim.l2_regularizer(0.00004),
                    activation_fn=None, normalizer_fn=None), \
                slim.arg_scope(
                    [slim.max_pool3d, slim.avg_pool3d],
                    stride=1, padding='SAME'), \
                slim.arg_scope(
                    [slim.conv3d],
                    stride=1, padding='SAME',
                    activation_fn=tf.nn.relu,
                    weights_regularizer=slim.l2_regularizer(0.00004),
                    biases_regularizer=slim.l2_regularizer(0.00004),
                    normalizer_fn=slim.batch_norm):
                with tf.variable_scope('stage32'):
                    sc = 'stage32'
                    net = slim.conv3d(
                        input_tensor, 32, 3, scope='conv32_3x3x3_1')
                    net = slim.conv3d(
                        net, 32, 3, stride=2, scope='conv32_3x3x3_2')
                    net = slim.max_pool3d(
                        net, 3, scope='maxpool32_3x3x3_1')
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                with tf.variable_scope('stage16'):
                    sc = 'stage16'
                    net = slim.conv3d(
                        net, 64, 3, scope='conv16_3x3x3_1')
                    net = slim.max_pool3d(
                        net, 3, stride=2, scope='maxpool16_3x3x3_2')
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                with tf.variable_scope('stage8'):
                    sc = 'stage8'
                    net = slim.conv3d(
                        net, 128, 3, scope='conv16_3x3x3_1')
                    net = slim.max_pool3d(
                        net, 3, stride=2, scope='maxpool16_3x3x3_2')
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                with tf.variable_scope('stage4'):
                    sc = 'stage_out'
                    net = slim.conv3d(net, 128, 1, scope='reduce4')
                    self.end_point_list.append('reduce4')
                    if add_and_check_final('reduce4', net):
                        return net, end_points
                    net = slim.conv3d(
                        net, 1024, net.get_shape()[1:4],
                        padding='VALID', scope='fullconn4')
                    self.end_point_list.append('fullconn4')
                    if add_and_check_final('fullconn4', net):
                        return net, end_points
                    net = slim.dropout(
                        net, 0.5, scope='dropout4')
                    net = slim.flatten(net)
                    net = slim.fully_connected(
                        net, self.out_dim, scope='output4')
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points

        raise ValueError('final_endpoint (%s) not recognized', final_endpoint)

    def placeholder_inputs(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        frames_tf = tf.placeholder(
            tf.float32, shape=(
                batch_size,
                self.crop_size, self.crop_size, self.crop_size,
                1))
        poses_tf = tf.placeholder(
            tf.float32, shape=(batch_size, self.out_dim))
        return frames_tf, poses_tf
