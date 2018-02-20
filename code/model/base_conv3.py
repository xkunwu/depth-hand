import os
from importlib import import_module
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from model.base_regre import base_regre
from utils.iso_boxes import iso_cube
from utils.regu_grid import regu_grid


class base_conv3(base_regre):
    """ This class holds baseline training approach using 3d CNN.
    """
    def __init__(self, args):
        super(base_conv3, self).__init__(args)
        self.batch_allot = getattr(
            import_module('model.batch_allot'),
            'batch_conv3'
        )
        self.net_rank = 3
        self.crop_size = 32
        # self.num_appen = 4

    def fetch_batch(self, fetch_size=None):
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
        self.batch_data['batch_frame'] = np.expand_dims(
            self.store_handle['pcnt3'][self.batch_beg:batch_end, ...],
            axis=-1)
        self.batch_data['batch_poses'] = \
            self.store_handle['pose_c'][self.batch_beg:batch_end, ...]
        self.batch_data['batch_index'] = \
            self.store_handle['index'][self.batch_beg:batch_end, ...]
        self.batch_data['batch_resce'] = \
            self.store_handle['resce'][self.batch_beg:batch_end, ...]
        self.batch_beg = batch_end
        return self.batch_data

    def receive_data(self, thedata, args):
        """ Receive parameters specific to the data """
        super(base_conv3, self).receive_data(thedata, args)
        self.store_name = {
            'index': self.train_file,
            'poses': self.train_file,
            'resce': self.train_file,
            'pose_c': os.path.join(self.prepare_dir, 'pose_c'),
            'pcnt3': os.path.join(
                self.prepare_dir, 'pcnt3_{}'.format(self.crop_size)),
        }
        self.store_precon = {
            'index': [],
            'poses': [],
            'resce': [],
            'pose_c': ['poses', 'resce'],
            'pcnt3': ['index', 'resce'],
        }

    def draw_random(self, thedata, args):
        import matplotlib.pyplot as mpplot
        from mpl_toolkits.mplot3d import Axes3D
        from mayavi import mlab

        # mlab.figure(size=(800, 800))
        # # cube = iso_cube()
        # # points3_trans = np.hstack(
        # #     (np.zeros((10, 2)), np.arange(-1, 1, 0.2).reshape(10, 1)))
        # # grid = regu_grid()
        # # grid.from_cube(cube, 6)
        # # pcnt = grid.fill(points3_trans)
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

        index_h5 = self.store_handle['index']
        store_size = index_h5.shape[0]
        frame_id = np.random.choice(store_size)
        # frame_id = 0
        img_id = index_h5[frame_id, ...]
        frame_h5 = self.store_handle['pcnt3'][frame_id, ...]
        poses_h5 = self.store_handle['pose_c'][frame_id, ...].reshape(-1, 3)
        resce_h5 = self.store_handle['resce'][frame_id, ...]

        print('[{}] drawing image #{:d} ...'.format(self.name_desc, img_id))
        print(np.min(frame_h5), np.max(frame_h5))
        print(np.histogram(frame_h5, range=(1e-4, np.max(frame_h5))))
        from colour import Color
        colors = [Color('orange').rgb, Color('red').rgb, Color('lime').rgb]
        resce3 = resce_h5[0:4]
        cube = iso_cube()
        cube.load(resce3)
        mpplot.subplots(nrows=2, ncols=2, figsize=(2 * 5, 2 * 5))

        ax = mpplot.subplot(2, 2, 1)
        img_name = args.data_io.index2imagename(img_id)
        img = args.data_io.read_image(os.path.join(self.image_dir, img_name))
        ax.imshow(img, cmap=mpplot.cm.bone_r)
        pose_raw = self.yanker(poses_h5, resce_h5, self.caminfo)
        args.data_draw.draw_pose2d(
            ax, thedata,
            args.data_ops.raw_to_2d(pose_raw, self.caminfo)
        )
        rects = cube.proj_rects_3(
            args.data_ops.raw_to_2d, self.caminfo
        )
        for ii, rect in enumerate(rects):
            rect.draw(ax, colors[ii])

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
        args.data_draw.draw_raw3d_pose(ax, thedata, poses_h5)
        corners = cube.transform_to_center(cube.get_corners())
        cube.draw_cube_wire(ax, corners)
        ax.view_init(azim=-120, elev=-150)

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
        ax.view_init(azim=-90, elev=-75)
        ax.set_zlabel('depth (mm)', labelpad=15)
        args.data_draw.draw_raw3d_pose(ax, thedata, pose_raw)
        corners = cube.get_corners()
        iso_cube.draw_cube_wire(ax, corners)

        voxize_crop = self.crop_size
        grid = regu_grid()
        grid.from_cube(cube, voxize_crop)
        vxcnt_crop = frame_h5

        def draw_voxel_pose(ax, poses, roll=0):
            pose3d = cube.transform_center_shrink(poses)
            pose2d, _ = cube.project_ortho(pose3d, roll=roll, sort=False)
            pose2d *= voxize_crop
            args.data_draw.draw_pose2d(
                ax, thedata,
                pose2d,
            )
            coord = grid.slice_ortho(vxcnt_crop, roll=roll)
            grid.draw_slice(ax, coord, 1.)
            ax.set_xlim([0, voxize_crop])
            ax.set_ylim([0, voxize_crop])
            ax.set_aspect('equal', adjustable='box')
            ax.invert_yaxis()

        ax = mpplot.subplot(2, 2, 4)
        draw_voxel_pose(ax, pose_raw, roll=0)

        # if not self.args.show_draw:
        #     mlab.options.offscreen = True
        #     mlab.figure(size=(800, 800))
        #     points3_trans = cube.transform_to_center(points3_sam)
        #     mlab.points3d(
        #         points3_trans[:, 0], points3_trans[:, 1], points3_trans[:, 2],
        #         scale_factor=8,
        #         color=Color('lightsteelblue').rgb)
        #     mlab.outline()

        if not self.args.show_draw:
            mlab.options.offscreen = True
        else:
            mlab.figure(size=(800, 800))
            # mlab.contour3d(frame)
            mlab.pipeline.volume(mlab.pipeline.scalar_field(frame_h5))
            mlab.pipeline.image_plane_widget(
                mlab.pipeline.scalar_field(frame_h5),
                plane_orientation='z_axes',
                slice_index=self.crop_size / 2)
            np.set_printoptions(precision=4)
            # print(frame[12:20, 12:20, 16])
            mlab.outline()

        mpplot.savefig(os.path.join(
            args.predict_dir,
            'draw_{}_{}.png'.format(self.name_desc, img_id)))
        if self.args.show_draw:
            mpplot.show()
            mlab.close()
        print('[{}] drawing image #{:d} - done.'.format(
            self.name_desc, img_id))

    def get_model(
            self, input_tensor, is_training,
            bn_decay, regu_scale,
            scope=None, final_endpoint='stage_out'):
        """ frames_tf: BxHxWxDx1
            out_dim: Bx(Jx3), where J is number of joints
        """
        end_points = {}
        self.end_point_list = []

        def add_and_check_final(name, net):
            end_points[name] = net
            return name == final_endpoint

        from inresnet3d import inresnet3d
        with tf.variable_scope(
                scope, self.name_desc, [input_tensor]):
            bn_epsilon = 0.001
            with \
                slim.arg_scope(
                    [slim.batch_norm],
                    is_training=is_training,
                    epsilon=bn_epsilon,
                    # # Make sure updates happen automatically
                    # updates_collections=None,
                    # Try zero_debias_moving_mean=True for improved stability.
                    # zero_debias_moving_mean=True,
                    decay=bn_decay), \
                slim.arg_scope(
                    [slim.dropout],
                    is_training=is_training), \
                slim.arg_scope(
                    [slim.fully_connected],
                    weights_regularizer=slim.l2_regularizer(regu_scale),
                    biases_regularizer=slim.l2_regularizer(regu_scale),
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm), \
                slim.arg_scope(
                    [slim.max_pool3d, slim.avg_pool3d],
                    stride=2, padding='SAME'), \
                slim.arg_scope(
                    [slim.conv3d_transpose],
                    stride=2, padding='SAME',
                    weights_regularizer=slim.l2_regularizer(regu_scale),
                    biases_regularizer=slim.l2_regularizer(regu_scale),
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm), \
                slim.arg_scope(
                    [slim.conv3d],
                    stride=1, padding='SAME',
                    weights_regularizer=slim.l2_regularizer(regu_scale),
                    biases_regularizer=slim.l2_regularizer(regu_scale),
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm):
                # with tf.variable_scope('stage64'):
                #     sc = 'stage64'
                #     net = slim.conv3d(input_tensor, 16, 3)
                #     # net = slim.max_pool3d(net, 3)
                #     net = inresnet3d.conv_maxpool(net, scope=sc)
                #     self.end_point_list.append(sc)
                #     if add_and_check_final(sc, net):
                #         return net, end_points
                with tf.variable_scope('stage32'):
                    sc = 'stage32'
                    net = slim.conv3d(input_tensor, 16, 3)
                    net = inresnet3d.conv_maxpool(net, scope=sc)
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                with tf.variable_scope('stage16'):
                    sc = 'stage16'
                    net = inresnet3d.conv_maxpool(net, scope=sc)
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                with tf.variable_scope('stage8'):
                    sc = 'stage_out'
                    net = inresnet3d.pullout8(
                        net, self.out_dim, is_training,
                        scope=sc)
                    # net = slim.conv3d(net, 128, 1, scope='reduce4')
                    # self.end_point_list.append('reduce4')
                    # if add_and_check_final('reduce4', net):
                    #     return net, end_points
                    # net = slim.conv3d(
                    #     net, 1024, net.get_shape()[1:4],
                    #     padding='VALID', scope='fullconn4')
                    # self.end_point_list.append('fullconn4')
                    # if add_and_check_final('fullconn4', net):
                    #     return net, end_points
                    # net = slim.dropout(
                    #     net, 0.5, scope='dropout4')
                    # net = slim.flatten(net)
                    # net = slim.fully_connected(
                    #     net, self.out_dim,
                    #     activation_fn=None, normalizer_fn=None,
                    #     scope='output4')
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
