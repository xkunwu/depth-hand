""" Hand in Depth
    https://github.com/xkunwu/depth-hand
"""
import os
from importlib import import_module
import numpy as np
from collections import namedtuple
import tensorflow as tf
from .base_conv3 import base_conv3
from utils.iso_boxes import iso_cube
from utils.regu_grid import regu_grid


class voxel_detect(base_conv3):
    """ basic 3d detection based method
    """
    @staticmethod
    def get_trainer(args, new_log):
        from train.train_voxel_detect import train_voxel_detect
        return train_voxel_detect(args, new_log)

    def __init__(self, args):
        super(voxel_detect, self).__init__(args)
        self.batch_allot = getattr(
            import_module('model.batch_allot'),
            'batch_vxhit'
        )
        # self.num_appen = 4
        self.crop_size = 64
        self.hmap_size = 32
        self.map_scale = self.crop_size / self.hmap_size

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
            store_handle['vxhit'][self.batch_beg:batch_end, ...],
            axis=-1)
        # self.batch_data['batch_poses'] = \
        #     store_handle['pose_hit'][self.batch_beg:batch_end, ...]
        self.batch_data['batch_poses'] = \
            store_handle['pose_lab'][
                self.batch_beg:batch_end, ...].astype(np.int32)
        self.batch_data['batch_index'] = \
            store_handle['index'][self.batch_beg:batch_end, ...]
        self.batch_data['batch_resce'] = \
            store_handle['resce'][self.batch_beg:batch_end, ...]
        self.batch_beg = batch_end
        return self.batch_data

    def receive_data(self, thedata, args):
        """ Receive parameters specific to the data """
        super(voxel_detect, self).receive_data(thedata, args)
        thedata.hmap_size = self.hmap_size
        self.out_dim = (self.hmap_size ** 3) * self.join_num
        self.store_name = {
            'index': thedata.annotation,
            'poses': thedata.annotation,
            'resce': thedata.annotation,
            'clean': 'clean_{}'.format(self.crop_size),
            # 'pose_hit': 'pose_hit_{}'.format(self.hmap_size),
            'pose_lab': 'pose_lab_{}'.format(self.hmap_size),
            'vxhit': 'vxhit_{}'.format(self.crop_size),
        }
        self.frame_type = 'clean'

    # def write_pred(self, fanno, caminfo,
    #                batch_index, batch_resce, pred):
    #     for ii in range(batch_index.shape[0]):
    #         img_name = self.args.data_io.index2imagename(batch_index[ii, 0])
    #         resce = batch_resce[ii, :]
    #         vxmap = pred[ii, ...]
    #         vxhit = np.argmax(vxmap, axis=0)  # convert to label
    #         pose_raw = self.yanker_hmap(
    #             resce, np.array(vxhit),
    #             self.hmap_size, caminfo)
    #         fanno.write(
    #             img_name +
    #             '\t' + '\t'.join("%.4f" % x for x in pose_raw.flatten()) +
    #             '\n')

    def yanker_hmap(self, pose_lab, resce, caminfo):
        cube = iso_cube()
        cube.load(resce)
        return self.data_module.ops.vxlab_to_raw(
            pose_lab, cube, caminfo)

    def evaluate_batch(self, pred_val):
        batch_index = self.batch_data['batch_index']
        batch_resce = self.batch_data['batch_resce']
        batch_poses = np.argmax(pred_val, axis=1)  # convert to label
        num_elem = batch_index.shape[0]
        poses_out = np.empty((num_elem, self.join_num * 3))
        for ei, resce, pose_lab in zip(range(num_elem), batch_resce, batch_poses):
            pose_raw = self.yanker_hmap(pose_lab, resce, self.caminfo)
            poses_out[ei] = pose_raw.reshape(1, -1)
        self.eval_pred.append(poses_out)

    def draw_random(self, thedata, args):
        import matplotlib.pyplot as mpplot
        from mpl_toolkits.mplot3d import Axes3D
        from mayavi import mlab

        # mode = 'train'
        mode = 'test'
        store_handle = self.store_handle[mode]
        index_h5 = store_handle['index']
        store_size = index_h5.shape[0]
        frame_id = np.random.choice(store_size)
        # frame_id = 0  # frame_id = img_id - 1
        # frame_id = 239
        img_id = index_h5[frame_id, ...]
        frame_h5 = store_handle['vxhit'][frame_id, ...]
        poses_h5 = store_handle['poses'][frame_id, ...].reshape(-1, 3)
        pose_lab_h5 = store_handle['pose_lab'][frame_id, ...]
        resce_h5 = store_handle['resce'][frame_id, ...]

        print('[{}] drawing image #{:d} ...'.format(self.name_desc, img_id))
        print(np.min(frame_h5), np.max(frame_h5))
        print(np.histogram(frame_h5, range=(1e-4, np.max(frame_h5))))
        print(resce_h5)
        resce3 = resce_h5[0:4]
        cube = iso_cube()
        cube.load(resce3)
        cube.show_dims()
        img_name = args.data_io.index2imagename(img_id)
        img = args.data_io.read_image(self.data_inst.images_join(img_name, mode))
        from colour import Color
        colors = [Color('orange').rgb, Color('red').rgb, Color('lime').rgb]
        fig, _ = mpplot.subplots(nrows=2, ncols=4, figsize=(4 * 5, 2 * 5))
        voxize_crop = self.crop_size
        voxize_hmap = self.hmap_size
        scale = self.map_scale

        ax = mpplot.subplot(2, 4, 3, projection='3d')
        points3 = args.data_ops.img_to_raw(img, self.caminfo)
        points3_trans = cube.pick(points3)
        # points3_trans = cube.transform_to_center(points3_trans)
        numpts = points3_trans.shape[0]
        if 1000 < numpts:
            points3_trans = points3_trans[
                np.random.choice(numpts, 1000, replace=False), :]
        ax.scatter(
            points3_trans[:, 0], points3_trans[:, 1], points3_trans[:, 2],
            color=Color('lightsteelblue').rgb)
        args.data_draw.draw_raw3d_pose(ax, thedata, poses_h5)
        corners = cube.get_corners()
        # corners = cube.transform_to_center(corners)
        cube.draw_cube_wire(ax, corners)
        ax.view_init(azim=-90, elev=-75)

        ax = mpplot.subplot(2, 4, 1)
        ax.imshow(img, cmap=mpplot.cm.bone_r)
        pose_raw = poses_h5
        args.data_draw.draw_pose2d(
            ax, thedata,
            args.data_ops.raw_to_2d(pose_raw, self.caminfo)
        )
        rects = cube.proj_rects_3(
            args.data_ops.raw_to_2d, self.caminfo
        )
        for ii, rect in enumerate(rects):
            rect.draw(ax, colors[ii])

        ax = mpplot.subplot(2, 4, 2, projection='3d')
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

        ax = mpplot.subplot(2, 4, 4, projection='3d')
        # grid = regu_grid()
        # grid.from_cube(cube, self.crop_size)
        # grid.draw_map(ax, frame_h5)
        args.data_draw.draw_raw3d_pose(ax, thedata, pose_raw)
        # from cv2 import resize as cv2resize
        # import cv2
        # img_hmap = cv2resize(img, (241, 241))
        # img_hmap = (img_hmap / np.max(img_hmap)) * 255
        # # fig2d, _ = mpplot.subplots()
        # # ax2d = mpplot.subplot(1, 1, 1)
        # # ax2d.imshow(img)
        # # img_hmap = fig2data(fig2d)
        # # img_hmap = (255. * img_hmap / np.max(img_hmap)).astype(int)
        # # img_hmap = img_hmap / np.max(img_hmap)
        # img_hmap = cv2.cvtColor(img_hmap, cv2.COLOR_GRAY2RGB)
        # # x, y = np.ogrid[0:cube.sidelen, 0:cube.sidelen]
        # x, y = np.mgrid[-cube.sidelen:cube.sidelen, -cube.sidelen:cube.sidelen]
        # ax.plot_surface(
        #     # x, y, cube.sidelen,
        #     x, y,
        #     np.ones_like(x) * cube.sidelen,
        #     rstride=2, cstride=2,
        #     facecolors=img_hmap)
        ax.view_init(azim=-90, elev=-75)
        # axlim = cube.sidelen * 0.8
        # ax.set_xlim([-axlim, axlim])
        # ax.set_ylim([-axlim, axlim])
        # ax.set_zlim([-axlim, axlim])
        # ax.set_aspect('equal', adjustable='box')
        ax.set_zlabel('depth (mm)', labelpad=15)

        pose_yank = self.yanker_hmap(pose_lab_h5, resce3, self.caminfo)
        diff = np.abs(pose_raw - pose_yank)
        print(diff)
        print(np.min(diff, axis=0), np.max(diff, axis=0))
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

        ax = mpplot.subplot(2, 4, 5)
        draw_voxel_pose(ax, pose_raw, roll=0)

        roll = 1
        ax = mpplot.subplot(2, 4, 6)
        draw_voxel_pose(ax, pose_yank, roll=roll)

        from utils.image_ops import draw_vxlab
        ax = mpplot.subplot(2, 4, 7)
        draw_vxlab(fig, ax, vxcnt_crop, pose_lab_h5, voxize_hmap, roll=0)

        ax = mpplot.subplot(2, 4, 8)
        draw_vxlab(fig, ax, vxcnt_crop, pose_lab_h5, voxize_hmap, roll=roll)

        # if not self.args.show_draw:
        #     mlab.options.offscreen = True
        # else:
        #     mlab.figure(size=(800, 800))
        #     # mlab.contour3d(frame_h5)
        #     mlab.pipeline.volume(mlab.pipeline.scalar_field(frame_h5))
        #     mlab.pipeline.image_plane_widget(
        #         mlab.pipeline.scalar_field(frame_h5),
        #         plane_orientation='z_axes',
        #         slice_index=self.crop_size / 2)
        #     np.set_printoptions(precision=4)
        #     # print(frame_h5[12:20, 12:20, 16])
        #     mlab.outline()

        # if not self.args.show_draw:
        #     mlab.options.offscreen = True
        # else:
        #     from utils.image_ops import draw_dist3
        #     vxmap = np.zeros(voxize_hmap * voxize_hmap * voxize_hmap)
        #     vxmap[pose_lab_h5[-1].astype(int)] = 1
        #     vxmap = vxmap.reshape((voxize_hmap, voxize_hmap, voxize_hmap))
        #     draw_dist3(vxmap, voxize_crop, scale)
        #     mlab.draw()
        #     mlab.savefig(os.path.join(
        #         self.predict_dir,
        #         'draw3d_{}_{}.png'.format(self.name_desc, img_id)))
        #
        fig.tight_layout()
        mpplot.savefig(os.path.join(
            self.predict_dir,
            'draw_{}_{}.png'.format(self.name_desc, img_id)))
        if self.args.show_draw:
            mpplot.show()
            # mlab.close(all=True)
        print('[{}] drawing image #{:d} - done.'.format(
            self.name_desc, img_id))

    def get_model(
            self, input_tensor, is_training,
            bn_decay, regu_scale,
            hg_repeat=2, scope=None):
        """ input_tensor: BxHxWxDxC
            out_dim: BxHxWxDxJ, where J is number of joints
        """
        end_points = {}
        self.end_point_list = []
        final_endpoint = 'hourglass_{}'.format(hg_repeat - 1)
        num_joint = self.join_num
        num_feature = 32
        num_vol = self.hmap_size * self.hmap_size * self.hmap_size

        def add_and_check_final(name, net):
            end_points[name] = net
            return name == final_endpoint

        from tensorflow.contrib import slim
        from inresnet3d import inresnet3d
        # ~/anaconda2/lib/python2.7/site-packages/tensorflow/contrib/layers/
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
                with tf.variable_scope('stage64'):
                    sc = 'stage64'
                    net = slim.conv3d(input_tensor, 8, 3)
                    net = inresnet3d.conv_maxpool(net, scope=sc)
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                    sc = 'stage32'
                    net = inresnet3d.resnet_k(
                        net, scope='stage32_res')
                    net = slim.conv3d(
                        net, num_feature, 1, scope='stage32_out')
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                for hg in range(hg_repeat):  # 32x32x32
                    sc = 'hourglass_{}'.format(hg)
                    with tf.variable_scope(sc):
                        # branch0 = inresnet3d.hourglass3d(
                        #     net, 2, scope=sc + '_hg')
                        branch0 = inresnet3d.resnet_k(
                            net, scope='_res')
                        branch_det = slim.conv3d(
                            branch0, num_joint, 1,
                            # normalizer_fn=None, activation_fn=tf.nn.softmax)
                            normalizer_fn=None, activation_fn=None)
                        branch_flat = tf.reshape(
                            branch_det,
                            [-1, num_vol, num_joint])
                        self.end_point_list.append(sc)
                        if add_and_check_final(sc, branch_flat):
                            return branch_flat, end_points
                        branch1 = slim.conv3d(
                            branch_det, num_feature, 1)
                        net = net + branch0 + branch1
        raise ValueError('final_endpoint (%s) not recognized', final_endpoint)

    def placeholder_inputs(self, batch_size=None):
        frames_tf = tf.placeholder(
            tf.float32, shape=(
                batch_size,
                self.crop_size, self.crop_size, self.crop_size,
                1))
        poses_tf = tf.placeholder(
            tf.int32, shape=(
                batch_size,
                # self.hmap_size, self.hmap_size, self.hmap_size,
                self.join_num))
        Placeholders = namedtuple("Placeholders", "frames_tf poses_tf")
        return Placeholders(frames_tf, poses_tf)

    def get_loss(self, pred, echt, end_points):
        """ simple sum-of-squares loss
            pred: BxHxWxDxJ
            echt: BxJ
        """
        loss_ce = 0
        for name, net in end_points.items():
            if not name.startswith('hourglass_'):
                continue
            echt_l = tf.unstack(echt, axis=-1)
            pred_l = tf.unstack(net, axis=-1)
            losses_vxhit = [
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=e, logits=p) for e, p in zip(echt_l, pred_l)]
            loss_ce += tf.reduce_mean(tf.add_n(losses_vxhit))
        # for name, net in end_points.items():
        #     if not name.startswith('hourglass_'):
        #         continue
        #     loss_ce += tf.nn.l2_loss(net - echt)  # already divided by 2
        loss_reg = tf.add_n(tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES))
        return loss_ce, loss_reg
