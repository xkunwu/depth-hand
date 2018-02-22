import os
from importlib import import_module
import numpy as np
import tensorflow as tf
from .base_conv3 import base_conv3
from utils.iso_boxes import iso_cube
from utils.regu_grid import regu_grid


class voxel_offset(base_conv3):
    """ 3d offset detection based method
    """
    @staticmethod
    def get_trainer(args, new_log):
        from train.train_voxel_offset import train_voxel_offset
        return train_voxel_offset(args, new_log)

    def __init__(self, args):
        super(voxel_offset, self).__init__(args)
        self.batch_allot = getattr(
            import_module('model.batch_allot'),
            'batch_vxudir'
        )
        self.crop_size = 32
        self.hmap_size = 16
        self.map_scale = self.crop_size / self.hmap_size

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
            self.store_handle['vxudir'][self.batch_beg:batch_end, ...]
        self.batch_data['batch_index'] = \
            self.store_handle['index'][self.batch_beg:batch_end, ...]
        self.batch_data['batch_resce'] = \
            self.store_handle['resce'][self.batch_beg:batch_end, ...]
        self.batch_beg = batch_end
        return self.batch_data

    def receive_data(self, thedata, args):
        """ Receive parameters specific to the data """
        super(voxel_offset, self).receive_data(thedata, args)
        thedata.hmap_size = self.hmap_size
        self.out_dim = (self.hmap_size ** 3) * self.join_num * 4
        self.store_name = {
            'index': self.train_file,
            'poses': self.train_file,
            'resce': self.train_file,
            'pcnt3': os.path.join(
                self.prepare_dir, 'pcnt3_{}'.format(self.crop_size)),
            # 'vxoff': os.path.join(
            #     self.prepare_dir, 'vxoff_{}'.format(self.hmap_size)),
            'vxudir': os.path.join(
                self.prepare_dir, 'vxudir_{}'.format(self.hmap_size)),
        }
        self.store_precon = {
            'index': [],
            'poses': [],
            'resce': [],
            'pcnt3': ['index', 'resce'],
            # 'vxoff': ['pcnt3', 'poses', 'resce'],
            'vxudir': ['pcnt3', 'poses', 'resce'],
        }

    def yanker_hmap(self, resce, vxhit, vxudir, caminfo):
        cube = iso_cube()
        cube.load(resce)
        return self.data_module.ops.vxudir_to_raw(
            vxhit, vxudir, cube, caminfo)

    def evaluate_batch(self, pred_val):
        batch_index = self.batch_data['batch_index']
        batch_resce = self.batch_data['batch_resce']
        batch_frame = self.batch_data['batch_frame']
        num_elem = batch_index.shape[0]
        poses_out = np.empty((num_elem, self.join_num * 3))
        for ei in range(num_elem):
            pose_raw = self.yanker_hmap(
                batch_resce[ei, ...],
                batch_frame[ei, ...],
                pred_val[ei, ...],
                self.caminfo)
            poses_out[ei] = pose_raw.reshape(1, -1)
        self.eval_pred.append(poses_out)

    def draw_random(self, thedata, args):
        import matplotlib.pyplot as mpplot
        from mpl_toolkits.mplot3d import Axes3D
        from mayavi import mlab

        index_h5 = self.store_handle['index']
        store_size = index_h5.shape[0]
        frame_id = np.random.choice(store_size)
        # frame_id = 0  # frame_id = img_id - 1
        img_id = index_h5[frame_id, ...]
        frame_h5 = self.store_handle['pcnt3'][frame_id, ...]
        poses_h5 = self.store_handle['poses'][frame_id, ...].reshape(-1, 3)
        resce_h5 = self.store_handle['resce'][frame_id, ...]
        vxudir_h5 = self.store_handle['vxudir'][frame_id, ...]
        print(self.store_handle['vxudir'])

        print('[{}] drawing image #{:d} ...'.format(self.name_desc, img_id))
        print(np.min(frame_h5), np.max(frame_h5))
        print(np.histogram(frame_h5, range=(1e-4, np.max(frame_h5))))
        print(np.min(poses_h5, axis=0), np.max(poses_h5, axis=0))
        print(resce_h5)
        resce3 = resce_h5[0:4]
        cube = iso_cube()
        cube.load(resce3)
        cube.show_dims()
        img_name = args.data_io.index2imagename(img_id)
        img = args.data_io.read_image(os.path.join(self.image_dir, img_name))
        from colour import Color
        colors = [Color('orange').rgb, Color('red').rgb, Color('lime').rgb]
        fig, _ = mpplot.subplots(nrows=2, ncols=3, figsize=(3 * 5, 2 * 5))
        vxcnt_crop = frame_h5
        voxize_crop = self.crop_size
        voxize_hmap = self.hmap_size
        scale = self.map_scale
        num_joint = self.join_num
        joint_id = num_joint - 1
        vxdist = vxudir_h5[..., joint_id]
        vxunit = vxudir_h5[..., num_joint + 3 * joint_id:num_joint + 3 * (joint_id + 1)]

        ax = mpplot.subplot(2, 3, 1)
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

        pose_out = self.yanker_hmap(
            resce_h5, vxcnt_crop, vxudir_h5, self.caminfo)
        err_re = np.sum(np.abs(pose_out - pose_raw))
        if 1e-2 < err_re:
            print('ERROR: reprojection error: {}'.format(err_re))
        else:
            print('reprojection looks good: {}'.format(err_re))
        grid = regu_grid()
        grid.from_cube(cube, voxize_crop)

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

        ax = mpplot.subplot(2, 3, 2)
        draw_voxel_pose(ax, pose_raw, roll=0)

        from utils.image_ops import draw_vxmap, draw_uomap3d
        # ax = mpplot.subplot(2, 3, 4, projection='3d')
        ax = mpplot.subplot(2, 3, 3)
        draw_uomap3d(fig, ax, vxcnt_crop, vxunit[voxize_hmap / 2, ...])

        ax = mpplot.subplot(2, 3, 4)
        draw_vxmap(fig, ax, vxcnt_crop, vxdist, voxize_hmap, reduce_fn=np.max, roll=0)
        ax = mpplot.subplot(2, 3, 5)
        draw_vxmap(fig, ax, vxcnt_crop, vxdist, voxize_hmap, reduce_fn=np.max, roll=1)
        ax = mpplot.subplot(2, 3, 6)
        draw_vxmap(fig, ax, vxcnt_crop, vxdist, voxize_hmap, reduce_fn=np.max, roll=2)

        if not self.args.show_draw:
            mlab.options.offscreen = True
        else:
            from utils.image_ops import draw_udir
            draw_udir(vxdist, vxunit, voxize_crop, scale)
            mlab.draw()
            mlab.savefig(os.path.join(
                args.predict_dir,
                'draw3d_{}_{}.png'.format(self.name_desc, img_id)))
            # joint_id = 0
            # vxdist = vxudir_h5[..., joint_id]
            # vxunit = vxudir_h5[..., num_joint + 3 * joint_id:num_joint + 3 * (joint_id + 1)]
            # draw_udir(vxdist, vxunit, voxize_crop, scale)
            # mlab.draw()

        mpplot.savefig(os.path.join(
            args.predict_dir,
            # 'draw_{}.png'.format(self.name_desc)))
            'draw_{}_{}.png'.format(self.name_desc, img_id)))
        if self.args.show_draw:
            mpplot.show()
            mlab.close(all=True)
        print('[{}] drawing image #{:d} - done.'.format(
            self.name_desc, img_id))

    def get_model(
            self, input_tensor, is_training,
            bn_decay, regu_scale,
            hg_repeat=2, scope=None):
        """ input_tensor: BxHxWxDxC
            out_dim: BxHxWxDx(J*4), where J is number of joints
        """
        end_points = {}
        self.end_point_list = []
        final_endpoint = 'hourglass_{}'.format(hg_repeat - 1)
        num_joint = self.join_num
        num_feature = 96

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
                    # sc = 'stage64'
                    # net = slim.conv3d(input_tensor, 16, 3)
                    # net = inresnet3d.conv_maxpool(net, scope=sc)
                    # self.end_point_list.append(sc)
                    # if add_and_check_final(sc, net):
                    #     return net, end_points
                    sc = 'stage32'
                    net = slim.conv3d(input_tensor, 16, 3)
                    # net = inresnet3d.resnet_k(
                    #     net, scope='stage32_res')
                    net = inresnet3d.conv_maxpool(net, scope=sc)
                    net = slim.conv3d(
                        net, num_feature, 1, scope='stage32_out')
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                for hg in range(hg_repeat):
                    sc = 'hourglass_{}'.format(hg)
                    with tf.variable_scope(sc):
                        # branch0 = inresnet3d.hourglass3d(
                        #     net, 2, scope=sc + '_hg')
                        branch0 = inresnet3d.resnet_k(
                            net, scope='_res')
                        branch_olm = slim.conv3d(
                            branch0, num_joint, 1,
                            # normalizer_fn=None, activation_fn=tf.nn.relu)
                            normalizer_fn=None, activation_fn=None)
                        branch_uom = slim.conv3d(
                            branch0, num_joint * 3, 1,
                            # normalizer_fn=None, activation_fn=tf.nn.sigmoid)
                            normalizer_fn=None, activation_fn=None)
                        net_maps = tf.concat(
                            [branch_olm, branch_uom],
                            axis=-1)
                        self.end_point_list.append(sc)
                        if add_and_check_final(sc, net_maps):
                            # flat_soft = tf.reshape(branch_hit, [-1, num_vol, num_joint])
                            # flat_soft = tf.nn.softmax(flat_soft, dim=1)
                            # branch_hit = tf.reshape(flat_soft, [-1, vol_shape, num_joint])
                            # net_maps = tf.concat(
                            #     [branch_hit, branch_olm, branch_uom],
                            #     axis=-1)
                            return net_maps, end_points
                        branch1 = slim.conv3d(
                            net_maps, num_feature, 1)
                        net = net + branch0 + branch1
        raise ValueError('final_endpoint (%s) not recognized', final_endpoint)

    def placeholder_inputs(self, batch_size=None):
        frames_tf = tf.placeholder(
            tf.float32, shape=(
                batch_size,
                self.crop_size, self.crop_size, self.crop_size,
                1))
        poses_tf = tf.placeholder(
            tf.float32, shape=(
                batch_size,
                self.hmap_size, self.hmap_size, self.hmap_size,
                self.join_num * 4))
        return frames_tf, poses_tf

    @staticmethod
    def smooth_l1(xa):
        return tf.where(
            1 < xa,
            xa - 0.5,
            0.5 * (xa ** 2)
        )

    def get_loss(self, pred, echt, end_points):
        """ simple sum-of-squares loss
            pred: BxHxWxDxJ
            echt: BxJ
        """
        num_j = self.join_num
        loss_udir = 0
        loss_unit = 0
        for name, net in end_points.items():
            if not name.startswith('hourglass_'):
                continue
            # loss_udir += tf.nn.l2_loss(net - vxudir)
            loss_udir += tf.reduce_sum(
                self.smooth_l1(tf.abs(net - echt)))
            vxunit_pred = tf.reshape(
                net[..., num_j:],
                (-1, 3))
            loss_unit += tf.reduce_sum(
                self.smooth_l1(tf.abs(
                    1 - tf.reduce_sum(vxunit_pred ** 2, axis=-1))))
        loss_reg = tf.add_n(tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES))
        return loss_udir, loss_unit, loss_reg
