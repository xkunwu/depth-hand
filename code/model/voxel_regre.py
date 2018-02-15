import os
from importlib import import_module
import numpy as np
import tensorflow as tf
import progressbar
import h5py
from .voxel_detect import voxel_detect
from utils.iso_boxes import iso_cube
from utils.regu_grid import regu_grid


class voxel_regre(voxel_detect):
    """ 3d offset detection based method
    """
    @staticmethod
    def get_trainer(args, new_log):
        from train.train_voxel_regre import train_voxel_regre
        return train_voxel_regre(args, new_log)

    def __init__(self, args):
        super(voxel_regre, self).__init__(args)
        self.batch_allot = getattr(
            import_module('model.batch_allot'),
            'batch_allot_vxdir'
        )
        self.crop_size = 32
        self.hmap_size = 16
        self.map_scale = 2

    def provider_worker(self, line, image_dir, caminfo):
        img_name, pose_raw = self.data_module.io.parse_line_annot(line)
        img = self.data_module.io.read_image(os.path.join(image_dir, img_name))
        pcnt, resce = self.data_module.ops.voxel_hit(
            img, pose_raw, caminfo.crop_size, caminfo)
        resce3 = resce[0:4]
        cube = iso_cube()
        cube.load(resce3)
        pose_pca = self.data_module.ops.raw_to_pca(pose_raw, resce3)
        vxhit = self.data_module.ops.raw_to_vxlab(
            pose_raw, cube, self.hmap_size, caminfo
        )
        _, olmap, uomap = self.data_module.ops.raw_to_vxoff(
            pcnt, pose_raw, cube, self.hmap_size, caminfo
        )
        index = self.data_module.io.imagename2index(img_name)
        return (index, np.expand_dims(pcnt, axis=-1),
                pose_pca.flatten().T, vxhit, olmap, uomap, resce)

    def yanker(self, pose_local, resce):
        resce3 = resce[0:4]
        return self.data_module.ops.pca_to_raw(pose_local, resce3)

    def yanker_hmap(
        self, resce, vxhit, olmap, uomap, frame,
            step, caminfo):
        resce3 = resce[0:4]
        cube = iso_cube()
        cube.load(resce3)
        return self.data_module.ops.vxoff_to_raw(
            vxhit, olmap, uomap, frame, cube, step, caminfo)

    @staticmethod
    def put_worker(
        args, image_dir, model_inst,
            caminfo, data_module, batchallot):
        bi = args[0]
        line = args[1]
        index, frame, poses, vxhit, olmap, uomap, resce = \
            model_inst.provider_worker(line, image_dir, caminfo)
        batchallot.batch_index[bi, :] = index
        batchallot.batch_frame[bi, ...] = frame
        batchallot.batch_poses[bi, :] = poses
        batchallot.batch_vxhit[bi, :] = vxhit
        batchallot.batch_olmap[bi, :] = olmap
        batchallot.batch_uomap[bi, :] = uomap
        batchallot.batch_resce[bi, :] = resce

    def evaluate_batch(self, writer, pred_val):
        self.write_pred(
            writer, self.caminfo,
            self.batch_data['batch_index'], self.batch_data['batch_resce'],
            self.batch_data['batch_frame'], pred_val
        )

    def write_pred(self, fanno, caminfo,
                   batch_index, batch_resce,
                   batch_frame, batch_poses):
        num_j = self.out_dim
        for ii in range(batch_index.shape[0]):
            img_name = self.data_module.io.index2imagename(batch_index[ii, 0])
            resce = batch_resce[ii, :]
            frame = np.squeeze(batch_frame[ii, ...])
            vxhit = batch_poses[ii, ..., 0 * num_j:1 * num_j]
            olmap = batch_poses[ii, ..., 1 * num_j:2 * num_j]
            uomap = batch_poses[ii, ..., 2 * num_j:]
            pose_raw = self.yanker_hmap(
                resce, vxhit, olmap, uomap, frame,
                self.hmap_size, caminfo)
            fanno.write(
                img_name +
                '\t' + '\t'.join("%.4f" % x for x in pose_raw.flatten()) +
                '\n')

    def fetch_batch(self, fetch_size=None):
        if fetch_size is None:
            fetch_size = self.batch_size
        batch_end = self.batch_beg + fetch_size
        if batch_end >= self.store_size:
            self.batch_beg = batch_end
            batch_end = self.batch_beg + fetch_size
            self.split_end -= self.store_size
        # print(self.batch_beg, batch_end, self.split_end)
        if batch_end >= self.split_end:
            return None
        # self.batch_data = {
        #     'batch_index': self.store_file['index'][self.batch_beg:batch_end, ...],
        #     'batch_frame': self.store_file['frame'][self.batch_beg:batch_end, ...],
        #     'batch_poses': self.store_file['poses'][self.batch_beg:batch_end, ...],
        #     'batch_vxhit': self.store_file['vxhit'][self.batch_beg:batch_end, ...],
        #     'batch_olmap': self.store_file['olmap'][self.batch_beg:batch_end, ...],
        #     'batch_uomap': self.store_file['uomap'][self.batch_beg:batch_end, ...],
        #     'batch_resce': self.store_file['resce'][self.batch_beg:batch_end, ...]
        # }
        batch_vxmap = []
        # batch_vxmap.append(self.store_file['vxhit'][self.batch_beg:batch_end, ...])
        batch_vxmap.append(self.store_file['olmap'][self.batch_beg:batch_end, ...])
        batch_vxmap.append(self.store_file['uomap'][self.batch_beg:batch_end, ...])
        self.batch_data = {
            'batch_index': self.store_file['index'][self.batch_beg:batch_end, ...],
            'batch_frame': self.store_file['frame'][self.batch_beg:batch_end, ...],
            'batch_poses': self.store_file['vxhit'][self.batch_beg:batch_end, ...].astype(np.int32),
            'batch_vxmap': np.concatenate(batch_vxmap, axis=-1),
            'batch_resce': self.store_file['resce'][self.batch_beg:batch_end, ...]
        }
        self.batch_beg = batch_end
        return self.batch_data

    def prepare_data(self, thedata, args,
                     batchallot, file_annot, name_appen):
        num_line = int(sum(1 for line in file_annot))
        file_annot.seek(0)
        batchallot.allot(num_line)
        store_size = batchallot.store_size
        num_stores = int(np.ceil(float(num_line) / store_size))
        self.logger.debug(
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
        crop_size = self.crop_size
        hmap_size = self.hmap_size
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
                    crop_size, crop_size, crop_size,
                    num_channel),
                chunks=(1,
                        crop_size, crop_size, crop_size,
                        num_channel),
                compression='lzf',
                # dtype=np.float32)
                dtype=float)
            h5file.create_dataset(
                'poses',
                (num_line, out_dim * 3),
                compression='lzf',
                # dtype=np.float32)
                dtype=float)
            h5file.create_dataset(
                'vxhit',
                (num_line, out_dim),
                compression='lzf',
                # dtype=np.float32)
                dtype=float)
            h5file.create_dataset(
                'olmap',
                (num_line, hmap_size, hmap_size, hmap_size, out_dim),
                compression='lzf',
                # dtype=np.float32)
                dtype=float)
            h5file.create_dataset(
                'uomap',
                (num_line, hmap_size, hmap_size, hmap_size, out_dim * 3),
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
                h5file['vxhit'][store_beg:store_beg + resline, ...] = \
                    batchallot.batch_vxhit[0:resline, ...]
                h5file['olmap'][store_beg:store_beg + resline, ...] = \
                    batchallot.batch_olmap[0:resline, ...]
                h5file['uomap'][store_beg:store_beg + resline, ...] = \
                    batchallot.batch_uomap[0:resline, ...]
                h5file['resce'][store_beg:store_beg + resline, ...] = \
                    batchallot.batch_resce[0:resline, ...]
                timerbar.update(bi)
                bi += 1
                store_beg += resline
        timerbar.finish()

    def draw_random(self, thedata, args):
        import matplotlib.pyplot as mpplot
        from mpl_toolkits.mplot3d import Axes3D
        from mayavi import mlab

        with h5py.File(self.appen_train, 'r') as h5file:
            store_size = h5file['index'].shape[0]
            frame_id = np.random.choice(store_size)
            # frame_id = 651
            img_id = h5file['index'][frame_id, 0]
            frame_h5 = np.squeeze(h5file['frame'][frame_id, ...], -1)
            poses_h5 = h5file['poses'][frame_id, ...].reshape(-1, 3)
            vxhit_h5 = h5file['vxhit'][frame_id, ...]
            olmap_h5 = h5file['olmap'][frame_id, ...]
            uomap_h5 = h5file['uomap'][frame_id, ...]
            resce_h5 = h5file['resce'][frame_id, ...]

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
        voxize_crop = self.crop_size
        voxize_hmap = self.hmap_size
        joint_id = self.out_dim - 1
        vol_shape = (voxize_hmap, voxize_hmap, voxize_hmap)
        vxhit_l = []
        for x, y, z in np.array(np.unravel_index(
                vxhit_h5.astype(int), vol_shape)).T:
            vm = np.zeros(vol_shape)
            vm[x, y, z] = 1
            vxhit_l.append(vm)
        vxhit = np.stack(vxhit_l, axis=3)
        olmap = olmap_h5[..., joint_id]
        uomap = uomap_h5[..., 3 * joint_id:3 * (joint_id + 1)]

        ax = mpplot.subplot(2, 3, 1)
        ax.imshow(img, cmap=mpplot.cm.bone_r)
        pose_raw = self.yanker(poses_h5, resce_h5)
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
            resce_h5, vxhit, olmap_h5, uomap_h5,
            frame_h5, self.hmap_size, self.caminfo)
        err_re = np.sum(np.abs(pose_out - cube.transform_add_center(poses_h5)))
        if 1e-2 < err_re:
            print('ERROR: reprojection error: {}'.format(err_re))
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

        ax = mpplot.subplot(2, 3, 2)
        draw_voxel_pose(ax, pose_raw, roll=0)

        ax = mpplot.subplot(2, 3, 3)
        draw_voxel_pose(ax, pose_out, roll=0)

        from utils.image_ops import draw_vxlab, draw_uomap3d, draw_vxmap
        ax = mpplot.subplot(2, 3, 4)
        draw_vxlab(fig, ax, vxcnt_crop, vxhit_h5, voxize_hmap, roll=0)
        ax = mpplot.subplot(2, 3, 5)
        draw_vxmap(fig, ax, vxcnt_crop, olmap, voxize_hmap, reduce_fn=np.max, roll=0)
        ax = mpplot.subplot(2, 3, 6, projection='3d')
        draw_uomap3d(fig, ax, frame_h5, uomap)

        if not self.args.show_draw:
            mlab.options.offscreen = True
        # should reverser y-axis
        mlab.figure(
            bgcolor=(1, 1, 1), fgcolor=(0., 0., 0.),
            size=(800, 800))
        xx, yy, zz = np.where(1e-2 < frame_h5)
        yy = 63 - yy
        mlab.points3d(
            xx, yy, zz,
            mode="cube", opacity=0.5,
            color=Color('khaki').rgb,
            scale_factor=0.9)
        xx, yy, zz = np.mgrid[0:63:2, 0:63:2, 0:63:2]
        yy = 63 - yy
        mlab.quiver3d(
            xx, yy, zz,
            uomap[..., 0], -uomap[..., 1], uomap[..., 2],
            mode="arrow",
            color=Color('red').rgb,
            line_width=8, scale_factor=2)
        mlab.gcf().scene.parallel_projection = True
        mlab.view(0, 0)
        mlab.gcf().scene.camera.zoom(1.5)
        # mlab.outline()
        mlab.draw()
        mlab.savefig(os.path.join(
            args.predict_dir,
            'draw3d_{}_{}.png'.format(self.name_desc, img_id)))
        if not self.args.show_draw:
            mlab.close()

        mpplot.savefig(os.path.join(
            args.predict_dir,
            # 'draw_{}.png'.format(self.name_desc)))
            'draw_{}_{}.png'.format(self.name_desc, img_id)))
        if self.args.show_draw:
            mpplot.show()
            mlab.close()
        print('[{}] drawing image #{:d} - done.'.format(
            self.name_desc, img_id))

    def get_model(
            self, input_tensor, is_training, bn_decay,
            hg_repeat=1, scope=None):
        """ input_tensor: BxHxWxDxC
            out_dim: BxHxWxDx(J*5), where J is number of joints
        """
        end_points = {}
        self.end_point_list = []
        final_endpoint = 'hourglass_{}'.format(hg_repeat - 1)
        num_joint = self.out_dim
        num_feature = 128
        num_vol = self.hmap_size * self.hmap_size * self.hmap_size
        vol_shape = (self.hmap_size, self.hmap_size, self.hmap_size)

        def add_and_check_final(name, net):
            end_points[name] = net
            return name == final_endpoint

        from tensorflow.contrib import slim
        from inresnet3d import inresnet3d
        # ~/anaconda2/lib/python2.7/site-packages/tensorflow/contrib/layers/
        with tf.variable_scope(
                scope, self.name_desc, [input_tensor]):
            weight_decay = 0.00004
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
                    weights_regularizer=slim.l2_regularizer(weight_decay),
                    biases_regularizer=slim.l2_regularizer(weight_decay),
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm), \
                slim.arg_scope(
                    [slim.max_pool3d, slim.avg_pool3d],
                    stride=2, padding='SAME'), \
                slim.arg_scope(
                    [slim.conv3d_transpose],
                    stride=2, padding='SAME',
                    weights_regularizer=slim.l2_regularizer(weight_decay),
                    biases_regularizer=slim.l2_regularizer(weight_decay),
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm), \
                slim.arg_scope(
                    [slim.conv3d],
                    stride=1, padding='SAME',
                    weights_regularizer=slim.l2_regularizer(weight_decay),
                    biases_regularizer=slim.l2_regularizer(weight_decay),
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
                    net = inresnet3d.resnet_k(
                        input_tensor, scope='stage32_residual')
                    net = slim.conv3d(
                        net, num_feature, 1, scope='stage32_out')
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                for hg in range(hg_repeat):
                    sc = 'hourglass_{}'.format(hg)
                    with tf.variable_scope(sc):
                        branch0 = inresnet3d.hourglass3d(
                            net, 2, scope=sc + '_hg')
                        branch0 = inresnet3d.resnet_k(
                            branch0, scope='_res')
                        branch_hit = slim.conv3d(
                            branch0, num_joint, 1,
                            # normalizer_fn=None, activation_fn=tf.nn.softmax)
                            normalizer_fn=None, activation_fn=None)
                        branch_olm = slim.conv3d(
                            branch0, num_joint, 1,
                            # normalizer_fn=None, activation_fn=tf.nn.relu)
                            normalizer_fn=None, activation_fn=None)
                        branch_uom = slim.conv3d(
                            branch0, num_joint * 3, 1,
                            # normalizer_fn=None, activation_fn=tf.nn.sigmoid)
                            normalizer_fn=None, activation_fn=None)
                        net_maps = tf.concat(
                            [branch_hit, branch_olm, branch_uom],
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
            tf.int32, shape=(
                batch_size,
                self.out_dim))
        vxmap_tf = tf.placeholder(
            tf.float32, shape=(
                batch_size,
                self.hmap_size, self.hmap_size, self.hmap_size,
                self.out_dim * 4))
        return frames_tf, poses_tf, vxmap_tf

    def get_loss(self, pred, echt, vxmap, end_points):
        """ simple sum-of-squares loss
            pred: BxHxWxDxJ
            echt: BxJ
        """
        # num_j = self.out_dim
        # num_vol = self.hmap_size * self.hmap_size * self.hmap_size
        # vol_shape = (self.hmap_size, self.hmap_size, self.hmap_size)
        num_j = 21
        num_vol = 32768
        vol_shape = (32, 32, 32)
        loss = 0
        batch_size = pred.shape[0]
        for name, net in end_points.items():
            if not name.startswith('hourglass_'):
                continue
            vxhit_pred = net[..., :num_j]
            vxmap_pred = net[..., num_j:]
            vxlab_pred = tf.reshape(
                vxhit_pred, [-1, num_vol, num_j])
            echt_l = tf.unstack(echt, axis=-1)
            pred_l = tf.unstack(vxlab_pred, axis=-1)
            losses_vxhit = [
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=e, logits=p) for e, p in zip(echt_l, pred_l)]
            loss += tf.reduce_sum(tf.add_n(losses_vxhit))
            loss += tf.nn.l2_loss(vxmap_pred - vxmap)
            uomap_pred = tf.reshape(
                vxmap_pred[..., num_j:],
                [batch_size, vol_shape, num_j, 3])
            loss_unit = tf.sum(uomap_pred ** 2, axis=-1)
            loss_unit = tf.sum(tf.abs(1 - loss_unit))
        losses_reg = tf.add_n(tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES))
        return loss + losses_reg
