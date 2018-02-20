import os
from importlib import import_module
import numpy as np
import tensorflow as tf
import progressbar
import h5py
import matplotlib.pyplot as mpplot
from cv2 import resize as cv2resize
from model.base_regre import base_regre
from utils.iso_boxes import iso_cube
from model.incept_resnet import incept_resnet
from model.hourglass import hourglass
from utils.image_ops import draw_hmap2, draw_olmap, draw_uomap


class dense_regre(base_regre):
    @staticmethod
    def get_trainer(args, new_log):
        from train.train_dense_regre import train_dense_regre
        return train_dense_regre(args, new_log)

    def __init__(self, args):
        super(dense_regre, self).__init__(args)
        self.batch_allot = getattr(
            import_module('model.batch_allot'),
            'batch_allot_hmap'
        )
        self.num_appen = 4
        self.crop_size = 64
        self.hmap_size = 16
        self.map_scale = 4

    def receive_data(self, thedata, args):
        """ Receive parameters specific to the data """
        super(dense_regre, self).receive_data(thedata, args)

    def provider_worker(self, line, image_dir, caminfo):
        img_name, pose_raw = self.data_module.io.parse_line_annot(line)
        img = self.data_module.io.read_image(os.path.join(image_dir, img_name))
        img_crop_resize, resce = self.data_module.ops.crop_resize_pca(
            img, pose_raw, caminfo, sort=True)  # have to fix order
        resce3 = resce[0:4]
        cube = iso_cube()
        cube.load(resce3)
        pose_pca = self.data_module.ops.raw_to_pca(pose_raw, resce3)
        hmap2 = self.data_module.ops.raw_to_heatmap2(
            pose_raw, cube, self.hmap_size, caminfo
        )
        _, olmap, uomap = self.data_module.ops.raw_to_offset(
            img_crop_resize, pose_raw, cube, self.hmap_size, caminfo
        )
        # _, olmap, uomap = self.data_module.ops.raw_to_offset(
        #     img, pose_raw, cube, self.hmap_size, caminfo
        # )
        index = self.data_module.io.imagename2index(img_name)
        return (index, np.expand_dims(img_crop_resize, axis=-1),
                pose_pca.flatten().T, hmap2, olmap, uomap, resce)

    def yanker(self, pose_local, resce):
        resce3 = resce[0:4]
        return self.data_module.ops.pca_to_raw(pose_local, resce3)

    def yanker_hmap(
        self, resce, hmap2, olmap, uomap, depth,
            hmap_size, caminfo):
        resce3 = resce[0:4]
        cube = iso_cube()
        cube.load(resce3)
        return self.data_module.ops.offset_to_raw(
            hmap2, olmap, uomap, depth, cube, hmap_size, caminfo)

    @staticmethod
    def put_worker(
        args, image_dir, model_inst,
            caminfo, data_module, batchallot):
        bi = args[0]
        line = args[1]
        index, frame, poses, hmap2, olmap, uomap, resce = \
            model_inst.provider_worker(line, image_dir, caminfo)
        batchallot.batch_index[bi, :] = index
        batchallot.batch_frame[bi, ...] = frame
        batchallot.batch_poses[bi, :] = poses
        batchallot.batch_hmap2[bi, :] = hmap2
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
        num_j = self.join_num
        for ii in range(batch_index.shape[0]):
            img_name = self.data_module.io.index2imagename(batch_index[ii, 0])
            resce = batch_resce[ii, :]
            depth = np.squeeze(batch_frame[ii, ...])
            hmap2 = batch_poses[ii, ..., 0 * num_j:1 * num_j]
            olmap = batch_poses[ii, ..., 1 * num_j:2 * num_j]
            uomap = batch_poses[ii, ..., 2 * num_j:]
            pose_raw = self.yanker_hmap(
                resce, hmap2, olmap, uomap, depth,
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
        #     'batch_hmap2': self.store_file['hmap2'][self.batch_beg:batch_end, ...],
        #     'batch_olmap': self.store_file['olmap'][self.batch_beg:batch_end, ...],
        #     'batch_uomap': self.store_file['uomap'][self.batch_beg:batch_end, ...],
        #     'batch_resce': self.store_file['resce'][self.batch_beg:batch_end, ...]
        # }
        batch_hmaps = []
        batch_hmaps.append(self.store_file['hmap2'][self.batch_beg:batch_end, ...])
        batch_hmaps.append(self.store_file['olmap'][self.batch_beg:batch_end, ...])
        batch_hmaps.append(self.store_file['uomap'][self.batch_beg:batch_end, ...])
        self.batch_data = {
            'batch_index': self.store_file['index'][self.batch_beg:batch_end, ...],
            'batch_frame': self.store_file['frame'][self.batch_beg:batch_end, ...],
            'batch_poses': np.concatenate(batch_hmaps, axis=-1),
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
        out_dim = self.join_num
        num_channel = self.num_channel
        num_appen = self.num_appen
        with h5py.File(os.path.join(self.prepare_dir, name_appen), 'w') as h5file:
            h5file.create_dataset(
                'index',
                (num_line, 1),
                compression='lzf',
                dtype='i4'
            )
            h5file.create_dataset(
                'frame',
                (num_line,
                    crop_size, crop_size,
                    num_channel),
                chunks=(1,
                        crop_size, crop_size,
                        num_channel),
                compression='lzf',
                # dtype='f4')
                dtype='f4')
            h5file.create_dataset(
                'poses',
                (num_line, out_dim * 3),
                compression='lzf',
                # dtype='f4')
                dtype='f4')
            h5file.create_dataset(
                'hmap2',
                (num_line, hmap_size, hmap_size, out_dim),
                compression='lzf',
                # dtype='f4')
                dtype='f4')
            h5file.create_dataset(
                'olmap',
                (num_line, hmap_size, hmap_size, out_dim),
                compression='lzf',
                # dtype='f4')
                dtype='f4')
            h5file.create_dataset(
                'uomap',
                (num_line, hmap_size, hmap_size, out_dim * 3),
                compression='lzf',
                # dtype='f4')
                dtype='f4')
            h5file.create_dataset(
                'resce',
                (num_line, num_appen),
                compression='lzf',
                # dtype='f4')
                dtype='f4')
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
                h5file['hmap2'][store_beg:store_beg + resline, ...] = \
                    batchallot.batch_hmap2[0:resline, ...]
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
        with h5py.File(self.appen_train, 'r') as h5file:
            store_size = h5file['index'].shape[0]
            frame_id = np.random.choice(store_size)
            # frame_id = 741  # showing pinky
            img_id = h5file['index'][frame_id, 0]  # img_id = frame_id + 1
            frame_h5 = np.squeeze(h5file['frame'][frame_id, ...], -1)
            poses_h5 = h5file['poses'][frame_id, ...].reshape(-1, 3)
            hmap2_h5 = h5file['hmap2'][frame_id, ...]
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
        sizel = np.floor(resce3[0]).astype(int)
        from colour import Color
        colors = [Color('orange').rgb, Color('red').rgb, Color('lime').rgb]
        fig, _ = mpplot.subplots(nrows=2, ncols=3, figsize=(3 * 5, 2 * 5))
        joint_id = self.join_num - 1
        hmap2 = hmap2_h5[..., joint_id]
        olmap = olmap_h5[..., joint_id]
        uomap = uomap_h5[..., 3 * joint_id:3 * (joint_id + 1)]
        depth_crop = cv2resize(frame_h5, (sizel, sizel))

        ax = mpplot.subplot(2, 3, 1)
        ax.imshow(depth_crop, cmap=mpplot.cm.bone_r)
        pose3d = cube.trans_scale_to(poses_h5)
        pose2d, _ = cube.project_ortho(pose3d, roll=0, sort=False)
        pose2d *= sizel
        args.data_draw.draw_pose2d(
            ax, thedata,
            pose2d,
        )

        ax = mpplot.subplot(2, 3, 4)
        draw_hmap2(fig, ax, frame_h5, hmap2)

        ax = mpplot.subplot(2, 3, 5)
        draw_uomap(fig, ax, frame_h5, uomap)

        ax = mpplot.subplot(2, 3, 6)
        draw_olmap(fig, ax, frame_h5, olmap)

        ax = mpplot.subplot(2, 3, 2)
        pose_out = self.yanker_hmap(
            resce_h5, hmap2_h5, olmap_h5, uomap_h5,
            frame_h5, self.hmap_size, self.caminfo)
        err_re = np.sum(np.abs(pose_out - cube.transform_add_center(poses_h5)))
        if 1e-2 < err_re:
            print('ERROR: reprojection error: {}'.format(err_re))
        ax.imshow(depth_crop, cmap=mpplot.cm.bone_r)
        pose3d = cube.transform_center_shrink(pose_out)
        pose2d, _ = cube.project_ortho(pose3d, roll=0, sort=False)
        pose2d *= sizel
        args.data_draw.draw_pose2d(
            ax, thedata,
            pose2d,
        )

        ax = mpplot.subplot(2, 3, 3)
        annot_line = args.data_io.get_line(
            thedata.training_annot_cleaned, img_id)
        img_name, pose_raw = args.data_io.parse_line_annot(annot_line)
        img = args.data_io.read_image(os.path.join(self.image_dir, img_name))
        ax.imshow(img, cmap=mpplot.cm.bone_r)
        args.data_draw.draw_pose2d(
            ax, thedata,
            args.data_ops.raw_to_2d(pose_raw, thedata))
        rects = cube.proj_rects_3(
            args.data_ops.raw_to_2d, self.caminfo
        )
        for ii, rect in enumerate(rects):
            rect.draw(ax, colors[ii])

        # hmap2 = self.data_module.ops.raw_to_heatmap2(
        #     pose_raw, cube, self.hmap_size, self.caminfo
        # )
        # offset, olmap, uomap = self.data_module.ops.raw_to_offset(
        #     frame_h5, pose_raw, cube, self.hmap_size, self.caminfo
        # )

        # mpplot.tight_layout()
        mpplot.savefig(os.path.join(
            args.predict_dir,
            'draw_{}_{}.png'.format(self.name_desc, img_id)))
        if self.args.show_draw:
            mpplot.show()
        print('[{}] drawing image #{:d} - done.'.format(
            self.name_desc, img_id))

    def get_model(
            self, input_tensor, is_training,
            bn_decay, regu_scale,
            hg_repeat=1, scope=None):
        """ input_tensor: BxHxWxC
            out_dim: BxHxWx(J*5), where J is number of joints
        """
        end_points = {}
        self.end_point_list = []
        final_endpoint = 'hourglass_{}'.format(hg_repeat - 1)
        num_joint = self.join_num
        # num_out_map = num_joint * 5  # hmap2, olmap, uomap
        num_feature = 128
        num_vol = self.hmap_size * self.hmap_size * self.hmap_size
        vol_shape = (self.hmap_size, self.hmap_size, self.hmap_size)

        def add_and_check_final(name, net):
            end_points[name] = net
            return name == final_endpoint

        from tensorflow.contrib import slim
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
                    [slim.max_pool2d, slim.avg_pool2d],
                    stride=2, padding='SAME'), \
                slim.arg_scope(
                    [slim.conv2d_transpose],
                    stride=2, padding='SAME',
                    weights_regularizer=slim.l2_regularizer(regu_scale),
                    biases_regularizer=slim.l2_regularizer(regu_scale),
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm), \
                slim.arg_scope(
                    [slim.conv2d],
                    stride=1, padding='SAME',
                    weights_regularizer=slim.l2_regularizer(regu_scale),
                    biases_regularizer=slim.l2_regularizer(regu_scale),
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm):
                with tf.variable_scope('stage128'):
                    sc = 'stage128_image'
                    net = slim.conv2d(input_tensor, 16, 3)
                    net = incept_resnet.conv_maxpool(net, scope=sc)
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                    sc = 'stage64_image'
                    net = incept_resnet.resnet_k(
                        net, scope='stage64_residual')
                    net = incept_resnet.reduce_net(
                        net, scope='stage64_reduce')
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                    sc = 'stage32_image'
                    net = incept_resnet.resnet_k(
                        net, scope='stage32_residual')
                    net = slim.conv2d(
                        net, num_feature, 1, scope='stage32_out')
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                for hg in range(hg_repeat):
                    sc = 'hourglass_{}'.format(hg)
                    with tf.variable_scope(sc):
                        branch0 = hourglass.hg_net(
                            net, 2, scope=sc + '_hg')
                        branch0 = incept_resnet.resnet_k(
                            branch0, scope='_res')
                        branch_hm2 = slim.conv2d(
                            branch0, num_joint, 1,
                            # normalizer_fn=None, activation_fn=tf.nn.softmax)
                            normalizer_fn=None, activation_fn=None)
                        branch_olm = slim.conv2d(
                            branch0, num_joint, 1,
                            # normalizer_fn=None, activation_fn=tf.nn.relu)
                            normalizer_fn=None, activation_fn=None)
                        branch_uom = slim.conv2d(
                            branch0, num_joint * 3, 1,
                            # normalizer_fn=None, activation_fn=tf.nn.sigmoid)
                            normalizer_fn=None, activation_fn=None)
                        net_maps = tf.concat(
                            [branch_hm2, branch_olm, branch_uom],
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
                        branch1 = slim.conv2d(
                            net_maps, num_feature, 1)
                        net = net + branch0 + branch1
        raise ValueError('final_endpoint (%s) not recognized', final_endpoint)

    def placeholder_inputs(self, batch_size=None):
        frames_tf = tf.placeholder(
            tf.float32, shape=(
                batch_size,
                self.crop_size, self.crop_size,
                1))
        # hmap2_tf = tf.placeholder(
        #     tf.float32, shape=(
        #         batch_size,
        #         self.hmap_size, self.hmap_size,
        #         self.join_num))
        # olmap_tf = tf.placeholder(
        #     tf.float32, shape=(
        #         batch_size,
        #         self.hmap_size, self.hmap_size,
        #         self.join_num))
        # uomap_tf = tf.placeholder(
        #     tf.float32, shape=(
        #         batch_size,
        #         self.hmap_size, self.hmap_size,
        #         self.join_num * 3))
        poses_tf = tf.placeholder(
            tf.float32, shape=(
                batch_size,
                self.hmap_size, self.hmap_size,
                self.join_num * 5))
        return frames_tf, poses_tf

    def get_loss(self, pred, echt, end_points):
        """ simple sum-of-squares loss
            pred: BxHxWx(J*5)
            echt: BxHxWx(J*5)
        """
        # num_joint = self.join_num
        loss = 0
        for name, net in end_points.items():
            if not name.startswith('hourglass_'):
                continue
            # loss_hm2 = tf.reduce_sum(
            #     tf.nn.cross_entropy_with_logits(
            #         targets=echt[:, :num_joint],
            #         logits=pred[:, :num_joint])
            # )
            loss += tf.nn.l2_loss(net - echt)  # already divided by 2
        loss_reg = tf.add_n(tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES))
        return loss + loss_reg
