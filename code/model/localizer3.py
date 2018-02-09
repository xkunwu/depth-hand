import os
import sys
from importlib import import_module
# from psutil import virtual_memory
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
# import progressbar
import h5py
from model.base_conv3 import base_conv3
import matplotlib.pyplot as mpplot
from colour import Color

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


class localizer3(base_conv3):
    def __init__(self, args):
        super(localizer3, self).__init__(args)
        self.net_type = 'locor'
        self.crop_size = 32
        self.anchor_num = 8
        self.num_appen = 4
        self.predict_file = os.path.join(
            self.predict_dir, 'detection_{}'.format(
                self.name_desc))
        self.loss_lambda = 1.

    def receive_data(self, thedata, args):
        """ Receive parameters specific to the data """
        super(localizer3, self).receive_data(thedata, args)
        self.out_dim = self.anchor_num ** 3 * 5
        self.provider_worker = self.provider.prow_localizer3
        self.yanker = self.provider.yank_localizer3

    def evaluate_batch(self, writer, batch_data, pred_val):
        self.provider.write_region(
            writer, self.yanker, self.caminfo,
            batch_data['batch_index'], batch_data['batch_resce'],
            pred_val
        )

    def end_evaluate(self, thedata, args):
        self.batchallot = None
        fig = mpplot.figure(figsize=(2 * 5, 1 * 5))
        self.draw_prediction(thedata, args)
        mpplot.tight_layout()
        fname = 'detection_{}.png'.format(self.name_desc)
        mpplot.savefig(os.path.join(self.predict_dir, fname))
        mpplot.close(fig)
        print('figures saved: {}'.format(fname))

    def convert_input(self, img, args, caminfo):
        pcnt = args.data_ops.voxelize_depth(img, self.crop_size, caminfo)
        return np.expand_dims(np.expand_dims(pcnt, axis=0), axis=-1)

    def convert_output(self, pred_val, args, caminfo):
        pred_val = pred_val.flatten()
        halflen = self.crop_range
        centre = np.append(
            pred_val[:2] * halflen,
            pred_val[2] * halflen + halflen,
        )
        cube = iso_cube(centre, self.region_size)
        return cube

    def debug_compare(self, batch_pred, logger):
        batch_echt = self.batch_data['batch_poses']
        np.set_printoptions(
            threshold=np.nan,
            formatter={'float_kind': lambda x: "%.2f" % x})
        anchor_num_sub = self.anchor_num
        anchor_num = anchor_num_sub ** 3
        pcnt_echt = batch_echt[0, :anchor_num].reshape(
            anchor_num_sub, anchor_num_sub, anchor_num_sub)
        index_echt = np.array(np.unravel_index(
            np.argmax(pcnt_echt), pcnt_echt.shape))
        pcnt_pred = batch_pred[0, :anchor_num].reshape(
            anchor_num_sub, anchor_num_sub, anchor_num_sub)
        index_pred = np.array(np.unravel_index(
            np.argmax(pcnt_pred), pcnt_pred.shape))
        logger.info(
            [index_echt, np.max(pcnt_echt), np.sum(pcnt_echt)])
        logger.info(
            [index_pred, np.max(pcnt_pred), np.sum(pcnt_pred)])
        anchors_echt = batch_echt[0, anchor_num:].reshape(
            anchor_num_sub, anchor_num_sub, anchor_num_sub, 4)
        anchors_pred = batch_pred[0, anchor_num:].reshape(
            anchor_num_sub, anchor_num_sub, anchor_num_sub, 4)
        logger.info([
            anchors_echt[index_echt[0], index_echt[1], index_echt[2], :],
        ])
        logger.info([
            anchors_pred[index_pred[0], index_pred[1], index_echt[2], :],
        ])
        z_echt = index_echt[2]
        logger.info('\n{}'.format(pcnt_pred[..., z_echt]))
        # anchors_diff = np.fabs(
        #     anchors_echt[..., z_echt, 0:3] -
        #     anchors_pred[..., z_echt, 0:3])
        # logger.info('\n{}'.format(
        #     np.sum(anchors_diff, axis=-1)))
        logger.info('\n{}'.format(
            np.fabs(anchors_echt[..., z_echt, 3] - anchors_pred[..., z_echt, 3])))

    def _debug_draw_prediction(self, did, pred_val):
        pass

    def draw_prediction(self, thedata, args):
        import linecache
        import re
        frame_id = np.random.randint(
            1, high=sum(1 for _ in open(self.predict_file, 'r')))
        with h5py.File(self.appen_test, 'r') as h5file:
            img_id = h5file['index'][frame_id, 0]
            # frame_h5 = np.squeeze(h5file['frame'][frame_id, ...], -1)
            # poses_h5 = h5file['poses'][frame_id, ...].reshape(-1, 3)
            resce_h5 = h5file['resce'][frame_id, ...]

        print('[{}] drawing image #{:d} ...'.format(self.name_desc, img_id))
        resce3 = resce_h5[0:4]
        cube = iso_cube()
        cube.load(resce3)
        ax = mpplot.subplot(1, 2, 1, projection='3d')
        annot_line = args.data_io.get_line(
            thedata.training_annot_cleaned, img_id)
        img_name, pose_raw = args.data_io.parse_line_annot(annot_line)
        img = args.data_io.read_image(os.path.join(self.image_dir, img_name))
        points3 = args.data_ops.img_to_raw(img, self.caminfo)
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
        mpplot.gca().set_title('Ground truth')

        ax = mpplot.subplot(1, 2, 2, projection='3d')
        ax.scatter(
            points3_sam[:, 0], points3_sam[:, 1], points3_sam[:, 2],
            color=Color('lightsteelblue').rgb)
        ax.view_init(azim=-90, elev=-60)
        ax.set_zlabel('depth (mm)', labelpad=15)
        args.data_draw.draw_raw3d_pose(thedata, pose_raw)

        line_pred = linecache.getline(self.predict_file, frame_id)
        pred_list = re.split(r'\s+', line_pred.strip())
        centre = np.array([float(i) for i in pred_list[1:4]])
        cube = iso_cube(centre, self.region_size)
        # cube.show_dims()
        corners = cube.get_corners()
        iso_cube.draw_cube_wire(corners)
        mpplot.gca().set_title('Prediction')

        # mpplot.show()
        print('[{}] drawing image #{:d} - done.'.format(
            self.name_desc, img_id))

    def draw_random(self, thedata, args):
        from mayavi import mlab
        with h5py.File(self.appen_train, 'r') as h5file:
            store_size = h5file['index'].shape[0]
            frame_id = np.random.choice(store_size)
            img_id = h5file['index'][frame_id, 0]
            frame_h5 = np.squeeze(h5file['frame'][frame_id, ...], axis=-1)
            poses_h5 = h5file['poses'][frame_id, ...]
            resce_h5 = h5file['resce'][frame_id, ...]

        print('[{}] drawing image #{:d} ...'.format(self.name_desc, img_id))
        # colors = [Color('orange').rgb, Color('red').rgb, Color('lime').rgb]
        mpplot.subplots(nrows=2, ncols=2, figsize=(2 * 5, 2 * 5))
        mpplot.subplot(2, 2, 1)
        mpplot.gca().set_title('test input')
        annot_line = args.data_io.get_line(
            thedata.training_annot_cleaned, img_id)
        img_name, pose_raw = args.data_io.parse_line_annot(annot_line)
        img = args.data_io.read_image(os.path.join(self.image_dir, img_name))
        mpplot.imshow(img, cmap='bone')
        args.data_draw.draw_pose2d(
            thedata,
            args.data_ops.raw_to_2d(pose_raw, self.caminfo))

        ax = mpplot.subplot(2, 2, 3, projection='3d')
        mpplot.gca().set_title('test storage read')
        resce3 = resce_h5[0:4]
        cube = iso_cube()
        cube.load(resce3)
        cube.show_dims()
        points3 = args.data_ops.img_to_raw(img, self.caminfo)
        numpts = points3.shape[0]
        if 1000 < numpts:
            points3_sam = points3[
                np.random.choice(numpts, 1000, replace=False), :]
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

        ax = mpplot.subplot(2, 2, 4)
        mpplot.gca().set_title('test output')
        img_name = args.data_io.index2imagename(img_id)
        img = args.data_io.read_image(os.path.join(self.image_dir, img_name))
        mpplot.imshow(img, cmap='bone')

        ax = mpplot.subplot(2, 2, 2, projection='3d')
        mpplot.gca().set_title('test storage write')
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

        mlab.figure(size=(800, 800))
        img_name, frame, poses, resce = self.provider_worker(
            annot_line, self.image_dir, self.caminfo)
        frame = np.squeeze(frame, axis=-1)
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
            args.predict_dir,
            'draw_{}.png'.format(self.name_desc)))
        mpplot.show()
        print('[{}] drawing image #{:d} - done.'.format(
            self.name_desc, img_id))

    def get_model(
            self, input_tensor, is_training, bn_decay,
            scope=None, final_endpoint='stage_out'):
        """ input_tensor: BxHxWxC
            out_dim: BxJ, where J is flattened 3D locations
        """
        # batch_size = frames_tf.get_shape()[0].value
        end_points = {}
        self.end_point_list = []

        def add_and_check_final(name, net):
            end_points[name] = net
            return name == final_endpoint

        with tf.variable_scope(scope, self.name_desc, [input_tensor]):
            with slim.arg_scope(
                    [slim.batch_norm, slim.dropout], is_training=is_training), \
                slim.arg_scope(
                    [slim.fully_connected],
                    activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm), \
                slim.arg_scope(
                    [slim.max_pool3d, slim.avg_pool3d],
                    stride=1, padding='SAME'), \
                slim.arg_scope(
                    [slim.conv3d],
                    stride=1, padding='SAME', activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm):
                with tf.variable_scope('stage0'):
                    sc = 'stage0'
                    net = slim.conv3d(input_tensor, 32, 3, scope='conv0_3x3_1')
                    net = slim.conv3d(net, 32, 3, stride=2, scope='conv0_3x3_2')
                    net = slim.max_pool3d(
                        net, 3, scope='maxpool0_3x3_1')
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                with tf.variable_scope('stage1'):
                    sc = 'stage1'
                    net = slim.conv3d(net, 64, 3, scope='conv1_3x3_1')
                    net = slim.max_pool3d(
                        net, 3, stride=2, scope='maxpool1_3x3_2')
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                # with tf.variable_scope('stage2'):
                #     sc = 'stage2'
                #     net = slim.conv3d(net, 64, 3, scope='conv2_3x3_1')
                #     net = slim.max_pool3d(
                #         net, 3, stride=2, scope='maxpool2_3x3_2')
                #     self.end_point_list.append(sc)
                #     if add_and_check_final(sc, net):
                #         return net, end_points
                with tf.variable_scope('stage8'):
                    sc = 'stage_out'
                    net = slim.max_pool3d(
                        net, 5, stride=3, padding='VALID',
                        scope='maxpool8_5x5_3')
                    net = slim.conv3d(net, 128, 1, scope='reduce8')
                    net = slim.conv3d(
                        net, 256, net.get_shape()[1:4], padding='VALID',
                        scope='fullconn8')
                    net = slim.flatten(net)
                    net = slim.dropout(
                        net, 0.5, is_training=is_training, scope='dropout8')
                    net = slim.fully_connected(
                        net, self.out_dim, activation_fn=None, scope='output8')
                    # self.end_point_list.append(sc)
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

    def get_loss(self, pred, echt, end_points):
        """ simple sum-of-squares loss
            pred: BxO
            echt: BxO
        """
        scale = self.crop_range / self.region_size
        loss = tf.nn.l2_loss((pred - echt) * scale)
        reg_losses = tf.add_n(tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES))
        return loss + reg_losses
