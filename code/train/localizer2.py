import os
import sys
from importlib import import_module
# from psutil import virtual_memory
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import progressbar
import h5py
from base_regre import base_regre
import matplotlib.pyplot as mpplot
from colour import Color
from batch_allot import batch_allot_loc2

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
iso_rect = getattr(
    import_module('utils.iso_boxes'),
    'iso_rect'
)
regu_grid = getattr(
    import_module('utils.regu_grid'),
    'regu_grid'
)


class localizer2(base_regre):
    def __init__(self, args):
        super(localizer2, self).__init__(args)
        self.batch_allot = batch_allot_loc2
        self.crop_size = 256
        self.num_appen = 9
        self.loss_lambda = 10.

    def start_train(self):
        self.batchallot = self.batch_allot(
            self.batch_size, self.caminfo.crop_size, self.out_dim,
            self.num_channel, self.num_appen)

    def start_evaluate(self, filepack):
        self.batchallot = self.batch_allot(
            self.batch_size, self.caminfo.crop_size, self.out_dim,
            self.num_channel, self.num_appen)
        return filepack.write_file(self.predict_file)

    def prepare_data(self, thedata, args, batchallot, file_annot, name_appen):
        num_line = int(sum(1 for line in file_annot))
        file_annot.seek(0)
        batchallot.allot(num_line)
        store_size = batchallot.store_size
        num_stores = int(np.ceil(float(num_line) / store_size))
        self.logger.debug(
            'preparing data [{}]: {:d} lines (producing {:.4f} GB for store size {:d}) ...'.format(
                self.__class__.__name__, num_line,
                float(batchallot.store_bytes) / (2 << 30), store_size))
        timerbar = progressbar.ProgressBar(
            maxval=num_stores,
            widgets=[
                progressbar.Percentage(),
                ' ', progressbar.Bar('=', '[', ']'),
                ' ', progressbar.ETA()]
        ).start()
        crop_size = self.caminfo.crop_size
        out_dim = batchallot.out_dim
        num_channel = batchallot.num_channel
        num_appen = batchallot.num_appen
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
                    crop_size, crop_size,
                    num_channel),
                chunks=(1,
                        crop_size, crop_size,
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

    def check_dir(self, thedata, args):
        first_run = False
        if (
                (not os.path.exists(self.appen_train)) or
                (not os.path.exists(self.appen_test))
        ):
            first_run = True
        if not first_run:
            return
        from timeit import default_timer as timer
        from datetime import timedelta
        time_s = timer()
        batchallot = self.batch_allot(
            self.batch_size, self.caminfo.crop_size, self.out_dim,
            self.num_channel, self.num_appen)
        with file_pack() as filepack:
            file_annot = filepack.push_file(thedata.training_annot_train)
            self.prepare_data(thedata, args, batchallot, file_annot, self.appen_train)
        with file_pack() as filepack:
            file_annot = filepack.push_file(thedata.training_annot_test)
            self.prepare_data(thedata, args, batchallot, file_annot, self.appen_test)
        time_e = str(timedelta(seconds=timer() - time_s))
        self.logger.info('data prepared [{}], time: {}'.format(
            self.__class__.__name__, time_e))

    def receive_data(self, thedata, args):
        """ Receive parameters specific to the data """
        super(localizer2, self).receive_data(thedata, args)
        self.out_dim = 3 + self.anchor_num ** 2
        self.predict_file = os.path.join(
            self.predict_dir, 'detection_{}'.format(self.__class__.__name__))
        self.provider_worker = self.provider.prow_localizer2
        self.yanker = self.provider.yank_localizer2

    def evaluate_batch(self, writer, batch_data, pred_val):
        self.provider.write_region2(
            writer, self.yanker, self.caminfo,
            batch_data['batch_index'], batch_data['batch_resce'],
            pred_val
        )

    def end_evaluate(self, thedata, args):
        self.batchallot = None
        mpplot.figure(figsize=(2 * 5, 1 * 5))
        self.draw_prediction(thedata, args)
        mpplot.tight_layout()
        fname = 'detection_{}.png'.format(self.__class__.__name__)
        mpplot.savefig(os.path.join(self.predict_dir, fname))
        print('figures saved')

    def convert_input(self, img, args, caminfo):
        return np.expand_dims(np.expand_dims(img, axis=0), axis=-1)

    def convert_output(self, pred_val):
        centre = self.yanker()
        pred_val = pred_val.flatten()
        halflen = self.crop_range
        centre = np.append(
            pred_val[:2] * halflen,
            pred_val[2] * halflen + halflen,
        )
        cube = iso_cube(centre, self.region_size)
        return cube

    def draw_prediction(self, thedata, args):
        import linecache
        import re
        frame_id = np.random.randint(
            1, high=sum(1 for _ in open(self.predict_file, 'r')))
        with h5py.File(self.appen_test, 'r') as h5file:
            img_id = h5file['index'][frame_id, 0]
            frame_h5 = np.squeeze(h5file['frame'][frame_id, ...], -1)
            # poses_h5 = h5file['poses'][frame_id, ...].reshape(-1, 3)
            resce_h5 = h5file['resce'][frame_id, ...]
            frame_h5 = args.data_ops.rescale_depth_inv(
                frame_h5, self.caminfo)

        print('[{}] drawing pose #{:d}'.format(self.__class__.__name__, img_id))
        colors = [Color('orange').rgb, Color('red').rgb, Color('green').rgb]
        mpplot.subplot(1, 2, 1)
        annot_line = args.data_io.get_line(
            thedata.training_annot_cleaned, img_id)
        img_name, _ = args.data_io.parse_line_annot(annot_line)
        img = args.data_io.read_image(os.path.join(self.image_dir, img_name))
        mpplot.imshow(img, cmap='bone')
        resce3 = resce_h5[3:7]
        cube = iso_cube()
        cube.load(resce3)
        cube.show_dims()
        rects = cube.proj_rects_3(
            args.data_ops.raw_to_2d, self.caminfo
        )
        for ii, rect in enumerate(rects):
            rect.draw(colors[ii])
        mpplot.gcf().gca().set_title('Ground truth')

        mpplot.subplot(1, 2, 2)
        img = frame_h5
        mpplot.imshow(img, cmap='bone')
        line_pred = linecache.getline(self.predict_file, frame_id)
        pred_list = re.split(r'\s+', line_pred.strip())
        centre = np.array([float(i) for i in pred_list[1:4]])
        cube = iso_cube(centre, self.region_size)
        cube.show_dims()
        rects = cube.proj_rects_3(
            args.data_ops.raw_to_2d, self.caminfo
        )
        for ii, rect in enumerate(rects):
            rect.draw(colors[ii])
        mpplot.gca().set_title('Prediction')

        # mpplot.show()

    def draw_random(self, thedata, args):
        with h5py.File(self.appen_train, 'r') as h5file:
            store_size = h5file['index'].shape[0]
            frame_id = np.random.choice(store_size)
            img_id = h5file['index'][frame_id, 0]
            frame_h5 = np.squeeze(h5file['frame'][frame_id, ...], -1)
            poses_h5 = h5file['poses'][frame_id, ...]
            resce_h5 = h5file['resce'][frame_id, ...]
            frame_h5 = args.data_ops.rescale_depth_inv(
                frame_h5, self.caminfo)

        print('[{}] drawing pose #{:d}'.format(self.__class__.__name__, img_id))
        mpplot.subplots(nrows=2, ncols=2, figsize=(2 * 5, 2 * 5))
        mpplot.subplot(2, 2, 1)
        annot_line = args.data_io.get_line(
            thedata.training_annot_cleaned, img_id)
        img_name, pose_raw = args.data_io.parse_line_annot(annot_line)
        img = args.data_io.read_image(os.path.join(self.image_dir, img_name))
        mpplot.imshow(img, cmap='bone')
        args.data_draw.draw_pose2d(
            thedata,
            args.data_ops.raw_to_2d(pose_raw, thedata))

        mpplot.subplot(2, 2, 2)
        img = frame_h5
        mpplot.imshow(img, cmap='bone')
        points2, wsizes = self.provider.yank_localizer2_rect(
            poses_h5, resce_h5)
        rect = iso_rect(points2 - wsizes, wsizes * 2)
        rect.draw()

        ax = mpplot.subplot(2, 2, 3, projection='3d')
        p2z = self.yanker(poses_h5, resce_h5)
        centre = args.data_ops.d2z_to_raw(p2z, self.caminfo).flatten()
        cube = iso_cube(centre, self.region_size)
        cube.show_dims()
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
        iso_cube.draw_cube_wire(corners)

        mpplot.subplot(2, 2, 4)
        img = frame_h5
        mpplot.imshow(img, cmap='bone')
        anchors, resce = args.data_ops.generate_anchors(
            img, pose_raw, self.caminfo.anchor_num, self.caminfo)
        resce3 = resce[3:7]
        cube = iso_cube()
        cube.load(resce3)
        cube.show_dims()
        rects = cube.proj_rects_3(
            args.data_ops.raw_to_2d, self.caminfo
        )
        colors = [Color('orange').rgb, Color('red').rgb, Color('green').rgb]
        for ii, rect in enumerate(rects):
            rect.draw(colors[ii])

        mpplot.savefig(os.path.join(
            args.predict_dir,
            'draw_{}.png'.format(self.__class__.__name__)))
        mpplot.show()

    def get_model(
            self, input_tensor, is_training,
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

        with tf.variable_scope(scope, self.__class__.__name__, [input_tensor]):
            with slim.arg_scope(
                    [slim.batch_norm, slim.dropout], is_training=is_training), \
                slim.arg_scope(
                    [slim.fully_connected],
                    activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm), \
                slim.arg_scope(
                    [slim.max_pool2d, slim.avg_pool3d],
                    stride=1, padding='SAME'), \
                slim.arg_scope(
                    [slim.conv2d],
                    stride=1, padding='SAME', activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm):
                with tf.variable_scope('stage0'):
                    sc = 'stage0'
                    net = slim.conv2d(input_tensor, 8, 3, scope='conv0a_3x3_1')
                    net = slim.conv2d(net, 8, 3, stride=2, scope='conv0a_3x3_2')
                    net = slim.max_pool2d(net, 3, scope='maxpool0a_3x3_1')
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                with tf.variable_scope('stage1'):
                    sc = 'stage1'
                    net = slim.conv2d(net, 16, 3, stride=2, scope='conv1a_3x3_2')
                    net = slim.max_pool2d(net, 3, scope='maxpool1a_3x3_1')
                    net = slim.conv2d(net, 32, 3, stride=2, scope='conv1b_3x3_2')
                    net = slim.max_pool2d(net, 3, scope='maxpool1b_3x3_1')
                    net = slim.conv2d(net, 64, 3, stride=2, scope='conv1c_3x3_2')
                    net = slim.max_pool2d(net, 3, scope='maxpool1c_3x3_1')
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                with tf.variable_scope('stage16'):
                    sc = 'stage16'
                    net = slim.conv2d(net, 128, 3, scope='conv16_3x3_1')
                    net = slim.max_pool2d(
                        net, 3, stride=2, scope='maxpool16_3x3_2')
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                with tf.variable_scope('stage8'):
                    sc = 'stage_out'
                    with tf.variable_scope('branch_cls'):
                        out_cls = slim.max_pool2d(
                            net, 5, stride=3, padding='VALID',
                            scope='maxpool8a_5x5_3')
                        out_cls = slim.flatten(out_cls)
                        out_cls = slim.dropout(
                            out_cls, 0.5,
                            is_training=is_training, scope='dropout8a')
                        out_cls = slim.fully_connected(
                            out_cls, self.anchor_num ** 2,
                            activation_fn=None, scope='Logits')
                        out_cls = tf.nn.softmax(out_cls, name='Predictions')
                    with tf.variable_scope('branch_reg'):
                        out_reg = slim.max_pool2d(
                            net, 5, stride=3, padding='VALID',
                            scope='maxpool8b_5x5_3')
                        out_reg = slim.conv2d(out_reg, 128, 1, scope='reduce8')
                        out_reg = slim.conv2d(
                            out_reg, 256, out_reg.get_shape()[1:3],
                            padding='VALID', scope='fullconn8')
                        out_reg = slim.flatten(out_reg)
                        out_reg = slim.dropout(
                            out_reg, 0.5,
                            is_training=is_training, scope='dropout8b')
                        out_reg = slim.fully_connected(
                            out_reg, 3,
                            activation_fn=None, scope='output8')
                    net = tf.concat(axis=1, values=[out_cls, out_reg])
                    # self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points

        raise ValueError('final_endpoint (%s) not recognized', final_endpoint)

    def placeholder_inputs(self, n_frame=None):
        if n_frame is None:
            n_frame = self.batch_size
        frames_tf = tf.placeholder(
            tf.float32, shape=(
                n_frame,
                self.crop_size, self.crop_size,
                1))
        poses_tf = tf.placeholder(
            tf.float32, shape=(n_frame, self.out_dim))
        return frames_tf, poses_tf

    @staticmethod
    def smooth_l1(xa):
        x1 = xa - 0.5
        x2 = 0.5 * (xa ** 2)
        return tf.minimum(x1, x2)

    def get_loss(self, pred, echt, end_points):
        """ simple sum-of-squares loss
            pred: BxO
            echt: BxO
        """
        loss_cls = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.argmax(echt[:, :256], axis=1),
            logits=pred[:, :256]
        ) / 256
        loss_reg = tf.reduce_sum(
            self.smooth_l1(tf.abs(pred[:, 256:] - echt[:, 256:])),
            axis=1)
        loss = tf.reduce_sum(
            loss_cls + self.loss_lambda * loss_reg)
        return loss
