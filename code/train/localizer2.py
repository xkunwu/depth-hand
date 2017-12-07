import os
import sys
from importlib import import_module
# from psutil import virtual_memory
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import progressbar
import h5py
from train.base_regre import base_regre
import matplotlib.pyplot as mpplot
from colour import Color
from train.batch_allot import batch_allot_loc2

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
        self.net_type = 'locor'
        self.batch_allot = batch_allot_loc2
        self.crop_size = 256
        self.anchor_num = 8
        self.num_appen = 4
        self.loss_lambda = 1.

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
        self.out_dim = self.anchor_num ** 2 * 4
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
        print('figures saved: {}'.format(fname))

        from sklearn.metrics import precision_recall_curve
        from sklearn.metrics import average_precision_score
        import re
        mpplot.gcf().clear()
        anchor_num = self.anchor_num ** 2
        with h5py.File(self.appen_test, 'r') as h5file:
            label = h5file['poses'][:, :anchor_num]
        num_p = label.shape[0]
        pred_conf = np.empty((num_p, anchor_num))
        with open(self.predict_file, 'r') as pred_f:
            for ii, line_pred in enumerate(pred_f):
                pred_list = re.split(r'\s+', line_pred.strip())
                pred_val = [float(i) for i in pred_list[4:]]
                pred_conf[ii, :] = np.array(pred_val)
        lid = np.argmax(label, axis=1)
        pred_conf_pos = pred_conf[np.arange(num_p), lid]
        percent = np.arange(num_p + 1) * 100 / num_p
        pred_conf_pos_sort = np.sort(
            np.append(pred_conf_pos, np.max(pred_conf_pos)))
        mpplot.plot(
            pred_conf_pos_sort, percent,
            '-',
            linewidth=2.0
        )
        mpplot.ylabel('Percentage (%)')
        mpplot.ylim([0, 100])
        mpplot.xlabel('Dectection confidence')
        mpplot.xlim([0, 1.])
        # mpplot.xlim(left=0)
        # mpplot.xlim(right=100)
        mpplot.tight_layout()
        fname = 'evaluate_confidence_{}.png'.format(
            self.__class__.__name__)
        mpplot.savefig(os.path.join(self.predict_dir, fname))
        print('figures saved: {}'.format(fname))

        mpplot.gcf().clear()
        label = label.flatten()
        pred_conf = pred_conf.flatten()
        pr, rc, _ = precision_recall_curve(
            label, pred_conf)
        avg_ps = average_precision_score(label, pred_conf)
        # prInv = np.fliplr([pr])[0]
        # rcInv = np.fliplr([rc])[0]
        # j = rc.shape[0] - 2
        # while 0 <= j:
        #     if prInv[j + 1] > prInv[j]:
        #         prInv[j] = prInv[j + 1]
        #     j -= 1
        # decreasing_max_pr = np.maximum.accumulate(
        #     prInv[::-1][::-1]
        # )
        # mpplot.plot(
        #     rcInv, decreasing_max_pr)
        # mpplot.fill_between(
        #     rcInv, decreasing_max_pr, step='post', alpha=0.2,
        #     color='b')
        mpplot.step(rc, pr, color='b', alpha=0.2, where='post')
        mpplot.fill_between(rc, pr, step='post', color='b', alpha=0.2)
        mpplot.ylim([0., 1.01])
        mpplot.xlim([0., 1.])
        mpplot.ylabel('Precision')
        mpplot.xlabel('Recall')
        mpplot.title('Hand detection: AP={0:0.2f}'.format(
            avg_ps))
        mpplot.tight_layout()
        # mpplot.show()
        fname = 'evaluate_pr_{}.png'.format(self.__class__.__name__)
        mpplot.savefig(os.path.join(self.predict_dir, fname))
        print('figures saved: {}'.format(fname))

    def convert_input(self, img, args, caminfo):
        return np.expand_dims(np.expand_dims(img, axis=0), axis=-1)

    def convert_output(self, pred_val, args, caminfo):
        centre, index, confidence = self.yanker(
            pred_val.flatten(), np.zeros(3), caminfo)
        cube = iso_cube(centre.flatten(), self.region_size)
        return cube, index, confidence

    def _draw_prediction(self, frame_h5, resce_h5, pred_val):
        frame_h5 = self.args.data_ops.rescale_depth_inv(
            frame_h5, self.caminfo)
        colors = [Color('orange').rgb, Color('red').rgb, Color('lime').rgb]
        mpplot.subplot(1, 2, 1)
        mpplot.imshow(frame_h5, cmap='bone')
        resce3 = resce_h5[0:4]
        cube = iso_cube()
        cube.load(resce3)
        cube.show_dims()
        rects = cube.proj_rects_3(
            self.args.data_ops.raw_to_2d, self.caminfo
        )
        for ii, rect in enumerate(rects):
            rect.draw(colors[ii])
        mpplot.gcf().gca().set_title('Ground truth')

        mpplot.subplot(1, 2, 2)
        mpplot.imshow(frame_h5, cmap='bone')
        cube, index, confidence = self.convert_output(
            pred_val, self.args, self.caminfo)
        cube.show_dims()
        rects = cube.proj_rects_3(
            self.args.data_ops.raw_to_2d, self.caminfo
        )
        for ii, rect in enumerate(rects):
            rect.draw(colors[ii])
        mpplot.gca().set_title('Prediction')

    def draw_prediction(self, thedata, args):
        import linecache
        import re
        frame_id = np.random.randint(
            1, high=sum(1 for _ in open(self.predict_file, 'r')))
        with h5py.File(self.appen_test, 'r') as h5file:
            img_id = h5file['index'][frame_id, 0]
            frame_h5 = np.squeeze(h5file['frame'][frame_id, ...], -1)
            # poses_h5 = h5file['poses'][frame_id, ...]
            resce_h5 = h5file['resce'][frame_id, ...]
            frame_h5 = args.data_ops.rescale_depth_inv(
                frame_h5, self.caminfo)

        print('[{}] drawing image #{:d}'.format(self.__class__.__name__, img_id))
        colors = [Color('orange').rgb, Color('red').rgb, Color('lime').rgb]
        mpplot.subplot(1, 2, 1)
        annot_line = args.data_io.get_line(
            thedata.training_annot_cleaned, img_id)
        img_name, _ = args.data_io.parse_line_annot(annot_line)
        img = args.data_io.read_image(os.path.join(self.image_dir, img_name))
        mpplot.imshow(img, cmap='bone')
        resce3 = resce_h5[0:4]
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

        print('[{}] drawing image #{:d}'.format(self.__class__.__name__, img_id))
        colors = [Color('orange').rgb, Color('red').rgb, Color('lime').rgb]
        mpplot.subplots(nrows=2, ncols=2, figsize=(2 * 5, 2 * 5))
        mpplot.subplot(2, 2, 1)
        mpplot.gcf().gca().set_title('test input')
        annot_line = args.data_io.get_line(
            thedata.training_annot_cleaned, img_id)
        img_name, pose_raw = args.data_io.parse_line_annot(annot_line)
        img = args.data_io.read_image(os.path.join(self.image_dir, img_name))
        mpplot.imshow(img, cmap='bone')
        args.data_draw.draw_pose2d(
            thedata,
            args.data_ops.raw_to_2d(pose_raw, thedata))

        ax = mpplot.subplot(2, 2, 3, projection='3d')
        mpplot.gcf().gca().set_title('test storage read')
        resce3 = resce_h5[0:4]
        cube = iso_cube()
        cube.load(resce3)
        cube.show_dims()
        points3 = args.data_ops.img_to_raw(frame_h5, thedata)
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
        mpplot.gcf().gca().set_title('test output')
        img_name = args.data_io.index2imagename(img_id)
        img = args.data_io.read_image(os.path.join(self.image_dir, img_name))
        mpplot.imshow(img, cmap='bone')
        anchor_num = self.anchor_num ** 2
        pcnt = poses_h5[:anchor_num].reshape(
            (self.anchor_num, self.anchor_num))
        print(pcnt)
        index = np.array(np.unravel_index(
            np.argmax(pcnt), pcnt.shape))
        anchors = poses_h5[anchor_num:]
        print(index)
        points2, wsizes, centre = self.provider.yank_localizer2_rect(
            index, anchors, self.caminfo)
        print(np.append(points2, wsizes).reshape(1, -1))
        rect = iso_rect(points2 - wsizes, wsizes * 2)
        rect.draw()
        cube = iso_cube(centre.flatten(), self.region_size)
        cube.show_dims()
        rects = cube.proj_rects_3(
            args.data_ops.raw_to_2d, self.caminfo
        )
        for ii, rect in enumerate(rects):
            rect.draw(colors[ii])

        mpplot.subplot(2, 2, 2)
        mpplot.gcf().gca().set_title('test storage write')
        img_name, frame, poses, resce = self.provider_worker(
            annot_line, self.image_dir, thedata)
        frame = args.data_ops.rescale_depth_inv(
            frame, self.caminfo)
        if (
                (1e-4 < np.linalg.norm(frame_h5 - frame)) or
                (1e-4 < np.linalg.norm(poses_h5 - poses))
        ):
            print(np.linalg.norm(frame_h5 - frame))
            print(np.linalg.norm(poses_h5 - poses))
            print('ERROR - h5 storage corrupted!')
        mpplot.imshow(frame, cmap='bone')
        resce3 = resce[0:4]
        cube = iso_cube()
        cube.load(resce3)
        cube.show_dims()
        rects = cube.proj_rects_3(
            args.data_ops.raw_to_2d, self.caminfo
        )
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

        with tf.variable_scope(
                scope, self.__class__.__name__, [input_tensor]):
            with slim.arg_scope(
                    [slim.batch_norm, slim.dropout],
                    is_training=is_training), \
                slim.arg_scope(
                    [slim.fully_connected],
                    weights_regularizer=slim.l2_regularizer(0.00004),
                    biases_regularizer=slim.l2_regularizer(0.00004),
                    activation_fn=None, normalizer_fn=None), \
                slim.arg_scope(
                    [slim.max_pool2d, slim.avg_pool2d],
                    stride=1, padding='SAME'), \
                slim.arg_scope(
                    [slim.conv2d],
                    stride=1, padding='SAME',
                    activation_fn=tf.nn.relu,
                    weights_regularizer=slim.l2_regularizer(0.00004),
                    biases_regularizer=slim.l2_regularizer(0.00004),
                    normalizer_fn=slim.batch_norm):
                with tf.variable_scope('stage0'):
                    sc = 'stage0_image'
                    net = slim.conv2d(
                        input_tensor, 8, 3, scope='conv0a_3x3_1')
                    net = slim.conv2d(
                        net, 8, 3, stride=2, scope='conv0a_3x3_2')
                    net = slim.max_pool2d(
                        net, 3, scope='maxpool0a_3x3_1')
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                with tf.variable_scope('stage1'):
                    sc = 'stage1_image'
                    net = slim.conv2d(
                        net, 16, 3, stride=2, scope='conv1a_3x3_2')
                    net = slim.max_pool2d(
                        net, 3, scope='maxpool1a_3x3_1')
                    net = slim.conv2d(
                        net, 32, 3, stride=2, scope='conv1b_3x3_2')
                    net = slim.max_pool2d(
                        net, 3, scope='maxpool1b_3x3_1')
                    net = slim.conv2d(
                        net, 64, 3, stride=2, scope='conv1c_3x3_2')
                    net = slim.max_pool2d(
                        net, 3, scope='maxpool1c_3x3_1')
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                with tf.variable_scope('stage16'):
                    sc = 'stage16_image'
                    net = slim.conv2d(
                        net, 128, 3, stride=2, scope='conv16_3x3_1')
                    net = slim.max_pool2d(
                        net, 3, scope='maxpool16_3x3_2')
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                with tf.variable_scope('stage8'):
                    sc = 'stage_out'
                    fshape = net.get_shape()[1:3]
                    anchor_num = self.anchor_num ** 2
                    with tf.variable_scope('branch_cls'):
                        out_cls = slim.conv2d(
                            net, 16, 1, scope='reduce_cls_a')
                        out_cls = slim.conv2d(
                            out_cls, 16, fshape,
                            scope='fullconn_cls_a')
                        out_cls = slim.dropout(
                            out_cls, 0.5, scope='dropout_cls')
                        # out_cls = slim.conv2d(out_cls, 1, 1, scope='reduce_cls_b')
                        # out_cls = slim.conv2d(
                        #     out_cls, 1, fshape, activation_fn=None,
                        #     scope='fullconn_cls_b')
                        out_cls = slim.flatten(out_cls)
                        out_cls = slim.fully_connected(
                            out_cls, anchor_num,
                            scope='fullconn_cls_b')
                        self.end_point_list.append('branch_cls')
                        if add_and_check_final('branch_cls', out_cls):
                            return net, end_points
                    with tf.variable_scope('branch_reg'):
                        out_reg = slim.conv2d(
                            net, 16, 1, scope='reduce_reg_a')
                        out_reg = slim.conv2d(
                            out_reg, 16, fshape,
                            scope='fullconn_reg_a')
                        # self.end_point_list.append('fullconn_reg_a')
                        # if add_and_check_final('fullconn_reg_a', out_reg):
                        #     return net, end_points
                        out_reg = slim.dropout(
                            out_reg, 0.5, scope='dropout_reg')
                        # self.end_point_list.append('dropout_reg')
                        # if add_and_check_final('dropout_reg', out_reg):
                        #     return net, end_points
                        # out_reg = slim.conv2d(out_reg, 3, 1, scope='reduce_reg_b')
                        # out_reg = slim.conv2d(
                        #     out_reg, 3, fshape, activation_fn=None,
                        #     scope='fullconn_reg_b')
                        out_reg = slim.flatten(out_reg)
                        out_reg = slim.fully_connected(
                            out_reg, anchor_num * 3,
                            scope='fullconn_reg_b')
                        self.end_point_list.append('branch_reg')
                        if add_and_check_final('branch_reg', out_reg):
                            return net, end_points
                    net = tf.concat(axis=1, values=[out_cls, out_reg])
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
                self.crop_size, self.crop_size,
                1))
        poses_tf = tf.placeholder(
            tf.float32, shape=(batch_size, self.out_dim))
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
            pred: BxO
            echt: BxO
        """
        anchor_num = self.anchor_num ** 2
        loss_cls = tf.reduce_sum(
            tf.nn.weighted_cross_entropy_with_logits(
                targets=echt[:, :anchor_num],
                logits=pred[:, :anchor_num],
                pos_weight=10))
        loss_reg = tf.reduce_sum(
            self.smooth_l1(tf.abs(
                pred[:, anchor_num:] - echt[:, anchor_num:]))
        )
        # loss_reg = tf.reduce_sum(
        #     (pred[:, anchor_num:] - echt[:, anchor_num:]) ** 2)
        loss = loss_cls + self.loss_lambda * loss_reg
        # loss = loss_cls
        # loss = loss_reg
        return loss
