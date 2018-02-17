import os
from importlib import import_module
import numpy as np
import tensorflow as tf
import progressbar
import matplotlib.pyplot as mpplot
from cv2 import resize as cv2resize
from utils.coder import file_pack
from utils.iso_boxes import iso_cube


class base_regre(object):
    """ This class holds baseline training approach using plain regression.
    """

    @staticmethod
    def get_trainer(args, new_log):
        from train.train_abc import train_abc
        return train_abc(args, new_log)

    def __init__(self, args):
        self.net_rank = 2
        self.net_type = 'poser'
        self.name_desc = self.__class__.__name__ + args.model_desc
        self.crop_size = 128
        # self.anchor_num = 16
        self.crop_range = 480.
        self.num_channel = 1
        # self.num_appen = 7
        self.batch_allot = getattr(
            import_module('model.batch_allot'),
            'batch_crop2'
        )
        self.batch_data = {}
        # receive arguments
        self.store_name = {}
        self.store_handle = {}
        self.args = args
        self.prepare_dir = args.prepare_dir
        self.predict_dir = args.predict_dir
        self.predict_file = os.path.join(
            self.predict_dir, 'predict_{}'.format(
                self.name_desc))
        self.batch_size = args.batch_size
        self.ckpt_path = os.path.join(
            args.out_dir, 'log', 'blinks',
            self.name_desc, 'model.ckpt')

    def write_args(self, writer):
        writer.write('--{}={}\n'.format('crop_size', self.crop_size))
        writer.write('--{}={}\n'.format('crop_range', self.crop_range))
        writer.write('--{}={}\n'.format('num_channel', self.num_channel))

    def tweak_arguments(self, args):
        args.crop_size = self.crop_size
        # args.anchor_num = self.anchor_num
        args.crop_range = self.crop_range

    def start_train(self):
        self.store_size = self.args.data_inst.train_test_split

    def start_epoch_train(self):
        # # new round starting from next portion of data
        # self.batch_beg = split_end
        # self.split_end = split_beg  # + self.store_size
        self.batch_beg = 0
        self.split_end = self.args.data_inst.train_valid_split

    def start_epoch_valid(self):
        self.batch_beg = self.args.data_inst.train_valid_split
        self.split_end = self.args.data_inst.train_test_split

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
            self.store_handle['crop2'][self.batch_beg:batch_end, ...],
            axis=-1)
        self.batch_data['batch_poses'] = \
            self.store_handle['pose_c'][self.batch_beg:batch_end, ...]
        self.batch_data['batch_index'] = \
            self.store_handle['index'][self.batch_beg:batch_end, ...]
        self.batch_data['batch_resce'] = \
            self.store_handle['resce'][self.batch_beg:batch_end, ...]
        self.batch_beg = batch_end
        return self.batch_data

    def end_train(self):
        pass

    def start_evaluate(self):
        self.batch_beg = self.args.data_inst.train_test_split
        self.split_end = self.args.data_inst.num_training
        self.store_size = self.split_end - self.batch_beg
        self.eval_pred = []

    def evaluate_batch(self, pred_val):
        batch_index = self.batch_data['batch_index']
        batch_resce = self.batch_data['batch_resce']
        batch_poses = pred_val
        num_elem = batch_index.shape[0]
        poses_out = np.empty(batch_poses.shape)
        for ei, resce, poses in zip(range(num_elem), batch_resce, batch_poses):
            pose_local = poses.reshape(-1, 3)
            pose_raw = self.yanker(pose_local, resce, self.caminfo)
            poses_out[ei] = pose_raw.reshape(1, -1)
        self.eval_pred.append(poses_out)

    def draw_image_pose(self, ax, line, image_dir, caminfo):
        img_name, pose = self.data_module.io.parse_line_annot(line)
        img_path = os.path.join(image_dir, img_name)
        img = self.data_module.io.read_image(img_path)
        cube = iso_cube(
            (np.max(pose, axis=0) + np.min(pose, axis=0)) / 2,
            caminfo.region_size
        )
        points3_pick = cube.pick(self.data_module.ops.img_to_raw(
            img, caminfo))
        coord, depth = cube.raw_to_unit(points3_pick, sort=False)
        image_size = np.floor(caminfo.region_size * 1.5).astype(int)
        img_crop_resize = cube.print_image(
            coord, depth, image_size)
        ax.imshow(img_crop_resize, cmap=mpplot.cm.bone_r)
        pose2d, _ = cube.raw_to_unit(pose, sort=False)
        pose2d *= image_size
        self.data_module.draw.draw_pose2d(
            ax, caminfo, pose2d)
        ax.axis('off')
        return img_name

    def draw_prediction_poses(self, image_dir, annot_echt, annot_pred, caminfo):
        import linecache
        img_id = 4
        line_echt = linecache.getline(annot_echt, img_id)
        line_pred = linecache.getline(annot_pred, img_id)
        ax = mpplot.subplot(2, 2, 1)
        img_name = self.draw_image_pose(ax, line_echt, image_dir, caminfo)
        ax = mpplot.subplot(2, 2, 2)
        img_name = self.draw_image_pose(ax, line_pred, image_dir, caminfo)
        print('draw predition #{:d}: {}'.format(img_id, img_name))
        img_id = np.random.randint(1, high=sum(1 for _ in open(annot_pred, 'r')))
        line_echt = linecache.getline(annot_echt, img_id)
        line_pred = linecache.getline(annot_pred, img_id)
        ax = mpplot.subplot(2, 2, 3)
        img_name = self.draw_image_pose(ax, line_echt, image_dir, caminfo)
        ax = mpplot.subplot(2, 2, 4)
        img_name = self.draw_image_pose(ax, line_pred, image_dir, caminfo)
        print('draw predition #{:d}: {}'.format(img_id, img_name))
        return img_id

    def end_evaluate(self, thedata, args):
        index = self.store_handle['index'][
            self.args.data_inst.train_test_split:, ...]
        poses = np.vstack(self.eval_pred)
        self.eval_pred = []
        # with h5py.File(self.predict_file + '.h5', 'w') as writer:
        #     self.data_module.io.write_h5(writer, index, poses)
        with open(self.predict_file, 'w') as writer:
            self.data_module.io.write_txt(writer, index, poses)

        fig = mpplot.figure(figsize=(2 * 5, 2 * 5))
        img_id = self.draw_prediction_poses(
            thedata.training_images,
            thedata.training_annot_test,
            self.predict_file,
            thedata
        )
        fname = 'detection_{}_{:d}.png'.format(self.name_desc, img_id)
        mpplot.savefig(os.path.join(self.predict_dir, fname))
        mpplot.close(fig)
        error_maxj = self.data_module.eval.evaluate_poses(
            self.caminfo, self.name_desc,
            self.predict_dir, self.predict_file)
        self.logger.info('maximal per-joint mean error: {}'.format(
            error_maxj
        ))

    def detect_write_images(self):
        outdir = os.path.join(self.args.log_dir_t, 'images')
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        with open(self.predict_file, 'r') as f:
            fig = mpplot.figure(figsize=(5, 5))
            ax = fig.add_axes([0, 0, 1, 1])
            num_line = int(sum(1 for line in f))
            f.seek(0)
            print('start writing {} images ...'.format(num_line))
            timerbar = progressbar.ProgressBar(
                maxval=num_line,
                widgets=[
                    progressbar.Percentage(),
                    ' ', progressbar.Bar('=', '[', ']'),
                    ' ', progressbar.ETA()]
            ).start()
            for li, line in enumerate(f):
                img_name = self.draw_image_pose(
                    ax, line,
                    self.args.data_inst.training_images,
                    self.caminfo)
                mpplot.savefig(os.path.join(outdir, img_name))
                ax.clear()
                timerbar.update(li)
            timerbar.finish()
            mpplot.close(fig)

    def yanker(self, pose_local, resce, caminfo):
        resce3 = resce[0:4]
        return self.data_module.ops.pca_to_raw(pose_local, resce3)

    def prepare_data(self, thedata, args,
                     filepack, prepare_h5file):
        file_annot = filepack.push_h5(thedata.training_annot_train)
        num_line = file_annot['index'].shape[0]
        batchallot = self.batch_allot(self, num_line)
        store_size = batchallot.store_size
        self.logger.info(
            '[{}] preparing data: {:d} lines (producing {:.4f} MB for store size {:d}) ...'.format(
                self.name_desc, num_line,
                float(batchallot.store_bytes) / (2 << 20),
                store_size))
        for key in prepare_h5file:
            prepare_h5file[key] = batchallot.create_fn[key](
                filepack, self.store_name[key], num_line)
        timerbar = progressbar.ProgressBar(
            maxval=num_line,
            widgets=[
                progressbar.Percentage(),
                ' ', progressbar.Bar('=', '[', ']'),
                ' ', progressbar.ETA()]
        ).start()
        store_beg = 0
        li = 0
        while True:
            store_end = min(store_beg + store_size, num_line)
            proc_size = store_end - store_beg
            if 0 >= proc_size:
                break
            index = file_annot['index'][store_beg:store_end, 0]
            poses = file_annot['poses'][store_beg:store_end, :]
            resce = file_annot['resce'][store_beg:store_end, :]
            args_zip = zip(range(proc_size), index, poses, resce)
            for key in prepare_h5file:
                resline = self.data_module.provider.puttensor_mt(
                    args_zip,
                    thedata.store_prow[key], thedata, batchallot
                )
                prepare_h5file[key][store_beg:store_end, ...] = \
                    batchallot.entry[key]
            store_beg = store_end
            li += resline
            timerbar.update(li)
        timerbar.finish()

    def prepare_data_recur(self, target, filepack, thedata):
        precon_list = self.store_precon[target]
        if not precon_list:
            return
        for precon in precon_list:
            self.prepare_data_recur(precon, filepack, thedata)
        if os.path.exists(self.store_name[target]):
            return
        precon_h5 = {}
        num_line = 0
        for precon in precon_list:
            precon_h5[precon] = filepack.push_h5(
                self.store_name[precon])
            num_line = precon_h5[precon][precon].shape[0]
        batchallot = self.batch_allot(self, num_line)
        store_size = batchallot.store_size
        self.logger.info(
            '[{}] preparing data ({}): {:d} lines with store size {:d} ...'.format(
                self.name_desc, target, num_line, store_size))
        target_h5, batch_data = batchallot.create_fn[target](
            filepack, self.store_name[target], num_line
        )
        timerbar = progressbar.ProgressBar(
            maxval=num_line,
            widgets=[
                progressbar.Percentage(),
                ' ', progressbar.Bar('=', '[', ']'),
                ' ', progressbar.ETA()]
        ).start()
        store_beg = 0
        li = 0
        while True:
            store_end = min(store_beg + store_size, num_line)
            proc_size = store_end - store_beg
            if 0 >= proc_size:
                break
            args = [range(proc_size)]
            for precon in precon_list:
                args.append(precon_h5[precon][precon][
                    store_beg:store_end, ...])
            # args_zip = zip(args)
            self.data_module.provider.puttensor_mt(
                args,
                thedata.store_prow[target], thedata, batch_data
            )
            target_h5[store_beg:store_end, ...] = batch_data
            store_beg = store_end
            li += proc_size
            timerbar.update(li)
        timerbar.finish()

    def check_dir(self, thedata, args):
        from timeit import default_timer as timer
        from datetime import timedelta
        time_s = timer()
        with file_pack() as filepack:
            for name in self.store_name:
                self.prepare_data_recur(
                    name, filepack, thedata)
        time_e = str(timedelta(seconds=timer() - time_s))
        self.logger.info('data prepared [{}], time: {}'.format(
            self.__class__.__name__, time_e))
        # prepare_h5file = {}  # missing data
        # for name, filename in self.store_name.items():
        #     if not os.path.exists(filename):
        #         prepare_h5file[name] = None
        # if prepare_h5file:
        #     from timeit import default_timer as timer
        #     from datetime import timedelta
        #     time_s = timer()
        #     with file_pack() as filepack:
        #         self.prepare_data(thedata, args, filepack, prepare_h5file)
        #     time_e = str(timedelta(seconds=timer() - time_s))
        #     self.logger.info('data prepared [{}], time: {}'.format(
        #         self.__class__.__name__, time_e))
        self.store_handle = {}  # pointing to h5 db
        for name, filename in self.store_name.items():
            h5file = args.filepack.push_h5(filename)
            self.store_handle[name] = h5file[name]
            # print(self.store_handle[name])

    def receive_data(self, thedata, args):
        """ Receive parameters specific to the data """
        self.logger = args.logger
        self.data_module = args.data_module
        self.out_dim = thedata.join_num * 3
        self.image_dir = thedata.training_images
        self.caminfo = thedata
        self.region_size = thedata.region_size
        self.train_file = args.data_inst.training_annot_train
        self.store_name = {
            'index': self.train_file,
            'poses': self.train_file,
            'resce': self.train_file,
            'pose_c': os.path.join(self.prepare_dir, 'pose_c'),
            'crop2': os.path.join(
                self.prepare_dir, 'crop2_{}'.format(self.crop_size)),
        }
        self.store_precon = {
            'index': [],
            'poses': [],
            'resce': [],
            'pose_c': ['poses', 'resce'],
            'crop2': ['index', 'resce'],
        }

    def debug_compare(self, batch_pred, logger):
        batch_echt = self.batch_data['batch_poses']
        np.set_printoptions(
            threshold=np.nan,
            formatter={'float_kind': lambda x: "%.2f" % x})
        pcnt_echt = batch_echt[0, :].reshape(21, 3)
        pcnt_pred = batch_pred[0, :].reshape(21, 3)
        logger.info(np.concatenate(
            (np.max(pcnt_echt, axis=0), np.min(pcnt_echt, axis=0))
        ))
        logger.info(np.concatenate(
            (np.max(pcnt_pred, axis=0), np.min(pcnt_pred, axis=0))
        ))
        logger.info('\n{}'.format(pcnt_echt))
        logger.info('\n{}'.format(pcnt_pred))
        logger.info('\n{}'.format(
            np.fabs(pcnt_echt - pcnt_pred)))

    def draw_random(self, thedata, args):
        index_h5 = self.store_handle['index']
        store_size = index_h5.shape[0]
        frame_id = np.random.choice(store_size)
        # frame_id = 0
        img_id = index_h5[frame_id, ...]
        frame_h5 = self.store_handle['crop2'][frame_id, ...]
        poses_h5 = self.store_handle['pose_c'][frame_id, ...].reshape(-1, 3)
        resce_h5 = self.store_handle['resce'][frame_id, ...]

        print('[{}] drawing image #{:d} ...'.format(self.name_desc, img_id))
        print(np.min(frame_h5), np.max(frame_h5))
        print(np.histogram(frame_h5, range=(1e-4, np.max(frame_h5))))
        print(np.min(poses_h5, axis=0), np.max(poses_h5, axis=0))
        print(resce_h5)
        resce3 = resce_h5[0:4]
        cube = iso_cube()
        cube.load(resce3)
        from colour import Color
        colors = [Color('orange').rgb, Color('red').rgb, Color('lime').rgb]
        mpplot.subplots(nrows=1, ncols=2, figsize=(2 * 5, 1 * 5))

        ax = mpplot.subplot(1, 2, 2)
        mpplot.gca().set_title('test storage read')
        # resize the cropped resion for eazier pose drawing in the commen frame
        sizel = np.floor(resce3[0]).astype(int)
        ax.imshow(
            cv2resize(frame_h5, (sizel, sizel)),
            cmap=mpplot.cm.bone_r)
        pose_raw = args.data_ops.local_to_raw(poses_h5, resce3)
        pose3d = cube.trans_scale_to(poses_h5)
        pose2d, _ = cube.project_ortho(pose3d, roll=0, sort=False)
        pose2d *= sizel
        args.data_draw.draw_pose2d(
            ax, thedata,
            pose2d,
        )

        ax = mpplot.subplot(1, 2, 1)
        mpplot.gca().set_title('test image - {:d}'.format(img_id))
        img_name = args.data_io.index2imagename(img_id)
        img = args.data_io.read_image(os.path.join(self.image_dir, img_name))
        ax.imshow(img, cmap=mpplot.cm.bone_r)
        pose_raw = self.yanker(poses_h5, resce_h5, thedata)
        args.data_draw.draw_pose2d(
            ax, thedata,
            args.data_ops.raw_to_2d(pose_raw, thedata)
        )
        rects = cube.proj_rects_3(
            args.data_ops.raw_to_2d, thedata
        )
        for ii, rect in enumerate(rects):
            rect.draw(ax, colors[ii])

        # img_name, frame, poses, resce = self.provider_worker(
        #     annot_line, self.image_dir, thedata)
        # frame = np.squeeze(frame, axis=-1)
        # poses = poses.reshape(-1, 3)
        # if (
        #         (1e-4 < np.linalg.norm(frame_h5 - frame)) or
        #         (1e-4 < np.linalg.norm(poses_h5 - poses))
        # ):
        #     print(np.linalg.norm(frame_h5 - frame))
        #     print(np.linalg.norm(poses_h5 - poses))
        #     print('ERROR - h5 storage corrupted!')

        mpplot.tight_layout()
        mpplot.savefig(os.path.join(
            args.predict_dir,
            'draw_{}_{}.png'.format(self.name_desc, img_id)))
        if self.args.show_draw:
            mpplot.show()
        print('[{}] drawing image #{:d} - done.'.format(
            self.name_desc, img_id))

    def get_model(
            self, input_tensor, is_training, bn_decay,
            scope=None, final_endpoint='stage_out'):
        """ input_tensor: BxHxWxC
            out_dim: BxJ, where J is flattened 3D locations
        """
        end_points = {}
        self.end_point_list = []

        # from importlib import import_module
        # tf_util = import_module('utils.tf_util')
        # net = tf_util.conv2d(
        #     input_tensor, 16, [5, 5], stride=[1, 1], scope='conv1',
        #     padding='VALID', is_training=is_training, bn=True, bn_decay=bn_decay)
        # self.end_point_list.append('conv1')
        # end_points['conv1'] = net
        # net = tf_util.max_pool2d(
        #     net, [4, 4], scope='maxpool1', padding='VALID')
        # self.end_point_list.append('maxpool1')
        # end_points['maxpool1'] = net
        # net = tf_util.conv2d(
        #     net, 32, [3, 3], stride=[1, 1], scope='conv2',
        #     padding='VALID', is_training=is_training, bn=True, bn_decay=bn_decay)
        # self.end_point_list.append('conv2')
        # end_points['conv2'] = net
        # net = tf_util.max_pool2d(
        #     net, [2, 2], scope='maxpool2', padding='VALID')
        # self.end_point_list.append('maxpool2')
        # end_points['maxpool2'] = net
        # net = tf_util.conv2d(
        #     net, 64, [3, 3], stride=[1, 1], scope='conv3',
        #     padding='VALID', is_training=is_training, bn=True, bn_decay=bn_decay)
        # self.end_point_list.append('conv3')
        # end_points['conv3'] = net
        # net = tf_util.max_pool2d(
        #     net, [2, 2], scope='maxpool3', padding='VALID')
        # self.end_point_list.append('maxpool3')
        # end_points['maxpool3'] = net
        #
        # # net = tf.reshape(net, [self.batch_size, -1])
        # net = tf.contrib.layers.flatten(net)
        # net = tf_util.fully_connected(
        #     net, 1024, scope='fullconn1',
        #     is_training=is_training, bn=True, bn_decay=bn_decay)
        # self.end_point_list.append('fullconn1')
        # end_points['fullconn1'] = net
        # net = tf_util.dropout(
        #     net, keep_prob=0.5, scope='dropout1', is_training=is_training)
        # self.end_point_list.append('dropout1')
        # end_points['dropout1'] = net
        # net = tf_util.fully_connected(
        #     net, self.out_dim, scope='fullconn3', activation_fn=None)
        # self.end_point_list.append('fullconn3')
        # end_points['fullconn3'] = net

        def add_and_check_final(name, net):
            end_points[name] = net
            return name == final_endpoint
        from tensorflow.contrib import slim
        from incept_resnet import incept_resnet
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
                    [slim.max_pool2d, slim.avg_pool2d],
                    stride=2, padding='SAME'), \
                slim.arg_scope(
                    [slim.conv2d_transpose],
                    stride=2, padding='SAME',
                    weights_regularizer=slim.l2_regularizer(weight_decay),
                    biases_regularizer=slim.l2_regularizer(weight_decay),
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm), \
                slim.arg_scope(
                    [slim.conv2d],
                    stride=1, padding='SAME',
                    weights_regularizer=slim.l2_regularizer(weight_decay),
                    biases_regularizer=slim.l2_regularizer(weight_decay),
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm):
                with tf.variable_scope('stage128'):
                    sc = 'stage128_image'
                    net = slim.conv2d(input_tensor, 8, 3)
                    net = incept_resnet.conv_maxpool(net, scope=sc)
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                    sc = 'stage64_image'
                    net = incept_resnet.conv_maxpool(net, scope=sc)
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                with tf.variable_scope('stage32'):
                    sc = 'stage32_image'
                    net = incept_resnet.conv_maxpool(net, scope=sc)
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                with tf.variable_scope('stage16'):
                    sc = 'stage16_image'
                    net = incept_resnet.conv_maxpool(net, scope=sc)
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                with tf.variable_scope('stage8'):
                    sc = 'stage_out'
                    net = incept_resnet.pullout8(
                        net, self.out_dim, is_training,
                        scope=sc)
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
        raise ValueError('final_endpoint (%s) not recognized', final_endpoint)

        return net, end_points

    def placeholder_inputs(self, batch_size=None):
        # using different batch size for evaluation and streaming
        # if batch_size is None:
        #     batch_size = self.batch_size
        frames_tf = tf.placeholder(
            tf.float32, shape=(
                batch_size,
                self.crop_size, self.crop_size,
                1))
        poses_tf = tf.placeholder(
            tf.float32, shape=(batch_size, self.out_dim))
        return frames_tf, poses_tf

    def get_loss(self, pred, anno, end_points):
        """ simple sum-of-squares loss
            pred: BxJ
            anno: BxJ
        """
        # loss = tf.reduce_sum(tf.pow(tf.subtract(pred, anno), 2)) / 2
        loss = tf.nn.l2_loss(pred - anno)  # already divided by 2
        # loss = tf.reduce_mean(tf.squared_difference(pred, anno)) / 2
        losses_reg = tf.add_n(tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES))
        return loss + losses_reg

    # @staticmethod
    # def base_arg_scope(is_training,
    #                    bn_decay=0.9997, bn_epsilon=0.001,
    #                    weight_decay=0.00004, activation_fn=tf.nn.relu):
    #     from tensorflow.contrib import slim
    #     with slim.arg_scope(
    #             [slim.batch_norm],
    #             is_training=is_training,
    #             epsilon=bn_epsilon,
    #             # # Make sure updates happen automatically
    #             # updates_collections=None,
    #             # Try zero_debias_moving_mean=True for improved stability.
    #             # zero_debias_moving_mean=True,
    #             decay=bn_decay):
    #                 with slim.arg_scope(
    #                         [slim.dropout],
    #                         is_training=is_training):
    #                         with slim.arg_scope(
    #                                 [slim.fully_connected],
    #                                 weights_regularizer=slim.l2_regularizer(weight_decay),
    #                                 biases_regularizer=slim.l2_regularizer(weight_decay),
    #                                 activation_fn=tf.nn.relu,
    #                                 normalizer_fn=slim.batch_norm):
    #                             with slim.arg_scope(
    #                                     [slim.max_pool2d, slim.avg_pool2d],
    #                                     stride=2, padding='SAME'):
    #                                 with slim.arg_scope(
    #                                         [slim.conv2d],
    #                                         stride=1, padding='SAME',
    #                                         weights_regularizer=slim.l2_regularizer(weight_decay),
    #                                         biases_regularizer=slim.l2_regularizer(weight_decay),
    #                                         activation_fn=tf.nn.relu,
    #                                         normalizer_fn=slim.batch_norm) as scope:
    #                                     return scope
