import os
from importlib import import_module
import numpy as np
import tensorflow as tf
import progressbar
import h5py
import matplotlib.pyplot as mpplot
from cv2 import resize as cv2resize
from utils.coder import file_pack
from utils.iso_boxes import iso_rect
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
        self.anchor_num = 16
        self.crop_range = 480.
        self.num_channel = 1
        self.num_appen = 7
        self.batch_allot = getattr(
            import_module('model.batch_allot'),
            'batch_allot'
        )
        self.store_file = None
        self.batch_data = None
        # receive arguments
        self.args = args
        self.prepare_dir = args.prepare_dir
        self.appen_train = os.path.join(
            self.prepare_dir, 'train_{}'.format(
                self.__class__.__name__))
        self.appen_test = os.path.join(
            self.prepare_dir, 'test_{}'.format(
                self.__class__.__name__))
        self.predict_dir = args.predict_dir
        self.predict_file = os.path.join(
            self.predict_dir, 'predict_{}'.format(
                self.name_desc))
        self.batch_size = args.batch_size
        self.ckpt_path = os.path.join(
            args.out_dir, 'log', 'blinks',
            self.name_desc, 'model.ckpt')

    def tweak_arguments(self, args):
        args.crop_size = self.crop_size
        args.anchor_num = self.anchor_num
        args.crop_range = self.crop_range

    def start_train(self, filepack):
        self.store_file = filepack.push_h5(self.appen_train)
        self.store_size = self.store_file['index'].shape[0]

    def start_epoch_train(self, split_beg, split_end):
        # new round starting from next portion of data
        self.batch_beg = split_end
        self.split_end = split_beg  # + self.store_size

    def start_epoch_valid(self, split_beg, split_end):
        self.batch_beg = split_beg
        self.split_end = split_end \
            if 0 != split_end \
            else self.store_size

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
        self.batch_data = {
            'batch_index': self.store_file['index'][self.batch_beg:batch_end, ...],
            'batch_frame': self.store_file['frame'][self.batch_beg:batch_end, ...],
            'batch_poses': self.store_file['poses'][self.batch_beg:batch_end, ...],
            'batch_resce': self.store_file['resce'][self.batch_beg:batch_end, ...]
        }
        self.batch_beg = batch_end
        return self.batch_data

    def end_train(self):
        pass

    def start_evaluate(self, filepack):
        self.store_file = filepack.push_h5(self.appen_test)
        self.store_size = self.store_file['index'].shape[0]
        self.batch_beg = 0
        self.split_end = self.store_size
        return filepack.write_file(self.predict_file)

    def evaluate_batch(self, writer, pred_val):
        self.write_pred(
            writer, self.caminfo,
            self.batch_data['batch_index'], self.batch_data['batch_resce'],
            pred_val
        )

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
        fig = mpplot.figure(figsize=(2 * 5, 2 * 5))
        # img_id = args.data_draw.draw_prediction_poses(
        #     thedata,
        #     thedata.training_images,
        #     thedata.training_annot_test,
        #     self.predict_file
        # )
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

    def provider_worker(self, line, image_dir, caminfo):
        img_name, pose_raw = self.data_module.io.parse_line_annot(line)
        img = self.data_module.io.read_image(os.path.join(image_dir, img_name))
        img_crop_resize, resce = self.data_module.ops.crop_resize(
            img, pose_raw, caminfo)
        resce3 = resce[0:4]
        pose_local = self.data_module.ops.raw_to_local(pose_raw, resce3)
        index = self.data_module.io.imagename2index(img_name)
        return (index, np.expand_dims(img_crop_resize, axis=-1),
                pose_local.flatten().T, resce)

    def yanker(self, pose_local, resce, caminfo):
        resce3 = resce[0:4]
        return self.data_module.ops.local_to_raw(pose_local, resce3)

    @staticmethod
    def put_worker(
        args, image_dir, model_inst,
            caminfo, data_module, batchallot):
        bi = args[0]
        line = args[1]
        index, frame, poses, resce = model_inst.provider_worker(
            line, image_dir, caminfo)
        batchallot.batch_index[bi, :] = index
        batchallot.batch_frame[bi, ...] = frame
        batchallot.batch_poses[bi, :] = poses
        batchallot.batch_resce[bi, :] = resce

    def write_pred(self, fanno, caminfo,
                   batch_index, batch_resce, batch_poses):
        for ii in range(batch_index.shape[0]):
            img_name = self.data_module.io.index2imagename(batch_index[ii, 0])
            pose_local = batch_poses[ii, :].reshape(-1, 3)
            resce = batch_resce[ii, :]
            pose_raw = self.yanker(pose_local, resce, caminfo)
            fanno.write(
                img_name +
                '\t' + '\t'.join("%.4f" % x for x in pose_raw.flatten()) +
                '\n')

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
                    image_size, image_size,
                    num_channel),
                chunks=(1,
                        image_size, image_size,
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
        batchallot = self.batch_allot(self)
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
        self.logger = args.logger
        self.data_module = args.data_module
        self.out_dim = thedata.join_num * 3
        self.image_dir = thedata.training_images
        self.caminfo = thedata
        self.region_size = thedata.region_size

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
        with h5py.File(self.appen_train, 'r') as h5file:
            store_size = h5file['index'].shape[0]
            frame_id = np.random.choice(store_size)
            # frame_id = 0
            img_id = h5file['index'][frame_id, 0]
            frame_h5 = np.squeeze(h5file['frame'][frame_id, ...], -1)
            poses_h5 = h5file['poses'][frame_id, ...].reshape(-1, 3)
            resce_h5 = h5file['resce'][frame_id, ...]

        print('[{}] drawing image #{:d} ...'.format(self.name_desc, img_id))
        print(np.min(frame_h5), np.max(frame_h5))
        print(np.histogram(frame_h5, range=(1e-4, np.max(frame_h5))))
        print(np.min(poses_h5, axis=0), np.max(poses_h5, axis=0))
        print(resce_h5)
        resce3 = resce_h5[0:4]
        resce2 = resce_h5[4:7]
        mpplot.subplots(nrows=2, ncols=2, figsize=(2 * 5, 2 * 5))

        ax = mpplot.subplot(2, 2, 3)
        mpplot.gca().set_title('test storage read')
        # resize the cropped resion for eazier pose drawing in the commen frame
        sizel = np.floor(resce2[0]).astype(int)
        resce_cp = np.copy(resce2)
        resce_cp[0] = 1
        ax.imshow(
            cv2resize(frame_h5, (sizel, sizel)),
            cmap=mpplot.cm.bone_r)
        pose_raw = args.data_ops.local_to_raw(poses_h5, resce3)
        args.data_draw.draw_pose2d(
            ax, thedata,
            args.data_ops.raw_to_2d(pose_raw, thedata, resce_cp)
        )

        ax = mpplot.subplot(2, 2, 4)
        mpplot.gca().set_title('test output')
        img_name = args.data_io.index2imagename(img_id)
        img = args.data_io.read_image(os.path.join(self.image_dir, img_name))
        ax.imshow(img, cmap=mpplot.cm.bone_r)
        pose_raw = self.yanker(
            poses_h5, resce_h5, self.caminfo)
        args.data_draw.draw_pose2d(
            ax, thedata,
            args.data_ops.raw_to_2d(pose_raw, thedata)
        )
        rect = iso_rect()
        rect.load(resce2)
        rect.draw(ax)

        ax = mpplot.subplot(2, 2, 1)
        mpplot.gca().set_title('test input #{:d}'.format(img_id))
        annot_line = args.data_io.get_line(
            thedata.training_annot_cleaned, img_id)
        img_name, pose_raw = args.data_io.parse_line_annot(annot_line)
        img = args.data_io.read_image(os.path.join(self.image_dir, img_name))
        ax.imshow(img, cmap=mpplot.cm.bone_r)
        args.data_draw.draw_pose2d(
            ax, thedata,
            args.data_ops.raw_to_2d(pose_raw, thedata))

        ax = mpplot.subplot(2, 2, 2)
        mpplot.gca().set_title('test storage write')
        img_name, frame, poses, resce = self.provider_worker(
            annot_line, self.image_dir, thedata)
        frame = np.squeeze(frame, axis=-1)
        poses = poses.reshape(-1, 3)
        if (
                (1e-4 < np.linalg.norm(frame_h5 - frame)) or
                (1e-4 < np.linalg.norm(poses_h5 - poses))
        ):
            print(np.linalg.norm(frame_h5 - frame))
            print(np.linalg.norm(poses_h5 - poses))
            print('ERROR - h5 storage corrupted!')
        resce3 = resce[0:4]
        resce2 = resce[4:7]
        sizel = np.floor(resce2[0]).astype(int)
        resce_cp = np.copy(resce2)
        resce_cp[0] = 1
        ax.imshow(
            cv2resize(frame, (sizel, sizel)),
            cmap=mpplot.cm.bone_r)
        pose_raw = args.data_ops.local_to_raw(poses, resce3)
        args.data_draw.draw_pose2d(
            ax, thedata,
            args.data_ops.raw_to_2d(pose_raw, thedata, resce_cp)
        )

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
