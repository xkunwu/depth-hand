import os
from importlib import import_module
import numpy as np
import tensorflow as tf
import progressbar
import h5py
import matplotlib.pyplot as mpplot
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
        self.args = args
        self.predict_dir = os.path.join(
            args.out_dir,
            'predict'
        )
        if not os.path.exists(self.predict_dir):
            os.makedirs(self.predict_dir)
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
        self.store_size = self.args.data_inst.num_training

    def start_epoch_train(self):
        # # new round starting from next portion of data
        # self.batch_beg = split_end
        # self.split_end = split_beg  # + self.store_size
        self.batch_beg = 0
        self.split_end = self.args.data_inst.train_valid_split
        print('{:d} images to be trained ...'.format(
            self.split_end - self.batch_beg))

    def start_epoch_valid(self):
        self.batch_beg = self.args.data_inst.train_valid_split
        self.split_end = self.args.data_inst.num_training
        print('{:d} images to be validated ...'.format(
            self.split_end - self.batch_beg))

    def fetch_batch(self, mode='train', fetch_size=None):
        if fetch_size is None:
            fetch_size = self.batch_size
        batch_end = self.batch_beg + fetch_size
        # if batch_end >= self.store_size:
        #     self.batch_beg = batch_end
        #     batch_end = self.batch_beg + fetch_size
        #     self.split_end -= self.store_size
        # # print(self.batch_beg, batch_end, self.split_end)
        # if batch_end > self.split_end:
        if batch_end >= self.split_end:  # BUG: forgot the last one
            return None
        store_handle = self.store_handle[mode]
        self.batch_data['batch_frame'] = np.expand_dims(
            store_handle['crop2'][self.batch_beg:batch_end, ...],
            axis=-1)
        self.batch_data['batch_poses'] = \
            store_handle['pose_c'][self.batch_beg:batch_end, ...]
        self.batch_data['batch_index'] = \
            store_handle['index'][self.batch_beg:batch_end, ...]
        self.batch_data['batch_resce'] = \
            store_handle['resce'][self.batch_beg:batch_end, ...]
        self.batch_beg = batch_end
        return self.batch_data

    def start_evaluate(self):
        self.batch_beg = self.args.data_inst.train_test_split
        self.split_end = self.batch_beg + self.args.data_inst.num_evaluate
        self.store_size = self.args.data_inst.num_evaluate
        print('{:d} images to be evaluated ...'.format(self.store_size))
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
            poses_out[ei] = pose_raw.flatten()
        self.eval_pred.append(poses_out)

    def _draw_image_pose(self, ax, frame, poses, resce, caminfo):
        cube = iso_cube()
        cube.load(resce)
        ax.imshow(frame, cmap=mpplot.cm.bone_r)
        pose2d, _ = cube.raw_to_unit(poses)
        pose2d *= caminfo.crop_size
        self.args.data_draw.draw_pose2d(
            ax, caminfo,
            pose2d,
        )

    def _draw_prediction_poses(self, annot_pred, caminfo):
        batch_beg = self.args.data_inst.train_test_split
        store_handle = self.store_handle['test']
        frame_h5 = store_handle[self.frame_type]
        poses_h5 = store_handle['poses']
        resce_h5 = store_handle['resce']
        with h5py.File(annot_pred, 'r') as pred_h5:
            # if 1000 < batch_beg:
            #     # img_id = 77554
            #     # img_id = 62550
            #     img_id = 82581
            # else:
            #     img_id = 4
            img_id = np.random.randint(1, high=pred_h5['poses'].shape[0])
            ax = mpplot.subplot(2, 4, 1)
            frame = frame_h5[(batch_beg + img_id), ...]
            poses = poses_h5[(batch_beg + img_id), ...].reshape(-1, 3)
            resce = resce_h5[(batch_beg + img_id), ...]
            self._draw_image_pose(
                ax,
                frame, poses, resce,
                caminfo)
            ax.set_title('test image - {:d}'.format(img_id))
            ax = mpplot.subplot(2, 4, 2)
            poses = pred_h5['poses'][img_id, ...].reshape(-1, 3)
            self._draw_image_pose(
                ax,
                frame, poses, resce,
                caminfo)
            print('draw predition #{:d}: {}'.format(
                img_id, self.args.data_io.index2imagename(img_id)))
            img_id = np.random.randint(1, high=pred_h5['poses'].shape[0])
            ax = mpplot.subplot(2, 4, 3)
            frame = frame_h5[(batch_beg + img_id), ...]
            poses = poses_h5[(batch_beg + img_id), ...].reshape(-1, 3)
            resce = resce_h5[(batch_beg + img_id), ...]
            self._draw_image_pose(
                ax,
                frame, poses, resce,
                caminfo)
            ax.set_title('test image - {:d}'.format(img_id))
            ax = mpplot.subplot(2, 4, 4)
            poses = pred_h5['poses'][img_id, ...].reshape(-1, 3)
            self._draw_image_pose(
                ax,
                frame, poses, resce,
                caminfo)
            print('draw predition #{:d}: {}'.format(
                img_id, self.args.data_io.index2imagename(img_id)))
            img_id = np.random.randint(1, high=pred_h5['poses'].shape[0])
            ax = mpplot.subplot(2, 4, 5)
            frame = frame_h5[(batch_beg + img_id), ...]
            poses = poses_h5[(batch_beg + img_id), ...].reshape(-1, 3)
            resce = resce_h5[(batch_beg + img_id), ...]
            self._draw_image_pose(
                ax,
                frame, poses, resce,
                caminfo)
            ax.set_title('test image - {:d}'.format(img_id))
            ax = mpplot.subplot(2, 4, 6)
            poses = pred_h5['poses'][img_id, ...].reshape(-1, 3)
            self._draw_image_pose(
                ax,
                frame, poses, resce,
                caminfo)
            print('draw predition #{:d}: {}'.format(
                img_id, self.args.data_io.index2imagename(img_id)))
            img_id = np.random.randint(1, high=pred_h5['poses'].shape[0])
            ax = mpplot.subplot(2, 4, 7)
            frame = frame_h5[(batch_beg + img_id), ...]
            poses = poses_h5[(batch_beg + img_id), ...].reshape(-1, 3)
            resce = resce_h5[(batch_beg + img_id), ...]
            self._draw_image_pose(
                ax,
                frame, poses, resce,
                caminfo)
            ax.set_title('test image - {:d}'.format(img_id))
            ax = mpplot.subplot(2, 4, 8)
            poses = pred_h5['poses'][img_id, ...].reshape(-1, 3)
            self._draw_image_pose(
                ax,
                frame, poses, resce,
                caminfo)
            print('draw predition #{:d}: {}'.format(
                img_id, self.args.data_io.index2imagename(img_id)))
        return img_id

    def end_evaluate(self, thedata, args):
        poses = np.vstack(self.eval_pred)
        self.eval_pred = []
        num_eval = poses.shape[0]
        batch_beg = self.args.data_inst.train_test_split
        store_handle = self.store_handle['test']
        index = store_handle['index'][
            batch_beg:(batch_beg + num_eval), ...]
        with h5py.File(self.predict_file, 'w') as writer:
            self.args.data_io.write_h5(writer, index, poses)
        # with open(self.predict_file + '.txt', 'w') as writer:
        #     self.args.data_io.write_txt(writer, index, poses)
        print('written annotations for {} test images'.format(num_eval))

        fig = mpplot.figure(figsize=(4 * 5, 2 * 5))
        img_id = self._draw_prediction_poses(
            self.predict_file,
            # self.args.data_inst.annotation_test,
            thedata
        )
        fig.tight_layout()
        fname = 'detection_{}_{:d}.png'.format(self.name_desc, img_id)
        mpplot.savefig(os.path.join(self.predict_dir, fname))
        mpplot.close(fig)
        error_maxj, err_mean = self.args.data_eval.evaluate_poses(
            self.caminfo, self.name_desc,
            self.predict_dir, self.predict_file)
        self.logger.info('maximal per-joint mean error: {}'.format(
            error_maxj
        ))
        self.logger.info('mean error: {}'.format(
            err_mean
        ))

    def _draw_image_pose_compare(
        self, ax, caminfo, frame, resce,
            poses_pred, poses_echt):
        cube = iso_cube()
        cube.load(resce)
        ax.imshow(frame, cmap=mpplot.cm.bone_r)
        pose2d_pred, _ = cube.raw_to_unit(poses_pred)
        pose2d_pred *= caminfo.crop_size
        pose2d_echt, _ = cube.raw_to_unit(poses_echt)
        pose2d_echt *= caminfo.crop_size
        self.args.data_draw.draw_pose2d_compare(
            ax, caminfo,
            pose2d_pred,
            pose2d_echt,
        )

    def detect_write_images(self):
        batch_beg = self.args.data_inst.train_test_split
        outdir_good = os.path.join(self.args.log_dir_t, 'good')
        if not os.path.exists(outdir_good):
            os.makedirs(outdir_good)
        outdir_bad = os.path.join(self.args.log_dir_t, 'bad')
        if not os.path.exists(outdir_bad):
            os.makedirs(outdir_bad)
        with h5py.File(self.predict_file, 'r') as pred_h5:
            fig = mpplot.figure(figsize=(5, 5))
            ax = fig.add_axes([0, 0, 1, 1])
            poses_pred_h5 = pred_h5['poses']
            num_line = poses_pred_h5.shape[0]
            print('start writing {} images ...'.format(num_line))
            timerbar = progressbar.ProgressBar(
                maxval=num_line,
                widgets=[
                    progressbar.Percentage(),
                    ' ', progressbar.Bar('=', '[', ']'),
                    ' ', progressbar.ETA()]
            ).start()
            store_handle = self.store_handle['test']
            frame_h5 = store_handle[self.frame_type]
            poses_echt_h5 = store_handle['poses']
            resce_h5 = store_handle['resce']
            index_h5 = store_handle['index']
            error_cap = (10, 20)
            bad_cnt = 0
            good_cnt = 0
            for li in np.arange(num_line):
                poses_pred = poses_pred_h5[li, ...].reshape(-1, 3)
                poses_echt = poses_echt_h5[(batch_beg + li), ...].reshape(-1, 3)
                error = np.mean(np.sqrt(
                    np.sum((poses_pred - poses_echt) ** 2, axis=1)
                ))
                if error_cap[1] < error:
                    bad_cnt += 1
                    # # self._draw_image_pose(
                    # #     ax,
                    # #     frame_h5[(batch_beg + li), ...],
                    # #     poses_pred,
                    # #     resce_h5[(batch_beg + li), ...],
                    # #     self.caminfo)
                    # self._draw_image_pose_compare(
                    #     ax, self.caminfo,
                    #     frame_h5[(batch_beg + li), ...],
                    #     resce_h5[(batch_beg + li), ...],
                    #     poses_pred, poses_echt)
                    # ax.axis('off')
                    # mpplot.savefig(os.path.join(
                    #     outdir_bad,
                    #     self.args.data_io.index2imagename(index_h5[li])))
                    # ax.clear()
                if error_cap[0] > error:
                    good_cnt += 1
                    # if 0 == (good_cnt % 100):
                    #     # self._draw_image_pose(
                    #     #     ax,
                    #     #     frame_h5[(batch_beg + li), ...],
                    #     #     poses_pred,
                    #     #     resce_h5[(batch_beg + li), ...],
                    #     #     self.caminfo)
                    #     self._draw_image_pose_compare(
                    #         ax, self.caminfo,
                    #         frame_h5[(batch_beg + li), ...],
                    #         resce_h5[(batch_beg + li), ...],
                    #         poses_pred, poses_echt)
                    #     ax.axis('off')
                    #     mpplot.savefig(os.path.join(
                    #         outdir_good,
                    #         self.args.data_io.index2imagename(index_h5[li])))
                    #     ax.clear()
                if 0 == (li % 1000):
                    self._draw_image_pose_compare(
                        ax, self.caminfo,
                        frame_h5[(batch_beg + li), ...],
                        resce_h5[(batch_beg + li), ...],
                        poses_pred, poses_echt)
                    ax.axis('off')
                    mpplot.savefig(os.path.join(
                        outdir_bad,
                        self.args.data_io.index2imagename(index_h5[li])))
                    ax.clear()
                timerbar.update(li)
            timerbar.finish()
            mpplot.close(fig)
            self.logger.info('{} good detections, {} bad detections for error cap [{}]'.format(
                good_cnt, bad_cnt, error_cap))

    def yanker(self, pose_local, resce, caminfo):
        cube = iso_cube()
        cube.load(resce)
        return cube.transform_add_center(pose_local)
        # return cube.transform_expand_move(pose_local)

    def check_dir(self, thedata, args):
        from timeit import default_timer as timer
        from datetime import timedelta
        time_s = timer()
        batchallot = self.batch_allot(self, thedata.num_annotation)
        with file_pack() as filepack:
            for name in self.store_name:
                thedata.prepare_data_recur(
                    name, self.store_name, filepack, batchallot)
        time_e = str(timedelta(seconds=timer() - time_s))
        self.logger.info('data prepared [{}], time: {}'.format(
            self.__class__.__name__, time_e))
        self.logger.info('the following stored data are required: {}'.format(
            list(self.store_name.values())))

        self.store_handle = {
            'train': {},
            'test': {},
        }
        for name, store in self.store_name.items():
            h5name = thedata.prepared_join(store, 'train')
            h5file = args.filepack.push_h5(h5name)
            self.store_handle['train'][name] = h5file[name]
        for name, store in self.store_name.items():
            h5name = thedata.prepared_join(store, 'test')
            h5file = args.filepack.push_h5(h5name)
            self.store_handle['test'][name] = h5file[name]

    def receive_data(self, thedata, args):
        """ Receive parameters specific to the data """
        self.logger = args.logger
        self.data_module = args.data_module
        self.data_inst = thedata
        self.join_num = thedata.join_num
        self.out_dim = self.join_num * 3
        # self.images_train = thedata.training_images
        # self.images_test = thedata.test_images
        self.caminfo = thedata
        self.region_size = thedata.region_size
        # self.train_file = thedata.annotation_train
        # self.test_file = thedata.annotation_test
        self.store_name = {
            'index': thedata.annotation,
            'poses': thedata.annotation,
            'resce': thedata.annotation,
            'pose_c': 'pose_c',
            'crop2': 'crop2_{}'.format(self.crop_size),
        }
        self.frame_type = 'crop2'

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
        # mode = 'train'
        mode = 'test'
        store_handle = self.store_handle[mode]
        index_h5 = store_handle['index']
        store_size = index_h5.shape[0]
        frame_id = np.random.choice(store_size)
        # frame_id = 0  # frame_id = img_id - 1
        # frame_id = 2600
        img_id = index_h5[frame_id, ...]
        frame_h5 = store_handle['crop2'][frame_id, ...]
        poses_h5 = store_handle['pose_c'][frame_id, ...].reshape(-1, 3)
        pose_raw_h5 = store_handle['poses'][frame_id, ...].reshape(-1, 3)
        resce_h5 = store_handle['resce'][frame_id, ...]

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
        fig, _ = mpplot.subplots(nrows=1, ncols=2, figsize=(2 * 5, 1 * 5))

        ax = mpplot.subplot(1, 2, 2)
        mpplot.gca().set_title('test storage read')
        ax.imshow(frame_h5, cmap=mpplot.cm.bone_r)
        # pose3d = poses_h5
        pose3d = cube.trans_scale_to(poses_h5)
        pose2d, _ = cube.project_ortho(pose3d, roll=0, sort=False)
        pose2d *= self.crop_size
        args.data_draw.draw_pose2d(
            ax, thedata,
            pose2d,
        )

        ax = mpplot.subplot(1, 2, 1)
        mpplot.gca().set_title('test image - {:d}'.format(img_id))
        img_name = args.data_io.index2imagename(img_id)
        img = args.data_io.read_image(self.data_inst.images_join(img_name, mode))
        ax.imshow(img, cmap=mpplot.cm.bone_r)
        pose2d = args.data_ops.raw_to_2d(pose_raw_h5, thedata)
        ax.plot(pose2d[:, 1], pose2d[:, 0], 'o')
        pose_raw = self.yanker(poses_h5, resce_h5, thedata)
        print(np.sum(np.abs(pose_raw - pose_raw_h5)))
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
        #     annot_line, image_dir, thedata)
        # frame = np.squeeze(frame, axis=-1)
        # if (
        #         (1e-4 < np.linalg.norm(frame_h5 - frame)) or
        #         (1e-4 < np.linalg.norm(poses_h5 - poses))
        # ):
        #     print(np.linalg.norm(frame_h5 - frame))
        #     print(np.linalg.norm(poses_h5 - poses))
        #     print('ERROR - h5 storage corrupted!')

        fig.tight_layout()
        mpplot.savefig(os.path.join(
            self.predict_dir,
            'draw_{}_{}.png'.format(self.name_desc, img_id)))
        if self.args.show_draw:
            mpplot.show()
        print('[{}] drawing image #{:d} - done.'.format(
            self.name_desc, img_id))

    def get_model(
            self, input_tensor, is_training,
            bn_decay, regu_scale,
            scope=None, final_endpoint='stage_out'):
        """ input_tensor: BxHxWxC
            out_dim: Bx(Jx3), where J is number of joints
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
        from model.incept_resnet import incept_resnet
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

    def get_loss(self, pred, echt, end_points):
        """ simple sum-of-squares loss
            pred: BxJ
            echt: BxJ
        """
        # loss_l2 = tf.reduce_sum(tf.pow(tf.subtract(pred, echt), 2)) / 2
        loss_l2 = tf.nn.l2_loss(pred - echt)  # already divided by 2
        # loss_l2 = tf.reduce_mean(tf.squared_difference(pred, echt)) / 2
        loss_reg = tf.add_n(tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES))
        return loss_l2, loss_reg

    # @staticmethod
    # def base_arg_scope(is_training,
    #                    bn_decay=0.9997, bn_epsilon=0.001,
    #                    regu_scale=0.00004, activation_fn=tf.nn.relu):
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
    #                                 weights_regularizer=slim.l2_regularizer(regu_scale),
    #                                 biases_regularizer=slim.l2_regularizer(regu_scale),
    #                                 activation_fn=tf.nn.relu,
    #                                 normalizer_fn=slim.batch_norm):
    #                             with slim.arg_scope(
    #                                     [slim.max_pool2d, slim.avg_pool2d],
    #                                     stride=2, padding='SAME'):
    #                                 with slim.arg_scope(
    #                                         [slim.conv2d],
    #                                         stride=1, padding='SAME',
    #                                         weights_regularizer=slim.l2_regularizer(regu_scale),
    #                                         biases_regularizer=slim.l2_regularizer(regu_scale),
    #                                         activation_fn=tf.nn.relu,
    #                                         normalizer_fn=slim.batch_norm) as scope:
    #                                     return scope
