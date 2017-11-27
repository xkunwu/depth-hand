import os
import sys
from importlib import import_module
from psutil import virtual_memory
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import progressbar
import h5py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
file_pack = getattr(
    import_module('coder'),
    'file_pack'
)
iso_rect = getattr(
    import_module('iso_boxes'),
    'iso_rect'
)
iso_aabb = getattr(
    import_module('iso_boxes'),
    'iso_aabb'
)


class base_regre(object):
    """ This class holds baseline training approach using plain regression.
    """
    def __init__(self):
        self.crop_size = 128
        self.num_channel = 1
        self.num_appen = 7
        self.batchallot = None
        self.batch_size = 0
        self.pose_dim = 0
        self.image_dir = ''
        self.provider = None
        self.provider_worker = None

    class batch_allot:
        def __init__(self, batch_size, image_size, pose_dim, num_channel, num_appen):
            self.batch_size = batch_size
            self.image_size = image_size
            self.pose_dim = pose_dim
            self.num_channel = num_channel
            self.num_appen = num_appen
            batch_data = {
                'batch_index': np.empty(
                    shape=(batch_size, 1), dtype=np.int32),
                'batch_frame': np.empty(
                    shape=(
                        batch_size,
                        image_size, image_size,
                        num_channel),
                    # dtype=np.float32),
                    dtype=float),
                'batch_poses': np.empty(
                    shape=(batch_size, pose_dim),
                    # dtype=np.float32),
                    dtype=float),
                'batch_resce': np.empty(
                    shape=(batch_size, num_appen),
                    # dtype=np.float32),
                    dtype=float),
            }
            self.batch_bytes = \
                batch_data['batch_index'].nbytes + batch_data['batch_frame'].nbytes + \
                batch_data['batch_poses'].nbytes + batch_data['batch_resce'].nbytes
            self.batch_beg = 0

        def allot(self, store_size=-1):
            store_cap_mult = (virtual_memory().total >> 2) // self.batch_bytes
            store_cap = store_cap_mult * self.batch_size
            if 0 > store_size:
                self.store_size = store_cap
            else:
                self.store_size = min(store_cap, store_size)
            self.store_bytes = self.store_size * self.batch_bytes / self.batch_size
            self.store_beg = 0
            self.batch_index = np.empty(
                shape=(self.store_size, 1), dtype=np.int32)
            self.batch_frame = np.empty(
                shape=(
                    self.store_size,
                    self.image_size, self.image_size,
                    self.num_channel),
                # dtype=np.float32)
                dtype=float)
            self.batch_poses = np.empty(
                shape=(self.store_size, self.pose_dim),
                # dtype=np.float32)
                dtype=float)
            self.batch_resce = np.empty(
                shape=(self.store_size, self.num_appen),
                # dtype=np.float32)
                dtype=float)

        def fetch_store(self):
            if self.store_beg >= self.file_size:
                return False
            store_end = min(
                self.store_beg + self.store_size,
                self.file_size
            )
            self.store_size = store_end - self.store_beg
            self.batch_index = self.store_file['index'][self.store_beg:store_end, ...]
            self.batch_frame = self.store_file['frame'][self.store_beg:store_end, ...]
            self.batch_poses = self.store_file['poses'][self.store_beg:store_end, ...]
            self.batch_resce = self.store_file['resce'][self.store_beg:store_end, ...]
            self.store_beg = store_end
            self.batch_beg = 0
            return True

        def assign(self, store_file):
            self.store_file = store_file
            self.file_size = self.store_file['index'].shape[0]
            self.store_size = min(
                self.file_size,
                ((virtual_memory().total >> 1) // self.batch_bytes) * self.batch_size
            )
            self.store_beg = 0
            self.fetch_store()

        def fetch_batch(self):
            # if self.batch_beg >= self.store_size:
            #     if not self.fetch_store():
            #         return None
            # batch_end = min(
            #     self.batch_beg + self.batch_size,
            #     self.store_size
            # )
            batch_end = self.batch_beg + self.batch_size
            if batch_end >= self.store_size:
                if not self.fetch_store():
                    return None
                batch_end = self.batch_beg + self.batch_size
                if batch_end >= self.store_size:
                    return None
            batch_data = {
                'batch_index': self.batch_index[self.batch_beg:batch_end, ...],
                'batch_frame': self.batch_frame[self.batch_beg:batch_end, ...],
                'batch_poses': self.batch_poses[self.batch_beg:batch_end, ...],
                'batch_resce': self.batch_resce[self.batch_beg:batch_end, ...]
            }
            self.batch_beg = batch_end
            return batch_data

    def start_train(self):
        self.batchallot = self.batch_allot(
            self.batch_size, self.crop_size, self.pose_dim,
            self.num_channel, self.num_appen)

    def start_epoch_train(self, filepack):
        self.batchallot.assign(
            filepack.push_h5(self.appen_train)
        )

    def start_epoch_test(self, filepack):
        self.batchallot.assign(
            filepack.push_h5(self.appen_test)
        )

    def fetch_batch_train(self):
        return self.batchallot.fetch_batch()

    def fetch_batch_test(self):
        return self.batchallot.fetch_batch()

    def end_train(self):
        self.batchallot = None

    def start_evaluate(self, filepack):
        self.batchallot = self.batch_allot(
            self.batch_size, self.crop_size, self.pose_dim,
            self.num_channel, self.num_appen)
        return filepack.write_file(self.predict_file)

    def end_evaluate(self):
        self.batchallot = None

    def tweak_args(self, args):
        """ Tweak algorithm specific parameters """
        args.crop_size = self.crop_size

    def prepare_data(self, thedata, args, batchallot, file_annot, name_appen):
        num_line = int(sum(1 for line in file_annot))
        file_annot.seek(0)
        batchallot.allot(num_line)
        store_size = batchallot.store_size
        num_stores = int(np.ceil(float(num_line) / store_size))
        args.logger.debug(
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
        image_size = self.crop_size
        pose_dim = batchallot.pose_dim
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
                (num_line, pose_dim),
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
            self.batch_size, self.crop_size, self.pose_dim,
            self.num_channel, self.num_appen)
        with file_pack() as filepack:
            file_annot = filepack.push_file(thedata.training_annot_train)
            self.prepare_data(thedata, args, batchallot, file_annot, self.appen_train)
        with file_pack() as filepack:
            file_annot = filepack.push_file(thedata.training_annot_test)
            self.prepare_data(thedata, args, batchallot, file_annot, self.appen_test)
        time_e = str(timedelta(seconds=timer() - time_s))
        args.logger.info('data prepared [{}], time: {}'.format(
            self.__class__.__name__, time_e))

    def receive_data(self, thedata, args):
        """ Receive parameters specific to the data """
        self.prepare_dir = args.prepare_dir
        self.appen_train = os.path.join(
            self.prepare_dir, 'train_{}'.format(self.__class__.__name__))
        self.appen_test = os.path.join(
            self.prepare_dir, 'test_{}'.format(self.__class__.__name__))
        self.predict_dir = args.predict_dir
        self.predict_file = os.path.join(
            self.predict_dir, 'predict_{}'.format(self.__class__.__name__))
        self.batch_size = args.batch_size
        self.pose_dim = thedata.join_num * 3
        self.image_dir = thedata.training_images
        self.caminfo = thedata
        self.provider = args.data_provider
        self.provider_worker = self.provider.prow_cropped
        self.yanker = self.provider.yank_cropped

    def draw_random(self, thedata, args):
        import matplotlib.pyplot as mpplot
        from cv2 import resize as cv2resize

        with h5py.File(self.appen_train, 'r') as h5file:
            store_size = h5file['index'].shape[0]
            frame_id = np.random.choice(store_size)
            img_id = h5file['index'][frame_id, 0]
            frame_h5 = np.squeeze(h5file['frame'][frame_id, ...], -1)
            poses_h5 = h5file['poses'][frame_id, ...].reshape(-1, 3)
            resce_h5 = h5file['resce'][frame_id, ...]
            print(np.min(frame_h5), np.max(frame_h5))
            print(np.histogram(frame_h5, range=(1e-4, np.max(frame_h5))))
            print(np.min(poses_h5, axis=0), np.max(poses_h5, axis=0))

        print('[{}] drawing pose #{:d}'.format(self.__class__.__name__, img_id))
        # aabb = iso_aabb(resce_h5[2:5], resce_h5[1])
        # rect = args.data_ops.get_rect2(aabb, thedata)
        # resce2 = np.append(resce_h5[0], rect.cll)
        resce2 = resce_h5[0:3]
        resce3 = resce_h5[3:7]
        mpplot.subplots(nrows=2, ncols=2, figsize=(2 * 5, 2 * 5))

        mpplot.subplot(2, 2, 3)
        sizel = np.floor(resce2[0]).astype(int)
        resce_cp = np.copy(resce2)
        resce_cp[0] = 1
        mpplot.imshow(
            cv2resize(frame_h5, (sizel, sizel)),
            cmap='bone')
        pose_raw = args.data_ops.local_to_raw(poses_h5, resce3)
        args.data_draw.draw_pose2d(
            thedata,
            args.data_ops.raw_to_2d(pose_raw, thedata, resce_cp)
        )

        mpplot.subplot(2, 2, 4)
        img_name = args.data_io.index2imagename(img_id)
        img = args.data_io.read_image(os.path.join(self.image_dir, img_name))
        mpplot.imshow(img, cmap='bone')
        pose_raw = self.yanker(poses_h5, resce_h5)
        args.data_draw.draw_pose2d(
            thedata,
            args.data_ops.raw_to_2d(pose_raw, thedata)
        )

        mpplot.subplot(2, 2, 1)
        annot_line = args.data_io.get_line(
            thedata.training_annot_cleaned, img_id)
        img_name, pose_raw = args.data_io.parse_line_annot(annot_line)
        img = args.data_io.read_image(os.path.join(self.image_dir, img_name))
        mpplot.imshow(img, cmap='bone')
        # rect = iso_rect(resce_h5[1:3], self.crop_size / resce_h5[0])
        rect = iso_rect()
        rect.load(resce2)
        rect.draw()
        args.data_draw.draw_pose2d(
            thedata,
            args.data_ops.raw_to_2d(pose_raw, thedata))

        mpplot.subplot(2, 2, 2)
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
        resce2 = resce[0:3]
        resce3 = resce[3:7]
        sizel = np.floor(resce2[0]).astype(int)
        resce_cp = np.copy(resce2)
        resce_cp[0] = 1
        mpplot.imshow(
            cv2resize(frame, (sizel, sizel)),
            cmap='bone')
        pose_raw = args.data_ops.local_to_raw(poses, resce3)
        args.data_draw.draw_pose2d(
            thedata,
            args.data_ops.raw_to_2d(pose_raw, thedata, resce_cp)
        )
        mpplot.savefig(os.path.join(
            args.predict_dir,
            'draw_{}.png'.format(self.__class__.__name__)))
        mpplot.show()

    def placeholder_inputs(self):
        frames_tf = tf.placeholder(
            tf.float32, shape=(
                self.batch_size,
                self.crop_size, self.crop_size,
                1))
        poses_tf = tf.placeholder(
            tf.float32, shape=(self.batch_size, self.pose_dim))
        return frames_tf, poses_tf

    def get_model(
            self, input_tensor, is_training,
            scope=None, final_endpoint='poseout'):
        """ input_tensor: BxHxWxC
            pose_dim: BxJ, where J is flattened 3D locations
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
                    [slim.max_pool2d, slim.avg_pool2d],
                    stride=1, padding='SAME'), \
                slim.arg_scope(
                    [slim.conv2d],
                    stride=1, padding='SAME', activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm):
                with tf.variable_scope('stage128'):
                    sc = 'stage128'
                    net = slim.conv2d(input_tensor, 16, 3, scope='conv128_3x3_1')
                    net = slim.conv2d(net, 16, 3, stride=2, scope='conv128_3x3_2')
                    net = slim.max_pool2d(net, 3, scope='maxpool128_3x3_1')
                    net = slim.conv2d(net, 32, 3, scope='conv64_3x3_1')
                    net = slim.max_pool2d(net, 3, stride=2, scope='maxpool64_3x3_2')
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                with tf.variable_scope('stage32'):
                    sc = 'stage32'
                    net = slim.conv2d(net, 64, 3, scope='conv32_3x3_1')
                    net = slim.max_pool2d(net, 3, stride=2, scope='maxpool32_3x3_2')
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                with tf.variable_scope('stage16'):
                    sc = 'stage16'
                    net = slim.conv2d(net, 128, 3, scope='conv16_3x3_1')
                    net = slim.max_pool2d(net, 3, stride=2, scope='maxpool16_3x3_2')
                    self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points
                with tf.variable_scope('stage8'):
                    net = slim.flatten(net)
                    sc = 'poseout'
                    net = slim.fully_connected(net, 768, scope='fullconn8')
                    net = slim.dropout(
                        net, 0.5, is_training=is_training, scope='dropout8')
                    net = slim.fully_connected(
                        net, self.pose_dim, activation_fn=None, scope='poseout')
                    # self.end_point_list.append(sc)
                    if add_and_check_final(sc, net):
                        return net, end_points

        raise ValueError('final_endpoint (%s) not recognized', final_endpoint)

        # shapestr = 'input: {}'.format(input_image.shape)
        # net, shapestr = tf_util.conv2d(
        #     input_image, 16, [5, 5], stride=[1, 1], scope='conv1', shapestr=shapestr,
        #     padding='VALID', is_training=is_training, bn=True, bn_decay=bn_decay)
        # net, shapestr = tf_util.max_pool2d(
        #     net, [4, 4], scope='maxpool1', shapestr=shapestr, padding='VALID')
        # net, shapestr = tf_util.conv2d(
        #     net, 32, [3, 3], stride=[1, 1], scope='conv2', shapestr=shapestr,
        #     padding='VALID', is_training=is_training, bn=True, bn_decay=bn_decay)
        # net, shapestr = tf_util.max_pool2d(
        #     net, [2, 2], scope='maxpool2', shapestr=shapestr, padding='VALID')
        # net, shapestr = tf_util.conv2d(
        #     net, 64, [3, 3], stride=[1, 1], scope='conv3', shapestr=shapestr,
        #     padding='VALID', is_training=is_training, bn=True, bn_decay=bn_decay)
        # net, shapestr = tf_util.max_pool2d(
        #     net, [2, 2], scope='maxpool3', shapestr=shapestr, padding='VALID')
        # # print(net.shape)
        #
        # net = tf.reshape(net, [batch_size, -1])
        # net, shapestr = tf_util.fully_connected(
        #     net, 1024, scope='fullconn1', shapestr=shapestr,
        #     is_training=is_training, bn=True, bn_decay=bn_decay)
        # # net = tf_util.conv2d(
        # #     net, 1024, [1, 1], stride=[1, 1], scope='fullconn1',
        # #     padding='VALID', is_training=is_training, bn=True, bn_decay=bn_decay)
        # net, shapestr = tf_util.dropout(
        #     net, keep_prob=0.5, scope='dropout1', shapestr=shapestr, is_training=is_training)
        # net, shapestr = tf_util.fully_connected(
        #     net, self.pose_dim, scope='fullconn3', shapestr=shapestr, activation_fn=None)
        # # net = tf_util.conv2d(
        # #     net, self.pose_dim, [1, 1], stride=[1, 1], scope='fullconn3',
        # #     padding='VALID', is_training=is_training, bn=True, bn_decay=bn_decay)

        # return net, end_points

    @staticmethod
    def get_loss(pred, anno, end_points):
        """ simple sum-of-squares loss
            pred: BxJ
            anno: BxJ
        """
        # loss = tf.reduce_sum(tf.pow(tf.subtract(pred, anno), 2)) / 2
        loss = tf.nn.l2_loss(pred - anno)  # already divided by 2
        # loss = tf.reduce_mean(tf.squared_difference(pred, anno)) / 2
        return loss
