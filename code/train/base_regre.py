import tensorflow as tf
import logging
import os
import sys
import numpy as np
# from train_abc import train_abc
from itertools import islice
from datetime import datetime
from hands17 import hands17
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
sys.path.append(BASE_DIR)
from args_holder import args_holder
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import tf_util


class base_regre():
    """ This class holds baseline training approach using plain regression.
    @Attributes:
        args: holds parameters.
    """

    def make_new_log(self):
        log_time = datetime.now().strftime('%y%m%d-%H%M%S')
        self.log_dir_t = os.path.join(
            self.args.log_dir, 'log-{}'.format(log_time)
        )
        os.makedirs(self.log_dir_t)
        os.symlink(self.log_dir_t, self.log_dir_ln + '-tmp')
        os.rename(self.log_dir_ln + '-tmp', self.log_dir_ln)

    def __init__(self, args, new_log=True):
        self.args = args

        self.log_dir_ln = "{0}/log-base_regre".format(self.args.log_dir)
        if (not os.path.exists(self.log_dir_ln) or new_log):
            self.make_new_log()
        else:
            self.log_dir_t = os.readlink(self.log_dir_ln)

        # add both console and file logging
        logFormatter = logging.Formatter(
            "%(asctime)s [%(levelname)-5.5s]  %(message)s")
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        fileHandler = logging.FileHandler(
            "{0}/{1}".format(self.log_dir_t, args.log_file))
        fileHandler.setFormatter(logFormatter)
        self.logger.addHandler(fileHandler)
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        self.logger.addHandler(consoleHandler)

        hands17.pre_provide(self.args.data_dir, self.args.out_dir)

    def get_learning_rate(self, batch):
        learning_rate = tf.train.exponential_decay(
            self.args.learning_rate,
            batch * self.args.batch_size,
            self.args.decay_step,
            self.args.decay_rate,
            staircase=True
        )
        # learning_rate = tf.maximum(learning_rate, 1e-6)
        return learning_rate

    def get_bn_decay(self, batch):
        """ Generally a value between .95 and .99.
            Lower decay factors tend to weight recent data more heavily.
        """
        bn_momentum = tf.train.exponential_decay(
            self.args.bn_momentum,
            batch * self.args.batch_size,
            float(self.args.decay_step),
            self.args.decay_rate,
            staircase=True
        )
        bn_decay = 1 - tf.maximum(1e-2, bn_momentum)
        return bn_decay

    @staticmethod
    def placeholder_inputs(batch_size, img_size, num_out):
        batch_frame = tf.placeholder(
            tf.float32, shape=(batch_size, img_size, img_size))
        pose_out = tf.placeholder(
            tf.float32, shape=(batch_size, num_out))
        return batch_frame, pose_out

    @staticmethod
    def get_model(batch_frame, pose_dim, is_training, bn_decay=None):
        """ directly predict all joints' location using regression
            batch_frame: BxHxW
            pose_dim: BxJ, where J is flattened 3D locations
        """
        batch_size = batch_frame.get_shape()[0].value
        end_points = {}
        input_image = tf.expand_dims(batch_frame, -1)

        # Point functions (MLP implemented as conv2d)
        net = tf_util.conv2d(
            input_image, 64, [3, 3],
            padding='VALID', stride=[1, 1],
            bn=True, is_training=is_training,
            scope='conv1', bn_decay=bn_decay)
        net = tf_util.max_pool2d(
            net, [1, 1],
            padding='VALID', scope='maxpool1')
        net = tf_util.conv2d(
            net, 256, [3, 3],
            padding='VALID', stride=[1, 1],
            bn=True, is_training=is_training,
            scope='conv2', bn_decay=bn_decay)
        net = tf_util.max_pool2d(
            net, [1, 1],
            padding='VALID', scope='maxpool2')
        net = tf_util.conv2d(
            net, 1024, [3, 3],
            padding='VALID', stride=[1, 1],
            bn=True, is_training=is_training,
            scope='conv3', bn_decay=bn_decay)
        net = tf_util.max_pool2d(
            net, [1, 1],
            padding='VALID', scope='maxpool3')

        # MLP on global point cloud vector
        net = tf.reshape(net, [batch_size, -1])
        net = tf_util.fully_connected(
            net, 512, bn=True, is_training=is_training,
            scope='fc1', bn_decay=bn_decay)
        net = tf_util.dropout(
            net, keep_prob=0.5, is_training=is_training,
            scope='dp1')
        net = tf_util.fully_connected(
            net, pose_dim, activation_fn=None, scope='fc2')

        return net, end_points

    @staticmethod
    def get_loss(pred, anno, end_points):
        """ simple sum-of-squares loss
            pred: BxJ
            anno: BxJ
        """
        # loss = tf.reduce_sum(tf.pow(tf.subtract(pred, anno), 2)) / 2
        # loss = tf.nn.l2_loss(pred - anno)  # already divided by 2
        loss = tf.reduce_mean(tf.squared_difference(pred, anno)) / 2
        return loss

    def train(self):
        with tf.Graph().as_default():
            with tf.device('/gpu:' + str(self.args.gpu_id)):
                batch_frame, pose_out = self.placeholder_inputs(
                    self.args.batch_size, self.args.img_size, self.args.pose_dim)
                is_training = tf.placeholder(tf.bool, shape=())

                # Note the global_step=batch parameter to minimize.
                batch = tf.Variable(0)
                bn_decay = self.get_bn_decay(batch)
                tf.summary.scalar('bn_decay', bn_decay)

                # Get model and loss
                pred, end_points = self.get_model(
                    batch_frame, self.args.pose_dim, is_training, bn_decay=bn_decay)
                loss = self.get_loss(pred, pose_out, end_points)
                regre_error = tf.sqrt(loss * 2)
                tf.summary.scalar('regression_error', regre_error)

                # Get training operator
                learning_rate = self.get_learning_rate(batch)
                tf.summary.scalar('learning_rate', learning_rate)

                optimizer = tf.train.AdamOptimizer(learning_rate)
                train_op = optimizer.minimize(loss, global_step=batch)

                # Add ops to save and restore all the variables.
                saver = tf.train.Saver()

            # Create a session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            sess = tf.Session(config=config)

            # Add summary writers
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(
                os.path.join(self.log_dir_t, 'train'),
                sess.graph)
            test_writer = tf.summary.FileWriter(
                os.path.join(self.log_dir_t, 'test'))

            # Init variables
            init = tf.global_variables_initializer()
            sess.run(init)

            ops = {
                'batch_frame': batch_frame,
                'pose_out': pose_out,
                'is_training': is_training,
                'merged': merged,
                'step': batch,
                'train_op': train_op,
                'loss': loss,
                'pred': pred
            }

            for epoch in range(self.args.max_epoch):
                self.logger.info('**** Epoch #{:03d} ****'.format(epoch))
                sys.stdout.flush()

                time_s = datetime.now()
                self.logger.info('** Training **')
                self.train_one_epoch(sess, ops, train_writer)
                self.logger.info('** Evaluate **')
                self.eval_one_epoch(sess, ops, test_writer)
                self.logger.info('Epoch #{:03d} processing time: {}'.format(
                    epoch,
                    datetime.now() - time_s))

                # Save the variables to disk.
                if epoch % 10 == 0:
                    save_path = saver.save(
                        sess, os.path.join(self.log_dir_t, self.args.model_ckpt))
                    self.logger.info("Model saved in file: {}".format(save_path))

    def train_one_epoch(self, sess, ops, train_writer):
        """ ops: dict mapping from string to tf ops """
        is_training = True
        batch_size = self.args.batch_size
        image_size = self.args.img_size
        batch_frame = np.empty(shape=(batch_size, image_size, image_size))
        batch_poses = np.empty(shape=(batch_size, self.args.pose_dim))
        with open(hands17.training_annot_training, 'r') as fanno:
            # batch_count = 0
            while True:
                next_n_lines = list(islice(fanno, batch_size))
                if not next_n_lines:
                    break
                if len(next_n_lines) < batch_size:
                    break
                for bi, annot_line in enumerate(next_n_lines):
                    img_name, pose_mat, _ = hands17.parse_line_pose(annot_line)
                    img = hands17.read_image(os.path.join(
                        hands17.training_cropped, img_name))
                    batch_frame[bi, :, :] = img
                    batch_poses[bi, :] = pose_mat.flatten().T
                feed_dict = {
                    ops['batch_frame']: batch_frame,
                    ops['pose_out']: batch_poses,
                    ops['is_training']: is_training
                }
                summary, step, _, loss_val, _ = sess.run(
                    [ops['merged'], ops['step'], ops['train_op'],
                        ops['loss'], ops['pred']],
                    feed_dict=feed_dict)
                train_writer.add_summary(summary, step)
                # batch_count += 1
                self.logger.info('batch training loss (half-squared): {}'.format(
                    loss_val))

    def eval_one_epoch(self, sess, ops, test_writer):
        """ ops: dict mapping from string to tf ops """
        is_training = False
        batch_size = self.args.batch_size
        image_size = self.args.img_size
        batch_frame = np.empty(shape=(batch_size, image_size, image_size))
        batch_poses = np.empty(shape=(batch_size, self.args.pose_dim))
        with open(hands17.training_annot_evaluation, 'r') as fanno:
            batch_count = 0
            loss_sum = 0
            while True:
                next_n_lines = list(islice(fanno, batch_size))
                if not next_n_lines:
                    break
                if len(next_n_lines) < batch_size:
                    break
                for bi, annot_line in enumerate(next_n_lines):
                    img_name, pose_mat, _ = hands17.parse_line_pose(annot_line)
                    img = hands17.read_image(os.path.join(
                        hands17.training_cropped, img_name))
                    batch_frame[bi, :, :] = img
                    batch_poses[bi, :] = pose_mat.flatten().T
                feed_dict = {
                    ops['batch_frame']: batch_frame,
                    ops['pose_out']: batch_poses,
                    ops['is_training']: is_training
                }
                summary, step, loss_val, _ = sess.run(
                    [ops['merged'], ops['step'],
                        ops['loss'], ops['pred']],
                    feed_dict=feed_dict)
                test_writer.add_summary(summary, step)
                batch_count += 1
                loss_sum += loss_val
                sys.stdout.write('.')
                sys.stdout.flush()
            print('\n')
            self.logger.info('epoch evaluate mean loss (half-squared): {}'.format(
                loss_sum / batch_count))

    def evaluate(self):
        with tf.device('/gpu:' + str(self.args.gpu_id)):
            batch_frame, pose_out = self.placeholder_inputs(
                self.args.batch_size, self.args.img_size, self.args.pose_dim)
            is_training = tf.placeholder(tf.bool, shape=())

            # Get model and loss
            pred, end_points = self.get_model(
                batch_frame, self.args.pose_dim, is_training)
            loss = self.get_loss(pred, pose_out, end_points)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = True
        sess = tf.Session(config=config)

        # Restore variables from disk.
        model_path = os.path.join(self.log_dir_t, self.args.model_ckpt)
        saver.restore(sess, model_path)
        self.logger.info("Model restored from: {}.".format(model_path))

        ops = {
            'batch_frame': batch_frame,
            'pose_out': pose_out,
            'is_training': is_training,
            'loss': loss,
            'pred': pred
        }

        with open(hands17.training_annot_prediction, 'w') as writer:
            self.eval_one_epoch_write(sess, ops, writer)

    def eval_one_epoch_write(self, sess, ops, writer):
        is_training = False
        batch_size = self.args.batch_size
        image_size = self.args.img_size
        batch_frame = np.empty(shape=(batch_size, image_size, image_size))
        batch_poses = np.empty(shape=(batch_size, self.args.pose_dim))
        batch_resce = np.empty(shape=(batch_size, 3))
        with open(hands17.training_annot_evaluation, 'r') as fanno:
            batch_count = 0
            loss_sum = 0
            while True:
                next_n_lines = list(islice(fanno, batch_size))
                if not next_n_lines:
                    break
                if len(next_n_lines) < batch_size:
                    break
                image_names = []
                for bi, annot_line in enumerate(next_n_lines):
                    img_name, pose_mat, rescen = hands17.parse_line_pose(annot_line)
                    img = hands17.read_image(os.path.join(
                        hands17.training_cropped, img_name))
                    batch_frame[bi, :, :] = img
                    batch_poses[bi, :] = pose_mat.flatten().T
                    image_names.append(img_name)
                    batch_resce[bi, :] = rescen
                feed_dict = {
                    ops['batch_frame']: batch_frame,
                    ops['pose_out']: batch_poses,
                    ops['is_training']: is_training
                }
                loss_val, pred_val = sess.run(
                    [ops['loss'], ops['pred']],
                    feed_dict=feed_dict)
                for bi, _ in enumerate(next_n_lines):
                    out_list = np.append(
                        pred_val[bi, :].flatten(),
                        batch_resce[bi, :].flatten()).flatten()
                    crimg_line = ''.join("%12.4f" % x for x in out_list)
                    writer.write(image_names[bi] + crimg_line + '\n')
                batch_count += 1
                loss_sum += loss_val
                sys.stdout.write('.')
                sys.stdout.flush()
            # print('\n')
            self.logger.info('epoch evaluate mean loss (half-squared): {}'.format(
                loss_sum / batch_count))


if __name__ == "__main__":
    argsholder = args_holder()
    argsholder.parse_args()

    trainer = base_regre(argsholder.args)
    trainer.train()
