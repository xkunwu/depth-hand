import tensorflow as tf
import logging
import os
import sys
import numpy as np
# from train_abc import train_abc
from hands17 import hands17
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(BASE_DIR, '..')
sys.path.append(BASE_DIR)
from args_holder import args_holder
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import tf_util


class base_regre():
    """ This class holds baseline training approach using plain regression.
    @Attributes:
        args: holds parameters.
    """

    def __init__(self, args):
        self.args = args

        # add both console and file logging
        logFormatter = logging.Formatter(
            "%(asctime)s [%(levelname)-5.5s]  %(message)s")
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        fileHandler = logging.FileHandler(
            "{0}/{1}".format(args.log_dir, args.log_file))
        fileHandler.setFormatter(logFormatter)
        self.logger.addHandler(fileHandler)
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        self.logger.addHandler(consoleHandler)

        hands17.pre_provide(self.args.data_dir)

    def get_learning_rate(self, batch):
        learning_rate = tf.train.exponential_decay(
            self.args.learning_rate,
            batch * self.args.batch_size,
            self.args.decay_step,
            self.args.decay_rate,
            staircase=True
        )
        learning_rate = tf.maximum(learning_rate, 0.00001)
        return learning_rate

    def get_bn_decay(self, batch):
        bn_momentum = tf.train.exponential_decay(
            0.5,
            batch * self.args.batch_size,
            float(self.args.decay_step),
            self.args.decay_rate,
            staircase=True
        )
        bn_decay = tf.minimum(0.99, 1 - bn_momentum)
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
        """ Classification PointNet, input is BxHxW, output BxF """
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
        regre_loss = tf.reduce_sum(tf.pow(tf.subtract(pred, anno), 2)) / (2 * anno.shape[1])
        tf.summary.scalar('regression loss', regre_loss)
        return regre_loss

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
                tf.summary.scalar('loss', loss)

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
                os.path.join(self.args.log_dir, 'train'),
                sess.graph)
            test_writer = tf.summary.FileWriter(
                os.path.join(self.args.log_dir, 'test'))

            # Init variables
            init = tf.global_variables_initializer()
            sess.run(init)

            ops = {
                'batch_frame': batch_frame,
                'pose_out': pose_out,
                'is_training': is_training,
                'pred': pred,
                'loss': loss,
                'train_op': train_op,
                'merged': merged,
                'step': batch
            }

            for epoch in range(self.args.max_epoch):
                self.logger.info('**** EPOCH {:03d} ****'.format(epoch))
                sys.stdout.flush()

                self.train_one_epoch(sess, ops, train_writer)
                self.eval_one_epoch(sess, ops, test_writer)

                # Save the variables to disk.
                if epoch % 10 == 0:
                    save_path = saver.save(
                        sess, os.path.join(self.args.log_dir, "model.ckpt"))
                    self.logger.info("Model saved in file: %s".format(save_path))

    def train_one_epoch(self, sess, ops, train_writer):
        """ ops: dict mapping from string to tf ops """
        is_training = True

        with open(hands17.training_annot_cropped, 'r') as fanno:
            loss_sum = 0
            batch_count = 0
            while True:
                batch_frame = np.empty(shape=(self.args.batch_size, self.args.img_size, self.args.img_size))
                batch_label = np.empty(shape=(self.args.batch_size, self.args.pose_dim))
                for bi in range(self.args.batch_size):
                    annot_line = fanno.readline()
                    if not annot_line:
                        bi = -1
                        break
                    img_name, pose2d = hands17.parse_line_pose(annot_line)
                    img = hands17.read_image(os.path.join(
                        hands17.training_cropped, img_name))
                    batch_frame[bi, :, :] = img
                    batch_label[bi, :] = pose2d.flatten().T
                feed_dict = {
                    ops['batch_frame']: batch_frame,
                    ops['pose_out']: batch_label,
                    ops['is_training']: is_training
                }
                summary, step, _, loss_val, pred_val = sess.run(
                    [ops['merged'], ops['step'],
                        ops['train_op'], ops['loss'], ops['pred']],
                    feed_dict=feed_dict)
                train_writer.add_summary(summary, step)
                pred_val = np.argmax(pred_val, 1)
                loss_sum += loss_val
                batch_count += 1
                self.logger.info('mean loss: {}'.format(loss_sum / batch_count))
                if 0 > bi:
                    break

    def eval_one_epoch(self, sess, ops, test_writer):
        """ ops: dict mapping from string to tf ops """
        is_training = False

        with open(hands17.evaluate_annot_cropped, 'r') as fanno:
            loss_sum = 0
            batch_count = 0
            while True:
                batch_frame = np.empty(shape=(self.args.batch_size, self.args.img_size, self.args.img_size))
                batch_label = np.empty(shape=(self.args.batch_size, self.args.pose_dim))
                for bi in range(self.args.batch_size):
                    annot_line = fanno.readline()
                    if not annot_line:
                        bi = -1
                        break
                    img_name, pose2d = hands17.parse_line_pose(annot_line)
                    img = hands17.read_image(os.path.join(
                        hands17.evaluate_cropped, img_name))
                    batch_frame[bi, :, :] = img
                    batch_label[bi, :] = pose2d.flatten().T
                feed_dict = {
                    ops['batch_frame']: batch_frame,
                    ops['pose_out']: batch_label,
                    ops['is_training']: is_training
                }
                summary, step, loss_val, pred_val = sess.run(
                    [ops['merged'], ops['step'],
                        ops['loss'], ops['pred']],
                    feed_dict=feed_dict)
                test_writer.add_summary(summary, step)
                pred_val = np.argmax(pred_val, 1)
                loss_sum += loss_val
                batch_count += 1
                self.logger.info('mean loss: {}'.format(loss_sum / batch_count))
                if 0 > bi:
                    break


if __name__ == "__main__":
    argsholder = args_holder()
    argsholder.parse_args()
    trainer = base_regre(argsholder.args)
    trainer.train()
