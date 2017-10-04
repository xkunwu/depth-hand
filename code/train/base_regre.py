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
        self.args.log_dir = os.path.join(
            self.args.log_dir,
            'log-{}'.format(datetime.now().strftime('%y%m%d-%H%M%S'))
        )
        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)
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
        print(tf.shape(anno)[1])
        regre_loss = tf.reduce_sum(tf.pow(tf.subtract(pred, anno), 2))  # / (2 * anno.shape[1])
        regre_loss = tf.reduce_mean(regre_loss)
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
                tf.summary.scalar('regression_loss', loss)

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
                self.logger.info('**** Epoque {:03d} ****'.format(epoch))
                sys.stdout.flush()

                self.logger.info('** Training **')
                self.train_one_epoch(sess, ops, train_writer)
                self.logger.info('** Evaluate **')
                self.eval_one_epoch(sess, ops, test_writer)

                # Save the variables to disk.
                if epoch % 10 == 0:
                    save_path = saver.save(
                        sess, os.path.join(self.args.log_dir, self.args.model_ckpt))
                    self.logger.info("Model saved in file: {}".format(save_path))

    def train_one_epoch(self, sess, ops, train_writer):
        """ ops: dict mapping from string to tf ops """
        is_training = True
        batch_size = self.args.batch_size
        image_size = self.args.img_size
        with open(hands17.annotation_training, 'r') as fanno:
            loss_sum = 0
            batch_count = 0
            while True:
                next_n_lines = list(islice(fanno, batch_size))
                if not next_n_lines:
                    break
                if len(next_n_lines) < batch_size:
                    break
                batch_frame = np.empty(shape=(batch_size, image_size, image_size))
                batch_label = np.empty(shape=(batch_size, self.args.pose_dim))
                for bi, annot_line in enumerate(next_n_lines):
                    img_name, pose_mat, _ = hands17.parse_line_pose(annot_line)
                    img = hands17.read_image(os.path.join(
                        hands17.training_cropped, img_name))
                    batch_frame[bi, :, :] = img
                    batch_label[bi, :] = pose_mat.flatten().T
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
                self.logger.info('training mean loss: {}'.format(loss_sum / batch_count))

    def eval_one_epoch(self, sess, ops, test_writer):
        """ ops: dict mapping from string to tf ops """
        is_training = False
        batch_size = self.args.batch_size
        image_size = self.args.img_size
        with open(hands17.annotation_evaluation, 'r') as fanno:
            loss_sum = 0
            batch_count = 0
            while True:
                next_n_lines = list(islice(fanno, batch_size))
                if not next_n_lines:
                    break
                if len(next_n_lines) < batch_size:
                    break
                batch_frame = np.empty(shape=(batch_size, image_size, image_size))
                batch_label = np.empty(shape=(batch_size, self.args.pose_dim))
                for bi, annot_line in enumerate(next_n_lines):
                    img_name, pose_mat, _ = hands17.parse_line_pose(annot_line)
                    img = hands17.read_image(os.path.join(
                        hands17.training_cropped, img_name))
                    batch_frame[bi, :, :] = img
                    batch_label[bi, :] = pose_mat.flatten().T
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
                self.logger.info('evaluate mean loss: {}'.format(loss_sum / batch_count))

    def evaluate(self, num_eval):
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

        model_path = os.path.join(self.args.log_dir, self.args.model_ckpt)
        saver.restore(sess, model_path)
        self.logger.info("Model restored from: {}.".format(model_path))

        ops = {
            'batch_frame': batch_frame,
            'pose_out': pose_out,
            'is_training': is_training,
            'pred': pred,
            'loss': loss,
        }

        self.eval_one_epoch(sess, ops, num_eval)

    # def eval_one_epoch(sess, ops, num_eval=1):
    #     error_cnt = 0
    #     is_training = False
    #     total_correct = 0
    #     total_seen = 0
    #     loss_sum = 0
    #     total_seen_class = [0 for _ in range(NUM_CLASSES)]
    #     total_correct_class = [0 for _ in range(NUM_CLASSES)]
    #     fout = open(os.path.join(DUMP_DIR, 'pred_label.txt'), 'w')
    #     for fn in range(len(TEST_FILES)):
    #         log_string('----'+str(fn)+'----')
    #         current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
    #         current_data = current_data[:,0:NUM_POINT,:]
    #         current_label = np.squeeze(current_label)
    #         print(current_data.shape)
    #
    #         file_size = current_data.shape[0]
    #         num_batches = file_size // BATCH_SIZE
    #         print(file_size)
    #
    #         for batch_idx in range(num_batches):
    #             start_idx = batch_idx * BATCH_SIZE
    #             end_idx = (batch_idx+1) * BATCH_SIZE
    #             cur_batch_size = end_idx - start_idx
    #
    #             # Aggregating BEG
    #             batch_loss_sum = 0 # sum of losses for the batch
    #             batch_pred_sum = np.zeros((cur_batch_size, NUM_CLASSES)) # score for classes
    #             batch_pred_classes = np.zeros((cur_batch_size, NUM_CLASSES)) # 0/1 for classes
    #             for vote_idx in range(num_votes):
    #                 rotated_data = provider.rotate_point_cloud_by_angle(current_data[start_idx:end_idx, :, :],
    #                                                   vote_idx/float(num_votes) * np.pi * 2)
    #                 feed_dict = {ops['pointclouds_pl']: rotated_data,
    #                              ops['labels_pl']: current_label[start_idx:end_idx],
    #                              ops['is_training_pl']: is_training}
    #                 loss_val, pred_val = sess.run([ops['loss'], ops['pred']],
    #                                           feed_dict=feed_dict)
    #                 batch_pred_sum += pred_val
    #                 batch_pred_val = np.argmax(pred_val, 1)
    #                 for el_idx in range(cur_batch_size):
    #                     batch_pred_classes[el_idx, batch_pred_val[el_idx]] += 1
    #                 batch_loss_sum += (loss_val * cur_batch_size / float(num_votes))
    #             # pred_val_topk = np.argsort(batch_pred_sum, axis=-1)[:,-1*np.array(range(topk))-1]
    #             # pred_val = np.argmax(batch_pred_classes, 1)
    #             pred_val = np.argmax(batch_pred_sum, 1)
    #             # Aggregating END
    #
    #             correct = np.sum(pred_val == current_label[start_idx:end_idx])
    #             # correct = np.sum(pred_val_topk[:,0:topk] == label_val)
    #             total_correct += correct
    #             total_seen += cur_batch_size
    #             loss_sum += batch_loss_sum
    #
    #             for i in range(start_idx, end_idx):
    #                 l = current_label[i]
    #                 total_seen_class[l] += 1
    #                 total_correct_class[l] += (pred_val[i-start_idx] == l)
    #                 fout.write('%d, %d\n' % (pred_val[i-start_idx], l))
    #
    #                 if pred_val[i-start_idx] != l and FLAGS.visu: # ERROR CASE, DUMP!
    #                     img_filename = '%d_label_%s_pred_%s.jpg' % (error_cnt, SHAPE_NAMES[l],
    #                                                            SHAPE_NAMES[pred_val[i-start_idx]])
    #                     img_filename = os.path.join(DUMP_DIR, img_filename)
    #                     output_img = pc_util.point_cloud_three_views(np.squeeze(current_data[i, :, :]))
    #                     scipy.misc.imsave(img_filename, output_img)
    #                     error_cnt += 1
    #
    #     log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    #     log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
    #     log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
    #
    #     class_accuracies = np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float)
    #     for i, name in enumerate(SHAPE_NAMES):
    #         log_string('%10s:\t%0.3f' % (name, class_accuracies[i]))
    #
if __name__ == "__main__":
    argsholder = args_holder()
    argsholder.parse_args()
    trainer = base_regre(argsholder.args)
    trainer.train()
