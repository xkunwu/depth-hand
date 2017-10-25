import os
import sys
from importlib import import_module
import logging
from datetime import datetime
import tensorflow as tf
import numpy as np
from itertools import islice

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
sys.path.append(BASE_DIR)
args_holder = getattr(
    import_module('args_holder'),
    'args_holder'
)


class train_abc():
    """ This is the training class.
        args: holds parameters.
    """

    def train(self):
        with tf.Graph().as_default():
            with tf.device('/gpu:' + str(self.args.gpu_id)):
                batch_frame, pose_out = self.args.model_class.placeholder_inputs(
                    self.args.batch_size,
                    self.args.crop_resize,
                    self.args.data_inst.join_num)
                is_training = tf.placeholder(tf.bool, shape=())

                # Note the global_step=batch parameter to minimize.
                batch = tf.Variable(0)
                bn_decay = self.get_bn_decay(batch)
                tf.summary.scalar('bn_decay', bn_decay)

                # Get model and loss
                pred, end_points = self.args.model_class.get_model(
                    batch_frame, self.args.pose_dim, is_training, bn_decay=bn_decay)
                loss = self.args.model_class.get_loss(pred, pose_out, end_points)
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
                self.test_one_epoch(sess, ops, test_writer)
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
        image_size = self.args.crop_resize
        batch_frame = np.empty(shape=(batch_size, image_size, image_size))
        batch_poses = np.empty(shape=(batch_size, self.args.pose_dim))
        # with open(self.args.data_inst.training_annot_train, 'r') as fanno:
        fanno = self.args.data_provider.read_train(self.args.data_inst)
        try:
            # batch_count = 0
            while True:
                next_n_lines = list(islice(fanno, batch_size))
                if not next_n_lines:
                    break
                if len(next_n_lines) < batch_size:
                    break
                # for bi, annot_line in enumerate(next_n_lines):
                #     img_name, pose_mat, _ = self.args.data_inst.parse_line_pose(
                #         annot_line)
                #     img = self.args.data_inst.read_image(os.path.join(
                #         self.args.data_inst.training_cropped, img_name))
                #     batch_frame[bi, :, :] = img
                #     batch_poses[bi, :] = pose_mat.flatten().T
                self.args.data_provider.put2d(
                    self.args.data_inst, next_n_lines,
                    batch_frame, batch_poses
                )
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
        finally:
            self.args.data_provider.close(self.args.data_inst, fanno)

    def test_one_epoch(self, sess, ops, test_writer):
        """ ops: dict mapping from string to tf ops """
        is_training = False
        batch_size = self.args.batch_size
        image_size = self.args.crop_resize
        batch_frame = np.empty(shape=(batch_size, image_size, image_size))
        batch_poses = np.empty(shape=(batch_size, self.args.pose_dim))
        # with open(self.args.data_inst.training_annot_test, 'r') as fanno:
        fanno = self.args.data_provider.read_test(self.args.data_inst)
        try:
            batch_count = 0
            loss_sum = 0
            while True:
                next_n_lines = list(islice(fanno, batch_size))
                if not next_n_lines:
                    break
                if len(next_n_lines) < batch_size:
                    break
                # for bi, annot_line in enumerate(next_n_lines):
                #     img_name, pose_mat, _ = self.args.data_inst.parse_line_pose(
                #         annot_line)
                #     img = self.args.data_inst.read_image(os.path.join(
                #         self.args.data_inst.training_cropped, img_name))
                #     batch_frame[bi, :, :] = img
                #     batch_poses[bi, :] = pose_mat.flatten().T
                self.args.data_provider.put2d(
                    self.args.data_inst, next_n_lines,
                    batch_frame, batch_poses
                )
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
        finally:
            self.args.data_provider.close(self.args.data_inst, fanno)

    def evaluate(self):
        with tf.device('/gpu:' + str(self.args.gpu_id)):
            batch_frame, pose_out = self.args.model_class.placeholder_inputs(
                self.args.batch_size,
                self.args.crop_resize,
                self.args.data_inst.join_num)
            is_training = tf.placeholder(tf.bool, shape=())

            # Get model and loss
            pred, end_points = self.args.model_class.get_model(
                batch_frame, self.args.pose_dim, is_training)
            loss = self.args.model_class.get_loss(pred, pose_out, end_points)

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

        # with open(self.args.data_inst.training_annot_predict, 'w') as writer:
        self.eval_one_epoch_write(sess, ops)

    def eval_one_epoch_write(self, sess, ops):
        is_training = False
        batch_size = self.args.batch_size
        image_size = self.args.crop_resize
        batch_frame = np.empty(shape=(batch_size, image_size, image_size))
        batch_poses = np.empty(shape=(batch_size, self.args.pose_dim))
        batch_resce = np.empty(shape=(batch_size, 3))
        # with open(self.args.data_inst.training_annot_test, 'r') as fanno:
        fanno = self.args.data_provider.read_test(self.args.data_inst)
        writer = self.args.data_provider.write_predict(self.args.data_inst)
        try:
            batch_count = 0
            loss_sum = 0
            while True:
                next_n_lines = list(islice(fanno, batch_size))
                if not next_n_lines:
                    break
                if len(next_n_lines) < batch_size:
                    break
                image_names = []
                # for bi, annot_line in enumerate(next_n_lines):
                #     img_name, pose_mat, rescen = self.args.data_inst.parse_line_pose(
                #         annot_line)
                #     img = self.args.data_inst.read_image(os.path.join(
                #         self.args.data_inst.training_cropped, img_name))
                #     batch_frame[bi, :, :] = img
                #     batch_poses[bi, :] = pose_mat.flatten().T
                #     image_names.append(img_name)
                #     batch_resce[bi, :] = rescen
                self.args.data_provider.put2d(
                    self.args.data_inst, next_n_lines,
                    batch_frame, batch_poses, image_names, batch_resce
                )
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
        finally:
            self.args.data_provider.close(self.args.data_inst, fanno)

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

    def make_new_log(self):
        log_time = datetime.now().strftime('%y%m%d-%H%M%S')
        # git_hash = subprocess.check_output(
        #     ['git', 'rev-parse', '--short', 'HEAD'])
        self.log_dir_t = os.path.join(
            self.args.log_dir, 'log-{}'.format(log_time)
        )
        os.makedirs(self.log_dir_t)
        os.symlink(self.log_dir_t, self.log_dir_ln + '-tmp')
        os.rename(self.log_dir_ln + '-tmp', self.log_dir_ln)

    def __init__(self, args, new_log=True):
        self.args = args
        self.log_dir_ln = "{}/log-{}".format(
            self.args.log_dir, self.args.model_name)
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
            "{0}/{1}".format(self.log_dir_t, self.args.log_file))
        fileHandler.setFormatter(logFormatter)
        self.logger.addHandler(fileHandler)
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        self.logger.addHandler(consoleHandler)


if __name__ == "__main__":
    argsholder = args_holder()
    argsholder.parse_args()
    ARGS = argsholder.args
    ARGS.batch_size = 16
    ARGS.max_epoch = 1
    argsholder.create_instance()
    trainer = train_abc(argsholder.args)
    trainer.train()
