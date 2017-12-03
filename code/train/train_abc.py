import os
import sys
from importlib import import_module
import logging
import numpy as np
import tensorflow as tf
# from tensorflow.contrib import slim
from functools import reduce

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
sys.path.append(BASE_DIR)
args_holder = getattr(
    import_module('args_holder'),
    'args_holder'
)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
file_pack = getattr(
    import_module('utils.coder'),
    'file_pack'
)


class train_abc():
    """ This is the training class.
        args: holds parameters.
    """

    @staticmethod
    def transform_image_summary(tensor, ii=0):
        """ tensor: BxIxIxF """
        isize = int(tensor.shape[1])
        fsize = int(tensor.shape[3])
        num = np.ceil(np.sqrt(int(tensor.shape[-1]))).astype(int)
        t0 = tf.expand_dims(tf.transpose(tensor[ii, ...], [2, 0, 1]), axis=-1)
        # return t0
        pad = tf.zeros([num * num - fsize, isize, isize, 1])
        t1 = tf.unstack(tf.concat(axis=0, values=[t0, pad]), axis=0)
        rows = []
        for ri in range(num):
            rows.append(tf.concat(
                axis=0, values=t1[ri * num:(ri + 1) * num]))
        t2 = tf.concat(axis=1, values=rows)
        return tf.expand_dims(t2, axis=0)

    def train(self):
        self.logger.info('######## Training ########')
        tf.reset_default_graph()
        with tf.Graph().as_default(), tf.device('/gpu:' + str(self.args.gpu_id)):
            frames_tf, poses_tf = self.args.model_inst.placeholder_inputs()
            is_training_tf = tf.placeholder(tf.bool, shape=())

            # Note global_step is the batch to minimize.
            # batch = tf.Variable(0)
            global_step = tf.train.create_global_step()

            # Get model and loss
            pred, end_points = self.args.model_inst.get_model(
                frames_tf, is_training_tf)
            shapestr = 'input: {}'.format(frames_tf.shape)
            for ends in self.args.model_inst.end_point_list:
                net = end_points[ends]
                shapestr += '\n{}: {} = ({}, {})'.format(
                    ends, net.shape,
                    net.shape[0], reduce(lambda x, y: x * y, net.shape[1:])
                )
                if 2 == self.args.model_inst.net_rank:
                    tf.summary.image(ends, self.transform_image_summary(net))
            self.args.logger.info('network structure:\n{}'.format(shapestr))
            loss = self.args.model_inst.get_loss(pred, poses_tf, end_points)
            regre_error = tf.sqrt(loss * 2)
            tf.summary.scalar('regression_error', regre_error)

            # Get training operator
            learning_rate = self.get_learning_rate(global_step)
            tf.summary.scalar('learning_rate', learning_rate)

            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=global_step)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver(max_to_keep=4)

            # Create a session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            with tf.Session(config=config) as sess:
                model_path = self.args.model_inst.ckpt_path
                # Init variables
                if self.args.retrain:
                    init = tf.global_variables_initializer()
                    sess.run(init)
                else:
                    saver.restore(sess, model_path)
                    self.logger.info('Model restored from: {}.'.format(model_path))

                # Add summary writers
                merged = tf.summary.merge_all()
                train_writer = tf.summary.FileWriter(
                    os.path.join(self.args.log_dir_t, 'train'),
                    sess.graph)
                test_writer = tf.summary.FileWriter(
                    os.path.join(self.args.log_dir_t, 'test'))

                ops = {
                    'batch_frame': frames_tf,
                    'batch_poses': poses_tf,
                    'is_training': is_training_tf,
                    'merged': merged,
                    'step': global_step,
                    'train_op': train_op,
                    'loss': loss,
                    'pred': pred
                }

                from timeit import default_timer as timer
                from datetime import timedelta
                with file_pack() as filepack:
                    time_all_s = timer()
                    self.args.model_inst.start_train()
                    for epoch in range(self.args.max_epoch):
                        self.logger.info('**** Epoch #{:03d} ****'.format(epoch))
                        sys.stdout.flush()

                        time_s = timer()
                        self.logger.info('** Training **')
                        self.args.model_inst.start_epoch_train(filepack)
                        self.train_one_epoch(sess, ops, train_writer)
                        self.logger.info('** Testing **')
                        self.args.model_inst.start_epoch_test(filepack)
                        self.test_one_epoch(sess, ops, test_writer)
                        time_e = str(timedelta(seconds=(timer() - time_s)))
                        self.args.logger.info('Epoch #{:03d} processing time: {}'.format(
                            epoch, time_e))

                        # Save the variables to disk.
                        if epoch % 10 == 0:
                            save_path = saver.save(sess, model_path)
                            self.logger.info("Model saved in file: {}".format(save_path))
                    self.args.model_inst.end_train()
                    time_all_e = str(timedelta(seconds=(timer() - time_all_s)))
                    self.args.logger.info('Total training time: {}'.format(
                        time_all_e))

    def train_one_epoch(self, sess, ops, train_writer):
        """ ops: dict mapping from string to tf ops """
        is_training = True
        batch_count = 0
        loss_sum = 0
        while True:
            batch_data = self.args.model_inst.fetch_batch_train()
            if batch_data is None:
                break
            feed_dict = {
                ops['batch_frame']: batch_data['batch_frame'],
                ops['batch_poses']: batch_data['batch_poses'],
                ops['is_training']: is_training
            }
            summary, step, _, loss_val, pred_val = sess.run(
                [ops['merged'], ops['step'], ops['train_op'],
                    ops['loss'], ops['pred']],
                feed_dict=feed_dict)
            loss_sum += loss_val
            batch_count += 1
            if batch_count % 10 == 0:
                train_writer.add_summary(summary, step)
                # np.set_printoptions(
                #     threshold=np.nan,
                #     formatter={'float_kind': lambda x: "%.2f" % x})
                # to_show = batch_data['batch_poses'][0, :256]
                # self.logger.info([np.argmax(to_show), np.max(to_show), np.sum(to_show)])
                # self.logger.info(batch_data['batch_poses'][0, ...])
                # to_show = pred_val[0, :256]
                # self.logger.info([np.argmax(to_show), np.max(to_show), np.sum(to_show)])
                # self.logger.info(pred_val[0, ...])
                # self.logger.info('batch {} training loss (half-squared): {}'.format(
                #     batch_count, loss_val))
                # print(np.sum((batch_data['batch_poses'] - pred_val) ** 2) / 2)
        mean_loss = loss_sum / batch_count
        self.args.logger.info('epoch training mean loss (half-squared): {:.4f}'.format(
            mean_loss))

    def test_one_epoch(self, sess, ops, test_writer):
        """ ops: dict mapping from string to tf ops """
        is_training = False
        batch_count = 0
        loss_sum = 0
        while True:
            batch_data = self.args.model_inst.fetch_batch_test()
            if batch_data is None:
                break
            feed_dict = {
                ops['batch_frame']: batch_data['batch_frame'],
                ops['batch_poses']: batch_data['batch_poses'],
                ops['is_training']: is_training
            }
            summary, step, loss_val, pred_val = sess.run(
                [ops['merged'], ops['step'],
                    ops['loss'], ops['pred']],
                feed_dict=feed_dict)
            loss_sum += loss_val
            batch_count += 1
            self.logger.info('batch {} testing loss (half-squared): {}'.format(
                batch_count, loss_val))
            # print(np.sum((batch_data['batch_poses'] - pred_val) ** 2) / 2)
            if batch_count % 10 == 0:
                test_writer.add_summary(summary, step)
                sys.stdout.write('.')
                sys.stdout.flush()
        print('\n')
        mean_loss = loss_sum / batch_count
        self.args.logger.info('epoch testing mean loss (half-squared): {:.4f}'.format(
            mean_loss))

    def evaluate(self):
        self.logger.info('######## Evaluating ########')
        tf.reset_default_graph()
        with tf.device('/gpu:' + str(self.args.gpu_id)):
            frames_tf, poses_tf = self.args.model_inst.placeholder_inputs()
            is_training_tf = tf.placeholder(tf.bool, shape=())

            # Get model and loss
            pred, end_points = self.args.model_inst.get_model(
                frames_tf, is_training_tf)
            loss = self.args.model_inst.get_loss(pred, poses_tf, end_points)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        with tf.Session(config=config) as sess:
            # Restore variables from disk.
            model_path = self.args.model_inst.ckpt_path
            saver.restore(sess, model_path)
            self.logger.info('Model restored from: {}.'.format(model_path))

            ops = {
                'batch_frame': frames_tf,
                'batch_poses': poses_tf,
                'is_training': is_training_tf,
                'loss': loss,
                'pred': pred
            }

            with file_pack() as filepack:
                writer = self.args.model_inst.start_evaluate(filepack)
                self.args.model_inst.start_epoch_test(filepack)
                self.eval_one_epoch_write(sess, ops, writer)
            self.args.model_inst.end_evaluate(
                self.args.data_inst, self.args)

    def eval_one_epoch_write(self, sess, ops, writer):
        is_training = False
        batch_count = 0
        loss_sum = 0
        while True:
            batch_data = self.args.model_inst.fetch_batch_test()
            if batch_data is None:
                break
            feed_dict = {
                ops['batch_frame']: batch_data['batch_frame'],
                ops['batch_poses']: batch_data['batch_poses'],
                ops['is_training']: is_training
            }
            loss_val, pred_val = sess.run(
                [ops['loss'], ops['pred']],
                feed_dict=feed_dict)
            self.args.model_inst.evaluate_batch(
                writer, batch_data, pred_val
            )
            loss_sum += loss_val
            batch_count += 1
            if batch_count % 10 == 0:
                sys.stdout.write('.')
                sys.stdout.flush()
        print('\n')
        mean_loss = loss_sum / batch_count
        self.args.logger.info('epoch evaluate mean loss (half-squared): {:.4f}'.format(
            mean_loss))

    def get_learning_rate(self, global_step):
        learning_rate = tf.train.exponential_decay(
            self.args.learning_rate,
            global_step,
            self.args.decay_step,
            self.args.decay_rate,
            staircase=True
        )
        # learning_rate = tf.maximum(learning_rate, 1e-6)
        return learning_rate

    def __init__(self, args, new_log=True):
        self.args = args
        self.logger = logging.getLogger('train')


if __name__ == "__main__":
    # python train_abc.py --max_epoch=1 --batch_size=16 --model_name=base_regre
    with args_holder() as argsholder:
        argsholder.parse_args()
        argsholder.create_instance()
        trainer = train_abc(argsholder.args)
        trainer.train()
        trainer.evaluate()
