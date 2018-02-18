import os
import sys
# from importlib import import_module
import logging
import numpy as np
import tensorflow as tf
import progressbar
from functools import reduce
from args_holder import args_holder
from utils.coder import file_pack


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
        t0 = tf.expand_dims(
            tf.transpose(tensor[ii, ...], [2, 0, 1]), axis=-1)
        pad = tf.zeros([num * num - fsize, isize, isize, 1])
        t1 = tf.unstack(
            tf.concat(axis=0, values=[t0, pad]), axis=0)
        rows = []
        for ri in range(num):
            rows.append(tf.concat(
                axis=0, values=t1[ri * num:(ri + 1) * num]))
        t2 = tf.concat(axis=1, values=rows)
        return tf.expand_dims(t2, axis=0)

    def _train_iter(self, sess, ops, saver,
                    model_path, train_writer, valid_writer):
        valid_loss = np.inf
        from timeit import default_timer as timer
        from datetime import timedelta
        epoch = 0
        time_all_s = timer()
        self.args.model_inst.start_train()
        while epoch < self.args.max_epoch:
            epoch += 1
            self.logger.info(
                '**** Epoch #{:03d} ****'.format(epoch))
            sys.stdout.flush()

            # split_beg, split_end = \
            #     self.args.data_inst.next_valid_split()
            # # print(split_beg, split_end)

            time_s = timer()
            self.logger.info('** Training **')
            self.args.model_inst.start_epoch_train()
            self.train_one_epoch(sess, ops, train_writer)
            self.logger.info('** Validating **')
            self.args.model_inst.start_epoch_valid()
            mean_loss = self.valid_one_epoch(
                sess, ops, valid_writer)
            time_e = str(timedelta(
                seconds=(timer() - time_s)))
            self.args.logger.info(
                'Epoch #{:03d} processing time: {}'.format(
                    epoch, time_e))
            if mean_loss > (valid_loss * 1.1):
                self.args.logger.info(
                    'Break due to validation loss starts to grow: {} --> {}'.format(
                        valid_loss, mean_loss))
                break
            elif mean_loss > valid_loss:
                self.args.logger.info(
                    'NOTE: validation loss starts to grow: {} --> {}'.format(
                        valid_loss, mean_loss))
            else:
                # only save model when validation loss decrease
                valid_loss = mean_loss
                save_path = saver.save(sess, model_path)
                self.logger.info(
                    'Model saved in file: {}'.format(save_path))
        self.args.model_inst.end_train()
        time_all_e = timer() - time_all_s
        self.args.logger.info(
            'Total training time: {} for {:d} epoches, average: {}.'.format(
                str(timedelta(seconds=time_all_e)), epoch,
                str(timedelta(seconds=(time_all_e / epoch)))))

    def train(self):
        self.logger.info('######## Training ########')
        tf.reset_default_graph()
        with tf.Graph().as_default(), \
                tf.device('/gpu:' + str(self.args.gpu_id)):
            frames_op, poses_op = \
                self.args.model_inst.placeholder_inputs()
            is_training_tf = tf.placeholder(
                tf.bool, shape=(), name='is_training')

            global_step = tf.train.create_global_step()

            pred_op, end_points = self.args.model_inst.get_model(
                frames_op, is_training_tf, self.args.bn_decay)
            shapestr = 'input: {}'.format(frames_op.shape)
            for ends in self.args.model_inst.end_point_list:
                net = end_points[ends]
                shapestr += '\n{}: {} = ({}, {})'.format(
                    ends, net.shape,
                    net.shape[0],
                    reduce(lambda x, y: x * y, net.shape[1:])
                )
                if (2 == self.args.model_inst.net_rank and
                        'image' in ends):
                    tf.summary.image(
                        ends, self.transform_image_summary(net))
            self.args.logger.info(
                'network structure:\n{}'.format(shapestr))
            loss_op = self.args.model_inst.get_loss(
                pred_op, poses_op, end_points)
            # regre_error = tf.sqrt(loss_op * 2)
            regre_error = loss_op
            tf.summary.scalar('regression_error', regre_error)

            learning_rate = self.get_learning_rate(global_step)
            tf.summary.scalar('learning_rate', learning_rate)

            tf.summary.histogram('out_value_echt', poses_op)
            tf.summary.histogram('out_value_pred', pred_op)

            # from model.base_clean import tfplot_pose_pred
            # frame = frames_op[0, ...]
            # pose_pred = pred_op[0, ...]
            # resce_op = tf.placeholder(tf.float32, shape=(None, 4,))
            # resce = resce_op[0, ...]
            # pose_pred_op = tf.expand_dims(tfplot_pose_pred(
            #     frame, pose_pred, resce,
            #     self.args.data_draw.draw_pose2d,
            #     self.args.data_inst), axis=0)
            # tf.summary.image('pose_pred/', pose_pred_op, max_outputs=1)

            optimizer = tf.train.AdamOptimizer(learning_rate)
            # train_op = optimizer.minimize(
            #     loss_op, global_step=global_step)
            from tensorflow.contrib import slim
            train_op = slim.learning.create_train_op(
                loss_op, optimizer,
                # summarize_gradients=True,
                update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS),
                global_step=global_step)
            # from tensorflow.python.ops import control_flow_ops
            # train_op = slim.learning.create_train_op(
            #     loss_op, optimizer, global_step=global_step)
            # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # if update_ops:
            #     updates = tf.group(*update_ops)
            #     loss_op = control_flow_ops.with_dependencies(
            #         [updates], loss_op)

            saver = tf.train.Saver(max_to_keep=4)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            with tf.Session(config=config) as sess:
                model_path = self.args.model_inst.ckpt_path
                if self.args.retrain:
                    init = tf.global_variables_initializer()
                    sess.run(init)
                else:
                    saver.restore(sess, model_path)
                    self.logger.info(
                        'model restored from: {}.'.format(
                            model_path))

                summary_op = tf.summary.merge_all()
                train_writer = tf.summary.FileWriter(
                    os.path.join(self.args.log_dir_t, 'train'),
                    sess.graph)
                valid_writer = tf.summary.FileWriter(
                    os.path.join(self.args.log_dir_t, 'valid'))

                ops = {
                    'batch_frame': frames_op,
                    'batch_poses': poses_op,
                    # 'batch_resce': resce_op,
                    'is_training': is_training_tf,
                    'summary_op': summary_op,
                    'step': global_step,
                    'train_op': train_op,
                    'loss_op': loss_op,
                    'pred_op': pred_op
                }
                self._train_iter(
                    sess, ops, saver,
                    model_path, train_writer, valid_writer)

    def train_one_epoch(self, sess, ops, train_writer):
        """ ops: dict mapping from string to tf ops """
        batch_count = 0
        loss_sum = 0
        while True:
            batch_data = self.args.model_inst.fetch_batch()
            if batch_data is None:
                break
            feed_dict = {
                ops['batch_frame']: batch_data['batch_frame'],
                ops['batch_poses']: batch_data['batch_poses'],
                ops['is_training']: True
            }
            # sess.run(ops['batch_resce'], feed_dict={
            #     ops['batch_resce']: batch_data['batch_resce']})
            summary, step, _, loss_val, pred_val = sess.run(
                [ops['summary_op'], ops['step'], ops['train_op'],
                    ops['loss_op'], ops['pred_op']],
                feed_dict=feed_dict)
            loss_sum += loss_val / self.args.batch_size
            if batch_count % 10 == 0:
                if 'locor' == self.args.model_inst.net_type:
                    self.args.model_inst.debug_compare(
                        pred_val, self.logger)
                    did = np.random.randint(0, self.args.batch_size)
                    self.args.model_inst._debug_draw_prediction(
                        did, pred_val[did, ...]
                    )
                # elif 'poser' == self.args.model_inst.net_type:
                #     self.args.model_inst.debug_compare(
                #         pred_val, self.logger)
                self.logger.info(
                    'batch {} training loss: {}'.format(
                        batch_count, loss_val))
            if batch_count % 100 == 0:
                train_writer.add_summary(summary, step)
            batch_count += 1
        mean_loss = loss_sum / batch_count
        self.args.logger.info(
            'epoch training mean loss: {:.4f}'.format(
                mean_loss))
        return mean_loss

    def valid_one_epoch(self, sess, ops, valid_writer):
        """ ops: dict mapping from string to tf ops """
        batch_count = 0
        loss_sum = 0
        while True:
            batch_data = self.args.model_inst.fetch_batch()
            if batch_data is None:
                break
            feed_dict = {
                ops['batch_frame']: batch_data['batch_frame'],
                ops['batch_poses']: batch_data['batch_poses'],
                ops['is_training']: False
            }
            summary, step, loss_val, pred_val = sess.run(
                [ops['summary_op'], ops['step'],
                    ops['loss_op'], ops['pred_op']],
                feed_dict=feed_dict)
            loss_sum += loss_val / self.args.batch_size
            if batch_count % 10 == 0:
                if 'locor' == self.args.model_inst.net_type:
                    self.args.model_inst.debug_compare(
                        pred_val, self.logger)
                # elif 'poser' == self.args.model_inst.net_type:
                #     self.args.model_inst.debug_compare(
                #         pred_val, self.logger)
                self.logger.info(
                    'batch {} validate loss: {}'.format(
                        batch_count, loss_val))
            if batch_count % 100 == 0:
                valid_writer.add_summary(summary, step)
            batch_count += 1
        mean_loss = loss_sum / batch_count
        self.args.logger.info(
            'epoch validate mean loss: {:.4f}'.format(
                mean_loss))
        return mean_loss

    def evaluate(self):
        self.logger.info('######## Evaluating ########')
        tf.reset_default_graph()
        with tf.Graph().as_default(), \
                tf.device('/gpu:' + str(self.args.gpu_id)):
            # sequential evaluate, suited for streaming
            frames_op, poses_op = \
                self.args.model_inst.placeholder_inputs(1)
            is_training_tf = tf.placeholder(
                tf.bool, shape=(), name='is_training')

            pred_op, end_points = self.args.model_inst.get_model(
                frames_op, is_training_tf, self.args.bn_decay)
            loss_op = self.args.model_inst.get_loss(
                pred_op, poses_op, end_points)

            saver = tf.train.Saver()

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            with tf.Session(config=config) as sess:
                model_path = self.args.model_inst.ckpt_path
                self.logger.info(
                    'restoring model from: {} ...'.format(model_path))
                saver.restore(sess, model_path)
                self.logger.info('model restored.')

                ops = {
                    'batch_frame': frames_op,
                    'batch_poses': poses_op,
                    'is_training': is_training_tf,
                    'loss_op': loss_op,
                    'pred_op': pred_op
                }

                self.args.model_inst.start_evaluate()
                self.eval_one_epoch_write(sess, ops)
                self.args.model_inst.end_evaluate(
                    self.args.data_inst, self.args)

    def eval_one_epoch_write(self, sess, ops):
        batch_count = 0
        loss_sum = 0
        num_stores = self.args.model_inst.store_size
        eval_size = 1
        timerbar = progressbar.ProgressBar(
            maxval=num_stores,
            widgets=[
                progressbar.Percentage(),
                ' ', progressbar.Bar('=', '[', ']'),
                ' ', progressbar.ETA()]
        ).start()
        while True:
            batch_data = self.args.model_inst.fetch_batch(eval_size)
            if batch_data is None:
                break
            feed_dict = {
                ops['batch_frame']: batch_data['batch_frame'],
                ops['batch_poses']: batch_data['batch_poses'],
                ops['is_training']: False
            }
            loss_val, pred_val = sess.run(
                [ops['loss_op'], ops['pred_op']],
                feed_dict=feed_dict)
            self.args.model_inst.evaluate_batch(pred_val)
            loss_sum += loss_val
            batch_count += eval_size
            timerbar.update(batch_count)
        timerbar.finish()
        mean_loss = loss_sum / batch_count
        self.args.logger.info(
            'epoch evaluate mean loss: {:.4f}'.format(
                mean_loss))
        return mean_loss

    def get_learning_rate(self, global_step):
        learning_rate = tf.train.exponential_decay(
            self.args.learning_rate,
            global_step,
            self.args.decay_step,
            self.args.decay_rate,
            staircase=True
        )
        learning_rate = tf.maximum(learning_rate, 1e-6)
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
