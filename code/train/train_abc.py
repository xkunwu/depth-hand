import os
import sys
from importlib import import_module
import logging
import numpy as np
import tensorflow as tf
# from tensorflow.contrib import slim
import progressbar
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
        with file_pack() as filepack:
            time_all_s = timer()
            self.args.model_inst.start_train(filepack)
            for epoch in range(self.args.max_epoch):
                self.logger.info(
                    '**** Epoch #{:03d} ****'.format(epoch))
                sys.stdout.flush()

                split_beg, split_end = \
                    self.args.data_inst.next_split_range()
                # print(split_beg, split_end, split_all)
                # continue

                time_s = timer()
                self.logger.info('** Training **')
                self.args.model_inst.start_epoch_train(
                    split_beg, split_end)
                self.train_one_epoch(sess, ops, train_writer)
                self.logger.info('** Validating **')
                self.args.model_inst.start_epoch_valid(
                    split_beg, split_end)
                mean_loss = self.valid_one_epoch(
                    sess, ops, valid_writer)
                time_e = str(timedelta(
                    seconds=(timer() - time_s)))
                self.args.logger.info(
                    'Epoch #{:03d} processing time: {}'.format(
                        epoch, time_e))
                if mean_loss > valid_loss:
                    self.args.logger.info(
                        'Break due to validation loss starts to grow: {} --> {}'.format(
                            valid_loss, mean_loss))
                    break
                valid_loss = mean_loss
                save_path = saver.save(sess, model_path)
                self.logger.info(
                    'Model saved in file: {}'.format(save_path))
            self.args.model_inst.end_train()
            time_all_e = str(timedelta(
                seconds=(timer() - time_all_s)))
            self.args.logger.info(
                'Total training time: {}'.format(
                    time_all_e))

    def train(self):
        self.logger.info('######## Training ########')
        tf.reset_default_graph()
        with tf.Graph().as_default(), \
                tf.device('/gpu:' + str(self.args.gpu_id)):
            frames_tf, poses_tf = \
                self.args.model_inst.placeholder_inputs()
            is_training_tf = tf.placeholder(
                tf.bool, name='is_training')

            global_step = tf.train.create_global_step()

            pred, end_points = self.args.model_inst.get_model(
                frames_tf, is_training_tf)
            shapestr = 'input: {}'.format(frames_tf.shape)
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
            loss = self.args.model_inst.get_loss(
                pred, poses_tf, end_points)
            # regre_error = tf.sqrt(loss * 2)
            regre_error = loss
            tf.summary.scalar('regression_error', regre_error)

            learning_rate = self.get_learning_rate(global_step)
            tf.summary.scalar('learning_rate', learning_rate)

            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(
                loss, global_step=global_step)

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

                merged = tf.summary.merge_all()
                train_writer = tf.summary.FileWriter(
                    os.path.join(self.args.log_dir_t, 'train'),
                    sess.graph)
                valid_writer = tf.summary.FileWriter(
                    os.path.join(self.args.log_dir_t, 'valid'))

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
                self._train_iter(
                    sess, ops, saver,
                    model_path, train_writer, valid_writer)

    def debug_locat_3d(self, batch_echt, batch_pred):
        np.set_printoptions(
            threshold=np.nan,
            formatter={'float_kind': lambda x: "%.2f" % x})
        anchor_num_sub = self.args.model_inst.anchor_num
        anchor_num = anchor_num_sub ** 3
        pcnt_echt = batch_echt[0, :anchor_num].reshape(
            anchor_num_sub, anchor_num_sub, anchor_num_sub)
        index_echt = np.array(np.unravel_index(
            np.argmax(pcnt_echt), pcnt_echt.shape))
        pcnt_pred = batch_pred[0, :anchor_num].reshape(
            anchor_num_sub, anchor_num_sub, anchor_num_sub)
        index_pred = np.array(np.unravel_index(
            np.argmax(pcnt_pred), pcnt_pred.shape))
        self.logger.info(
            [index_echt, np.max(pcnt_echt), np.sum(pcnt_echt)])
        self.logger.info(
            [index_pred, np.max(pcnt_pred), np.sum(pcnt_pred)])
        anchors_echt = batch_echt[0, anchor_num:].reshape(
            anchor_num_sub, anchor_num_sub, anchor_num_sub, 4)
        anchors_pred = batch_pred[0, anchor_num:].reshape(
            anchor_num_sub, anchor_num_sub, anchor_num_sub, 4)
        self.logger.info([
            anchors_echt[index_echt[0], index_echt[1], index_echt[2], :],
        ])
        self.logger.info([
            anchors_pred[index_pred[0], index_pred[1], index_echt[2], :],
        ])
        z_echt = index_echt[2]
        self.logger.info('\n{}'.format(pcnt_pred[..., z_echt]))
        anchors_diff = np.fabs(
            anchors_echt[..., z_echt, 0:3] -
            anchors_pred[..., z_echt, 0:3])
        self.logger.info('\n{}'.format(
            np.sum(anchors_diff, axis=-1)))
        self.logger.info('\n{}'.format(
            np.fabs(anchors_echt[..., z_echt, 3] - anchors_pred[..., z_echt, 3])))

    def debug_locat_2d(self, batch_echt, batch_pred):
        np.set_printoptions(
            threshold=np.nan,
            formatter={'float_kind': lambda x: "%.2f" % x})
        anchor_num_sub = self.args.model_inst.anchor_num
        anchor_num = anchor_num_sub ** 2
        pcnt_echt = batch_echt[0, :anchor_num].reshape(
            anchor_num_sub, anchor_num_sub)
        index_echt = np.array(np.unravel_index(
            np.argmax(pcnt_echt), pcnt_echt.shape))
        pcnt_pred = batch_pred[0, :anchor_num].reshape(
            anchor_num_sub, anchor_num_sub)
        index_pred = np.array(np.unravel_index(
            np.argmax(pcnt_pred), pcnt_pred.shape))
        self.logger.info(
            [index_echt, np.max(pcnt_echt), np.sum(pcnt_echt)])
        self.logger.info(
            [index_pred, np.max(pcnt_pred), np.sum(pcnt_pred)])
        anchors_echt = batch_echt[0, anchor_num:].reshape(
            anchor_num_sub, anchor_num_sub, 3)
        anchors_pred = batch_pred[0, anchor_num:].reshape(
            anchor_num_sub, anchor_num_sub, 3)
        self.logger.info([
            anchors_echt[index_echt[0], index_echt[1], :],
            # anchors_echt[index_pred[0], index_pred[1], :],
        ])
        self.logger.info([
            # anchors_pred[index_echt[0], index_echt[1], :],
            anchors_pred[index_pred[0], index_pred[1], :],
        ])
        self.logger.info('\n{}'.format(pcnt_pred))
        self.logger.info('\n{}'.format(
            np.fabs(anchors_echt[..., 0:2] - anchors_pred[..., 0:2])))
        self.logger.info('\n{}'.format(
            np.fabs(anchors_echt[..., 2] - anchors_pred[..., 2])))

    def debug_detec_2d(self, batch_echt, batch_pred):
        np.set_printoptions(
            threshold=np.nan,
            formatter={'float_kind': lambda x: "%.2f" % x})
        pcnt_echt = batch_echt[0, :].reshape(21, 3)
        pcnt_pred = batch_pred[0, :].reshape(21, 3)
        self.logger.info(np.concatenate(
            (np.max(pcnt_echt, axis=0), np.min(pcnt_echt, axis=0))
        ))
        self.logger.info(np.concatenate(
            (np.max(pcnt_pred, axis=0), np.min(pcnt_pred, axis=0))
        ))
        self.logger.info('\n{}'.format(pcnt_echt))
        self.logger.info('\n{}'.format(pcnt_pred))
        self.logger.info('\n{}'.format(
            np.fabs(pcnt_echt - pcnt_pred)))

    def debug_prediction(self, frame_h5, resce_h5, pred_val):
        self.args.model_inst._debug_draw_prediction(
            frame_h5, resce_h5, pred_val)

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
            summary, step, _, loss_val, pred_val = sess.run(
                [ops['merged'], ops['step'], ops['train_op'],
                    ops['loss'], ops['pred']],
                feed_dict=feed_dict)
            loss_sum += loss_val / self.args.batch_size
            if batch_count % 10 == 0:
                if 'locor' == self.args.model_inst.net_type:
                    if 2 == self.args.model_inst.net_rank:
                        self.debug_locat_2d(batch_data['batch_poses'], pred_val)
                    elif 3 == self.args.model_inst.net_rank:
                        self.debug_locat_3d(batch_data['batch_poses'], pred_val)
                    did = np.random.randint(0, self.args.batch_size)
                    self.debug_prediction(
                        np.squeeze(batch_data['batch_frame'][did, ...], -1),
                        batch_data['batch_resce'][did, ...],
                        pred_val[did, ...]
                    )
                # elif 'poser' == self.args.model_inst.net_type:
                #     self.debug_detec_2d(batch_data['batch_poses'], pred_val)
                train_writer.add_summary(summary, step)
                self.logger.info(
                    'batch {} training loss: {}'.format(
                        batch_count, loss_val))
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
                # HACK: somehow TF use different strategy
                ops['is_training']: True
            }
            summary, step, loss_val, pred_val = sess.run(
                [ops['merged'], ops['step'],
                    ops['loss'], ops['pred']],
                feed_dict=feed_dict)
            loss_sum += loss_val / self.args.batch_size
            if batch_count % 10 == 0:
                if 'locor' == self.args.model_inst.net_type:
                    if 2 == self.args.model_inst.net_rank:
                        self.debug_locat_2d(
                            batch_data['batch_poses'], pred_val)
                    elif 3 == self.args.model_inst.net_rank:
                        self.debug_locat_3d(
                            batch_data['batch_poses'], pred_val)
                # elif 'poser' == self.args.model_inst.net_type:
                #     self.debug_detec_2d(batch_data['batch_poses'], pred_val)
                valid_writer.add_summary(summary, step)
                self.logger.info(
                    'batch {} validate loss: {}'.format(
                        batch_count, loss_val))
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
            frames_tf, poses_tf = \
                self.args.model_inst.placeholder_inputs()
            is_training_tf = tf.placeholder(
                tf.bool, name='is_training')

            pred, end_points = self.args.model_inst.get_model(
                frames_tf, is_training_tf)
            loss = self.args.model_inst.get_loss(
                pred, poses_tf, end_points)

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
                    'batch_frame': frames_tf,
                    'batch_poses': poses_tf,
                    'is_training': is_training_tf,
                    'loss': loss,
                    'pred': pred
                }

                with file_pack() as filepack:
                    writer = self.args.model_inst.start_evaluate(
                        filepack)
                    self.eval_one_epoch_write(sess, ops, writer)
                self.args.model_inst.end_evaluate(
                    self.args.data_inst, self.args)

    def eval_one_epoch_write(self, sess, ops, writer):
        batch_count = 0
        loss_sum = 0
        num_stores = self.args.model_inst.store_size
        timerbar = progressbar.ProgressBar(
            maxval=num_stores,
            widgets=[
                progressbar.Percentage(),
                ' ', progressbar.Bar('=', '[', ']'),
                ' ', progressbar.ETA()]
        ).start()
        while True:
            batch_data = self.args.model_inst.fetch_batch()
            if batch_data is None:
                break
            feed_dict = {
                ops['batch_frame']: batch_data['batch_frame'],
                ops['batch_poses']: batch_data['batch_poses'],
                # HACK: somehow TF use different strategy
                ops['is_training']: True
            }
            loss_val, pred_val = sess.run(
                [ops['loss'], ops['pred']],
                feed_dict=feed_dict)
            self.args.model_inst.evaluate_batch(
                writer, batch_data, pred_val
            )
            loss_sum += loss_val
            timerbar.update(batch_count)
            batch_count += 1
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
