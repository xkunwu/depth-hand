import os
import sys
from importlib import import_module
import logging
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
sys.path.append(BASE_DIR)
args_holder = getattr(
    import_module('args_holder'),
    'args_holder'
)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
file_pack = getattr(
    import_module('coder'),
    'file_pack'
)


class train_abc():
    """ This is the training class.
        args: holds parameters.
    """

    def train(self):
        self.logger.info('######## Training ########')
        tf.reset_default_graph()
        with tf.Graph().as_default():
            with tf.device('/gpu:' + str(self.args.gpu_id)):
                frames_tf, poses_tf = self.args.model_inst.placeholder_inputs()
                is_training_tf = tf.placeholder(tf.bool, shape=())

                # Note the global_step=batch parameter to minimize.
                batch = tf.Variable(0)
                bn_decay = self.get_bn_decay(batch)
                tf.summary.scalar('bn_decay', bn_decay)

                # Get model and loss
                pred, shapestr, end_points = self.args.model_inst.get_model(
                    frames_tf, is_training_tf, bn_decay=bn_decay)
                self.args.logger.info('network structure:\n{}'.format(shapestr))
                loss = self.args.model_inst.get_loss(pred, poses_tf, end_points)
                regre_error = tf.sqrt(loss * 2)
                tf.summary.scalar('regression_error', regre_error)

                # Get training operator
                learning_rate = self.get_learning_rate(batch)
                tf.summary.scalar('learning_rate', learning_rate)

                optimizer = tf.train.AdamOptimizer(learning_rate)
                train_op = optimizer.minimize(loss, global_step=batch)

                # Add ops to save and restore all the variables.
                saver = tf.train.Saver(max_to_keep=4)

            # Create a session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            with tf.Session(config=config) as sess:
                # Add summary writers
                merged = tf.summary.merge_all()
                train_writer = tf.summary.FileWriter(
                    os.path.join(self.args.log_dir_t, 'train'),
                    sess.graph)
                test_writer = tf.summary.FileWriter(
                    os.path.join(self.args.log_dir_t, 'test'))

                model_path = os.path.join(self.args.log_dir_t, 'model.ckpt')
                # Init variables
                if self.args.retrain:
                    init = tf.global_variables_initializer()
                    sess.run(init)
                else:
                    saver.restore(sess, model_path)
                    self.logger.info('Model restored from: {}.'.format(model_path))

                ops = {
                    'batch_frame': frames_tf,
                    'batch_poses': poses_tf,
                    'is_training': is_training_tf,
                    'merged': merged,
                    'step': batch,
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
                        time_e = "{:0>8}".format(timedelta(seconds=(timer() - time_s)))
                        self.args.logger.info('Epoch #{:03d} processing time: {}'.format(
                            epoch, time_e))

                        # Save the variables to disk.
                        if epoch % 10 == 0:
                            save_path = saver.save(sess, model_path)
                            self.logger.info("Model saved in file: {}".format(save_path))
                    self.args.model_inst.end_train()
                    time_all_e = "{:0>8}".format(timedelta(seconds=(timer() - time_all_s)))
                    self.args.logger.info('Total training time: {}'.format(
                        time_all_e))

    def train_one_epoch(self, sess, ops, train_writer):
        """ ops: dict mapping from string to tf ops """
        is_training = True
        while True:
            batch_data = self.args.model_inst.fetch_batch_train()
            if batch_data is None:
                break
            feed_dict = {
                ops['batch_frame']: batch_data['batch_frame'],
                ops['batch_poses']: batch_data['batch_poses'],
                ops['is_training']: is_training
            }
            summary, step, _, loss_val, _ = sess.run(
                [ops['merged'], ops['step'], ops['train_op'],
                    ops['loss'], ops['pred']],
                feed_dict=feed_dict)
            train_writer.add_summary(summary, step)
            self.logger.info('batch training loss (half-squared): {}'.format(
                loss_val))

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
        mean_loss = loss_sum / batch_count
        self.logger.info('epoch testing mean loss (half-squared): {}'.format(
            mean_loss))

    def evaluate(self):
        self.logger.info('######## Evaluating ########')
        tf.reset_default_graph()
        with tf.device('/gpu:' + str(self.args.gpu_id)):
            frames_tf, poses_tf = self.args.model_inst.placeholder_inputs()
            is_training_tf = tf.placeholder(tf.bool, shape=())

            # Get model and loss
            pred, _, end_points = self.args.model_inst.get_model(
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
            model_path = os.path.join(self.args.log_dir_t, 'model.ckpt')
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
                self.args.model_inst.end_evaluate()

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
            self.args.data_provider.write2d(
                writer,
                self.args.model_inst.yanker,
                batch_data['batch_index'],
                batch_data['batch_resce'],
                pred_val
            )
            batch_count += 1
            loss_sum += loss_val
            sys.stdout.write('.')
            sys.stdout.flush()
        print('\n')
        mean_loss = loss_sum / batch_count
        self.logger.info('epoch evaluate mean loss (half-squared): {}'.format(
            mean_loss))

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
