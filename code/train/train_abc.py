import os
import sys
from importlib import import_module
import logging
from timeit import default_timer as timer
from datetime import datetime
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
        tf.reset_default_graph()
        with tf.Graph().as_default():
            with tf.device('/gpu:' + str(self.args.gpu_id)):
                frames_tf, poses_tf = self.args.model_inst.placeholder_inputs(
                    self.args.batch_size,
                    self.args.crop_size,
                    self.args.model_inst.pose_dim)
                is_training_tf = tf.placeholder(tf.bool, shape=())

                # Note the global_step=batch parameter to minimize.
                batch = tf.Variable(0)
                bn_decay = self.get_bn_decay(batch)
                tf.summary.scalar('bn_decay', bn_decay)

                # Get model and loss
                pred, end_points = self.args.model_inst.get_model(
                    frames_tf, self.args.model_inst.pose_dim, is_training_tf, bn_decay=bn_decay)
                loss = self.args.model_inst.get_loss(pred, poses_tf, end_points)
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
            with tf.Session(config=config) as sess:
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
                    'batch_frame': frames_tf,
                    'batch_poses': poses_tf,
                    'is_training': is_training_tf,
                    'merged': merged,
                    'step': batch,
                    'train_op': train_op,
                    'loss': loss,
                    'pred': pred
                }

                self.args.model_inst.start_train(self.args.batch_size)
                for epoch in range(self.args.max_epoch):
                    self.logger.info('**** Epoch #{:03d} ****'.format(epoch))
                    sys.stdout.flush()

                    time_s = timer()
                    self.args.model_inst.start_epoch_train()
                    self.logger.info('** Training **')
                    self.train_one_epoch(sess, ops, train_writer)
                    self.args.model_inst.start_epoch_test()
                    self.logger.info('** Testing **')
                    self.test_one_epoch(sess, ops, test_writer)
                    self.logger.info('Epoch #{:03d} processing time: {}'.format(
                        epoch,
                        timer() - time_s))

                    # Save the variables to disk.
                    if epoch % 10 == 0:
                        save_path = saver.save(
                            sess, os.path.join(self.log_dir_t, self.args.model_ckpt))
                        self.logger.info("Model saved in file: {}".format(save_path))

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
            # batch_count += 1
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
        self.logger.info('epoch evaluate mean loss (half-squared): {}'.format(
            loss_sum / batch_count))

    def evaluate(self):
        tf.reset_default_graph()
        with tf.device('/gpu:' + str(self.args.gpu_id)):
            frames_tf, poses_tf = self.args.model_inst.placeholder_inputs(
                self.args.batch_size,
                self.args.crop_size,
                self.args.model_inst.pose_dim)
            is_training_tf = tf.placeholder(tf.bool, shape=())

            # Get model and loss
            pred, end_points = self.args.model_inst.get_model(
                frames_tf, self.args.model_inst.pose_dim, is_training_tf)
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
            model_path = os.path.join(self.log_dir_t, self.args.model_ckpt)
            saver.restore(sess, model_path)
            self.logger.info("Model restored from: {}.".format(model_path))

            ops = {
                'batch_frame': frames_tf,
                'batch_poses': poses_tf,
                'is_training': is_training_tf,
                'loss': loss,
                'pred': pred
            }

            self.args.model_inst.start_train(self.args.batch_size)
            self.args.model_inst.start_epoch_test()
            self.eval_one_epoch_write(sess, ops)

    def eval_one_epoch_write(self, sess, ops):
        is_training = False
        batch_count = 0
        loss_sum = 0
        with file_pack() as filepack:
            predict_file = os.path.join(
                self.args.data_inst.predict_dir, self.args.model_inst.predict_file)
            writer = filepack.push_file(predict_file, 'w')
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
                # for bi, _ in enumerate(next_n_lines):
                #     out_list = np.append(
                #         pred_val[bi, :].flatten(),
                #         batch_resce[bi, :].flatten()).flatten()
                #     crimg_line = ''.join("%12.4f" % x for x in out_list)
                #     writer.write(image_names[bi] + crimg_line + '\n')
                batch_count += 1
                loss_sum += loss_val
                sys.stdout.write('.')
                sys.stdout.flush()
            print('\n')
            self.logger.info('epoch evaluate mean loss (half-squared): {}'.format(
                loss_sum / batch_count))

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
    def write_args(log_dir, args):
        with open(os.path.join(log_dir, 'args.txt'), 'w') as writer:
            for arg in vars(args):
                writer.write('--{}={}\n'.format(arg, getattr(args, arg)))
                # print(arg, getattr(args, arg))

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
        blinks = os.path.join(self.args.log_dir, 'blinks')
        if not os.path.exists(blinks):
            os.makedirs(blinks)
        self.log_dir_ln = os.path.join(
            blinks, self.args.model_name)
        if (not os.path.exists(self.log_dir_ln) or new_log):
            self.make_new_log()
        else:
            self.log_dir_t = os.readlink(self.log_dir_ln)
        self.write_args(self.log_dir_t, args)

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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)
        logging.shutdown()
        del self.logger
        if exc_type is not None:
            print(exc_type, exc_value, exc_traceback)
        return self


if __name__ == "__main__":
    # python train_abc.py --max_epoch=1 --batch_size=16 --store_level=6 --model_name=base_regre
    argsholder = args_holder()
    argsholder.parse_args()
    argsholder.create_instance()
    trainer = train_abc(argsholder.args)
    trainer.train()
    trainer.evaluate()
