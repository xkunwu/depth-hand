import os
import numpy as np
import tensorflow as tf
import progressbar
import h5py
from datetime import datetime
import logging


class train_dense_regre:
    def train(self, restore_step=None):
        with tf.Graph().as_default():
            global_step = tf.train.create_global_step()
            learning_rate = self.get_learning_rate(global_step)
            tf.summary.scalar('learning_rate', learning_rate)

            batches = model.batch_input(model.train_dataset)

            loss = model.loss(*batches)
            tf.summary.scalar('loss', loss)

            if model.is_validate:
                # set batch_size as 3 since tensorboard visualization
                val_batches = model.batch_input(model.val_dataset, 3)
                model.test(*val_batches) # don't need the name

            batchnorm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION)

            saver = tf.train.Saver(tf.global_variables())
            summary_op = tf.summary.merge_all()

            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

            sess = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False))

            sess.run(init_op)
            start_step = 0
            # to resume the training
            if restore_step is not None and restore_step>0:
                checkpoint_path = os.path.join(model.train_dir, 'model.ckpt-%d'%restore_step)
                saver.restore(sess, checkpoint_path)
                start_step = restore_step

            tf.train.start_queue_runners(sess=sess)

            #TODO: change to tf.train.SummaryWriter()
            summary_writer = tf.summary.FileWriter(
                model.summary_dir,
                graph=sess.graph)

            # finally into the training loop
            print('finally into the long long training loop')

            log_path = os.path.join(model.train_dir, 'training_log.txt')
            f = open(log_path, 'a')

            for step in range(start_step, model.max_steps):
                if f.closed:
                    f = open(log_path, 'a')

                start_time = time.time()
                ave_loss = 0
                sess.run(reset_op)
                for sub_step in range(int(accu_steps)):
                    _, _, loss_value = sess.run([accum_op, batchnorm_update_op, loss])
                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                    ave_loss += loss_value

                _ = sess.run([train_op])
                ave_loss /= accu_steps
                duration = time.time() - start_time

                if step%5 == 0:
                    format_str = '[model/train_multi_gpu] %s: step %d/%d, loss = %.3f, %.3f sec/batch, %.3f sec/sample'
                    print(format_str%(datetime.now(), step, model.max_steps, ave_loss, duration, duration/(FLAGS.batch_size*accu_steps)))
                    f.write(format_str%(datetime.now(), step, model.max_steps, ave_loss, duration, duration/(FLAGS.batch_size*accu_steps))+'\n')
                    f.flush()

                if step%20 == 0:
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)


                if step%40 == 0 and hasattr(model, 'do_test'):
                    model.do_test(sess, summary_writer, step)

                if step%100 == 0 or (step+1) == model.max_steps:
                    if not os.path.exists(model.train_dir):
                        os.makedirs(model.train_dir)
                    checkpoint_path = os.path.join(model.train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
                    print('model has been saved to %s\n'%checkpoint_path)
                    f.write('model has been saved to %s\n'%checkpoint_path)
                    f.flush()

            print('finish train')
            f.close()

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
