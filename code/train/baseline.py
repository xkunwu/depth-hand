import tensorflow as tf
import logging
import os
import sys
from train_abc import train_abc
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(BASE_DIR, '..')
sys.path.append(BASE_DIR)
from args_holder.py import args_holder
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import tf_util


class baseline(train_abc):
    """ This class holds baseline training approach.
    @Attributes:
        args: holds parameters.
    """

    def __init__(self, args):
        self.args = args
        logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
        rootLogger = logging.getLogger()

        fileHandler = logging.FileHandler("{0}/{1}.log".format(logPath, fileName))
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        rootLogger.addHandler(consoleHandler)

    def log_string(out_str):
        LOG_FOUT.write('{}\n'.format(out_str))
        LOG_FOUT.flush()
        print(out_str)

    def get_learning_rate(batch):
        learning_rate = tf.train.exponential_decay(
            BASE_LEARNING_RATE,  # Base learning rate.
            batch * BATCH_SIZE,  # Current index into the dataset.
            DECAY_STEP,          # Decay step.
            DECAY_RATE,          # Decay rate.
            staircase=True
        )
        learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
        return learning_rate

    def get_bn_decay(self, batch):
        bn_momentum = tf.train.exponential_decay(
            BN_INIT_DECAY,
            batch*BATCH_SIZE,
            BN_DECAY_DECAY_STEP,
            BN_DECAY_DECAY_RATE,
            staircase=True
        )
        bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
        return bn_decay

    def get_model(point_cloud, is_training, bn_decay=None):
        """ Classification PointNet, input is BxNx3, output Bx40 """
        batch_size = point_cloud.get_shape()[0].value
        num_point = point_cloud.get_shape()[1].value
        end_points = {}
        input_image = tf.expand_dims(point_cloud, -1)

        # Point functions (MLP implemented as conv2d)
        net = tf_util.conv2d(input_image, 64, [1,3],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv1', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 64, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv2', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 64, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv3', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 128, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv4', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 1024, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv5', bn_decay=bn_decay)

        # Symmetric function: max pooling
        net = tf_util.max_pool2d(net, [num_point,1],
                                 padding='VALID', scope='maxpool')

        # MLP on global point cloud vector
        net = tf.reshape(net, [batch_size, -1])
        net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                      scope='fc1', bn_decay=bn_decay)
        net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                      scope='fc2', bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                              scope='dp1')
        net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')

        return net, end_points

    def get_loss(pred, label, end_points):
        """ pred: B*NUM_CLASSES,
            label: B, """
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
        classify_loss = tf.reduce_mean(loss)
        tf.summary.scalar('classify loss', classify_loss)
        return classify_loss

    def train(self):
        with tf.Graph().as_default():
            with tf.device('/gpu:' + str(self.args.gpu_id)):

                # Note the global_step=batch parameter to minimize.
                # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
                batch = tf.Variable(0)
                bn_decay = self.get_bn_decay(batch)
                tf.summary.scalar('bn_decay', bn_decay)

                # Get model and loss
                pred = self.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
                loss = self.get_loss(pred, labels_pl)
                tf.summary.scalar('loss', loss)

                correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
                accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
                tf.summary.scalar('accuracy', accuracy)

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
                os.path.join(LOG_DIR, 'train'),
                sess.graph)
            test_writer = tf.summary.FileWriter(
                os.path.join(LOG_DIR, 'test'))

            # Init variables
            init = tf.global_variables_initializer()
            sess.run(init)

            ops = {
                'pointclouds_pl': pointclouds_pl,
                'labels_pl': labels_pl,
                'is_training_pl': is_training_pl,
                'pred': pred,
                'loss': loss,
                'train_op': train_op,
                'merged': merged,
                'step': batch
            }

            for epoch in range(MAX_EPOCH):
                log_string('**** EPOCH {:03d} ****'.format(epoch))
                sys.stdout.flush()

                self.train_one_epoch(sess, ops, train_writer)
                self.eval_one_epoch(sess, ops, test_writer)

                # Save the variables to disk.
                if epoch % 10 == 0:
                    save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                    self.log_string("Model saved in file: %s" % save_path)

    def train_one_epoch(self, sess, ops, train_writer):
        """ ops: dict mapping from string to tf ops """
        is_training = True

        # Shuffle train files
        train_file_idxs = np.arange(0, len(TRAIN_FILES))
        np.random.shuffle(train_file_idxs)

        for fn in range(len(TRAIN_FILES)):
            log_string('----' + str(fn) + '-----')
            current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
            current_data = current_data[:,0:NUM_POINT,:]
            current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))
            current_label = np.squeeze(current_label)

            file_size = current_data.shape[0]
            num_batches = file_size // BATCH_SIZE

            total_correct = 0
            total_seen = 0
            loss_sum = 0

            for batch_idx in range(num_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = (batch_idx+1) * BATCH_SIZE

                # Augment batched point clouds by rotation and jittering
                rotated_data = provider.rotate_point_cloud(current_data[start_idx:end_idx, :, :])
                jittered_data = provider.jitter_point_cloud(rotated_data)
                feed_dict = {ops['pointclouds_pl']: jittered_data,
                ops['labels_pl']: current_label[start_idx:end_idx],
                ops['is_training_pl']: is_training,}
                summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
                train_writer.add_summary(summary, step)
                pred_val = np.argmax(pred_val, 1)
                correct = np.sum(pred_val == current_label[start_idx:end_idx])
                total_correct += correct
                total_seen += BATCH_SIZE
                loss_sum += loss_val

                log_string('mean loss: %f' % (loss_sum / float(num_batches)))
                log_string('accuracy: %f' % (total_correct / float(total_seen)))


    def eval_one_epoch(sess, ops, test_writer):
        """ ops: dict mapping from string to tf ops """
        is_training = False
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]

        for fn in range(len(TEST_FILES)):
            log_string('----' + str(fn) + '-----')
            current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
            current_data = current_data[:,0:NUM_POINT,:]
            current_label = np.squeeze(current_label)

            file_size = current_data.shape[0]
            num_batches = file_size // BATCH_SIZE

            for batch_idx in range(num_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = (batch_idx+1) * BATCH_SIZE

                feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                ops['labels_pl']: current_label[start_idx:end_idx],
                ops['is_training_pl']: is_training}
                summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['loss'], ops['pred']], feed_dict=feed_dict)
                pred_val = np.argmax(pred_val, 1)
                correct = np.sum(pred_val == current_label[start_idx:end_idx])
                total_correct += correct
                total_seen += BATCH_SIZE
                loss_sum += (loss_val*BATCH_SIZE)
                for i in range(start_idx, end_idx):
                    l = current_label[i]
                    total_seen_class[l] += 1
                    total_correct_class[l] += (pred_val[i-start_idx] == l)

                    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
                    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
                    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))


if __name__ == "__main__":
    parser = args_holder()
    baseline.train(parser.args)
