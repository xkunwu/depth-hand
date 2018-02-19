import os
# from importlib import import_module
import tensorflow as tf
from functools import reduce
from train.train_abc import train_abc
from utils.image_ops import tfplot_hmap2, tfplot_olmap, tfplot_uomap


class train_dense_regre(train_abc):
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
                frames_op, is_training_tf,
                self.args.bn_decay, self.args.regu_scale)
            shapestr = 'input: {}'.format(frames_op.shape)
            for ends in self.args.model_inst.end_point_list:
                net = end_points[ends]
                shapestr += '\n{}: {} = ({}, {})'.format(
                    ends, net.shape,
                    net.shape[0],
                    reduce(lambda x, y: x * y, net.shape[1:])
                )
            self.args.logger.info(
                'network structure:\n{}'.format(shapestr))
            loss_op = self.args.model_inst.get_loss(
                pred_op, poses_op, end_points)
            # regre_error = tf.sqrt(loss_op * 2)
            regre_error = loss_op
            tf.summary.scalar('regression_error', regre_error)

            learning_rate = self.get_learning_rate(global_step)
            tf.summary.scalar('learning_rate', learning_rate)

            num_j = self.args.model_inst.out_dim
            joint_id = num_j - 1
            frame = frames_op[0, ..., 0]
            hmap2_echt = poses_op[0, ..., joint_id]
            hmap2_pred = pred_op[0, ..., joint_id]
            olmap_echt = poses_op[0, ..., num_j + joint_id]
            olmap_pred = pred_op[0, ..., num_j + joint_id]
            uomap_echt = poses_op[
                0, ...,
                2 * num_j + 3 * joint_id:2 * num_j + 3 * (joint_id + 1)]
            uomap_pred = pred_op[
                0, ...,
                2 * num_j + 3 * joint_id:2 * num_j + 3 * (joint_id + 1)]
            hmap2_echt_op = tf.expand_dims(tfplot_hmap2(
                frame, hmap2_echt), axis=0)
            tf.summary.image('hmap2_echt/', hmap2_echt_op, max_outputs=1)
            hmap2_pred_op = tf.expand_dims(tfplot_hmap2(
                frame, hmap2_pred), axis=0)
            tf.summary.image('hmap2_pred/', hmap2_pred_op, max_outputs=1)
            olmap_echt_op = tf.expand_dims(tfplot_olmap(
                frame, olmap_echt), axis=0)
            tf.summary.image('olmap_echt/', olmap_echt_op, max_outputs=1)
            olmap_pred_op = tf.expand_dims(tfplot_olmap(
                frame, olmap_pred), axis=0)
            tf.summary.image('olmap_pred/', olmap_pred_op, max_outputs=1)
            uomap_echt_op = tf.expand_dims(tfplot_uomap(
                frame, uomap_echt), axis=0)
            tf.summary.image('uomap_echt/', uomap_echt_op, max_outputs=1)
            uomap_pred_op = tf.expand_dims(tfplot_uomap(
                frame, uomap_pred), axis=0)
            tf.summary.image('uomap_pred/', uomap_pred_op, max_outputs=1)

            num_j = self.args.model_inst.out_dim
            tf.summary.histogram(
                'hmap2_value_echt', poses_op[..., :num_j])
            tf.summary.histogram(
                'hmap2_value_pred', pred_op[..., :num_j])
            tf.summary.histogram(
                'olmap_value_echt', poses_op[..., num_j:num_j * 2])
            tf.summary.histogram(
                'olmap_value_pred', pred_op[..., num_j:num_j * 2])
            tf.summary.histogram(
                'uomap_value_echt', poses_op[..., - num_j * 3:])
            tf.summary.histogram(
                'uomap_value_pred', pred_op[..., - num_j * 3:])

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
