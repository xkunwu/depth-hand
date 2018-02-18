import os
import numpy as np
import tensorflow as tf
import progressbar
from functools import reduce
from utils.coder import file_pack
from utils.tf_utils import unravel_index
from train.train_abc import train_abc
from utils.image_ops import tfplot_vxmap, tfplot_uomap3d


class train_voxel_offset(train_abc):
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
            self.args.logger.info(
                'network structure:\n{}'.format(shapestr))
            loss_op = self.args.model_inst.get_loss(
                pred_op, poses_op, end_points)
            # regre_error = tf.sqrt(loss_op * 2)
            regre_error = loss_op
            tf.summary.scalar('regression_error', regre_error)

            learning_rate = self.get_learning_rate(global_step)
            tf.summary.scalar('learning_rate', learning_rate)

            hmap_size = self.args.model_inst.hmap_size
            num_j = self.args.model_inst.out_dim
            joint_id = num_j - 1
            frame = frames_op[0, ..., 0]
            vxdist_echt = poses_op[0, ..., joint_id]
            vxdist_pred = pred_op[0, ..., joint_id]
            uomap_echt = poses_op[
                0, hmap_size / 2, ...,
                num_j + 3 * joint_id:num_j + 3 * (joint_id + 1)]
            uomap_pred = pred_op[
                0, hmap_size / 2, ...,
                num_j + 3 * joint_id:num_j + 3 * (joint_id + 1)]
            vxdist_echt_op = tf.expand_dims(tfplot_vxmap(
                frame, vxdist_echt, hmap_size, reduce_fn=np.max), axis=0)
            tf.summary.image('vxdist_echt/', vxdist_echt_op, max_outputs=1)
            vxdist_pred_op = tf.expand_dims(tfplot_vxmap(
                frame, vxdist_pred, hmap_size, reduce_fn=np.max), axis=0)
            tf.summary.image('vxdist_pred/', vxdist_pred_op, max_outputs=1)
            vxunit_echt_op = tf.expand_dims(tfplot_uomap3d(
                frame, uomap_echt), axis=0)
            tf.summary.image('vxunit_echt/', vxunit_echt_op, max_outputs=1)
            vxunit_pred_op = tf.expand_dims(tfplot_uomap3d(
                frame, uomap_pred), axis=0)
            tf.summary.image('vxunit_pred/', vxunit_pred_op, max_outputs=1)

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
