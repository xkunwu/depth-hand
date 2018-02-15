import os
import tensorflow as tf
from functools import reduce
from train.train_abc import train_abc
from utils.image_ops import tfplot_vxlab, tfplot_vxflt


def unravel_index(indices, shape):
    indices = tf.expand_dims(indices, 0)
    shape = tf.expand_dims(shape, 1)
    shape = tf.cast(shape, tf.float32)
    strides = tf.cumprod(shape, reverse=True)
    strides_shifted = tf.cumprod(shape, exclusive=True, reverse=True)
    strides = tf.cast(strides, tf.int32)
    strides_shifted = tf.cast(strides_shifted, tf.int32)

    def even():
        rem = indices - (indices // strides) * strides
        return rem // strides_shifted

    def odd():
        div = indices // strides_shifted
        return div - (div // strides) * strides
    rank = tf.rank(shape)
    return tf.cond(tf.equal(rank - (rank // 2) * 2, 0), even, odd)


class train_voxel_detect(train_abc):
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
            vxflt_pred = pred_op[0, :, joint_id]
            vxlab_echt = poses_op[0, joint_id]
            frame = frames_op[0, ...]
            vxmap_echt_op = tf.expand_dims(tfplot_vxlab(
                frame, vxlab_echt, hmap_size), axis=0)
            tf.summary.image('vxmap_echt/', vxmap_echt_op, max_outputs=1)
            vxmap_pred_op = tf.expand_dims(tfplot_vxflt(
                frame, vxflt_pred, hmap_size), axis=0)
            tf.summary.image('vxflt_pred/', vxmap_pred_op, max_outputs=1)

            tf.summary.histogram(
                'vxhit_value_pred', vxflt_pred)
            vxidx_echt = unravel_index(
                vxlab_echt, (hmap_size, hmap_size, hmap_size))
            vxidx_pred = unravel_index(
                tf.argmax(vxflt_pred, output_type=tf.int32),
                (hmap_size, hmap_size, hmap_size))
            tf.summary.scalar(
                'vxhit_diff_echt',
                tf.reduce_sum(tf.abs(vxidx_echt - vxidx_pred)))
            joint_id = 0
            vxflt_pred = vxflt_pred
            tf.summary.histogram(
                'vxhit_1_value_pred', vxflt_pred)
            vxidx_echt = unravel_index(
                vxlab_echt, (hmap_size, hmap_size, hmap_size))
            vxidx_pred = unravel_index(
                tf.argmax(vxflt_pred, output_type=tf.int32),
                (hmap_size, hmap_size, hmap_size))
            tf.summary.scalar(
                'vxhit_1_diff_echt',
                tf.reduce_sum(tf.abs(vxidx_echt - vxidx_pred)))

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
