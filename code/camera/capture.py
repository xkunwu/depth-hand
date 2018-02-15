import os
import sys
from importlib import import_module
import numpy as np
# from matplotlib import pyplot as mpplot
# from matplotlib.colors import NoNorm
import tensorflow as tf
# from tensorflow.contrib import slim
import matplotlib.pyplot as mpplot
from colour import Color
import cv2
import pyrealsense as pyrs
import time
# from multiprocessing import Queue, Pool

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
sys.path.append(BASE_DIR)
args_holder = getattr(
    import_module('args_holder'),
    'args_holder'
)
iso_cube = getattr(
    import_module('utils.iso_boxes'),
    'iso_cube'
)


class capture:
    class caminfo_ir:
        image_size = [480, 640]
        z_range = (100., 1060.)
        region_size = 120
        anchor_num = 8
        # intrinsic paramters of Intel Realsense SR300
        # self.fx, self.fy = 463.889, 463.889
        # self.cx, self.cy = 320, 240
        focal = (463.889, 463.889)
        centre = (320, 240)

        def __init__():
            pass

    def __init__(self, args):
        self.args = args

    @staticmethod
    def read_frame_from_device(dev):
        dev.wait_for_frames()
        # img_rgb = dev.colour
        # img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        depth = dev.depth * dev.depth_scale * 1000
        return depth

    def show_results_stream(self, img, cube, index, confidence):
        img = np.minimum(img, self.args.data_inst.z_range[1])
        img = (img - img.min()) / (img.max() - img.min())
        img = np.uint8(img * 255)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        print('hand at: ({:d}, {:d}), confidence: {:.4f}'.format(
            index[0], index[1], confidence
        ))
        if 0.9 < confidence:
            rects = cube.proj_rects_3(
                self.args.data_ops.raw_to_2d,
                # self.args.data_inst
                self.caminfo_ir
            )
            colors = [Color('orange').rgb, Color('red').rgb, Color('lime').rgb]
            for ii, rect in enumerate(rects):
                cll = np.floor(rects[ii].cll + 0.5).astype(int)
                ctr = np.floor(rects[ii].cll + rects[ii].sidelen + 0.5).astype(int)
                cv2.rectangle(
                    img,
                    (cll[1], cll[0]),
                    (ctr[1], ctr[0]),
                    tuple(c * 255 for c in colors[ii][::-1]), 2)
        img = np.flip(img, axis=1)
        cv2.imshow('ouput', img)

    def show_results(self, img, cube):
        mpplot.clf()
        ax = mpplot.subplot(1, 2, 1, projection='3d')
        points3 = self.args.data_ops.img_to_raw(img, self.caminfo_ir)
        numpts = points3.shape[0]
        if 1000 < numpts:
            samid = np.random.choice(numpts, 1000, replace=False)
            points3_sam = points3[samid, :]
        else:
            points3_sam = points3
        ax.scatter(
            points3_sam[:, 0], points3_sam[:, 1], points3_sam[:, 2],
            color=Color('lightsteelblue').rgb)
        ax.view_init(azim=-90, elev=-75)
        ax.set_zlabel('depth (mm)', labelpad=15)
        corners = cube.get_corners()
        iso_cube.draw_cube_wire(ax, corners)
        ax = mpplot.subplot(1, 2, 2)
        ax.imshow(img, cmap=mpplot.cm.bone_r)
        rects = cube.proj_rects_3(
            self.args.data_ops.raw_to_2d,
            # self.args.data_inst
            self.caminfo_ir
        )
        colors = [Color('orange').rgb, Color('red').rgb, Color('lime').rgb]
        for ii, rect in enumerate(rects):
            rect.draw(ax, colors[ii])
            rect.show_dims()
        mpplot.tight_layout()
        mpplot.show()

    def preprocess_input(self, depth):
        # depth = depth[:, ::-1]  # flip
        return depth

    def detect_region(self, depth, sess, ops):
        depth_rescale = self.args.data_ops.resize_localizer(
            depth, self.args.data_inst)
        feed_dict = {
            ops['batch_frame']: self.args.model_inst.convert_input(
                depth_rescale, self.args, self.args.data_inst
            ),
            ops['is_training']: True
        }
        pred_val = sess.run(
            ops['pred'],
            feed_dict=feed_dict)
        cube, index, confidence = self.args.model_inst.convert_output(
            pred_val, self.args, self.caminfo_ir)
        return cube, index, confidence

    def capture_detect(self, serv, dev):
        tf.reset_default_graph()
        with tf.device('/gpu:' + str(self.args.gpu_id)):
            frames_tf, _ = self.args.model_inst.placeholder_inputs(1)
            is_training_tf = tf.placeholder(tf.bool, shape=())
            pred, end_points = self.args.model_inst.get_model(
                frames_tf, is_training_tf, self.args.bn_decay)
            saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        with tf.Session(config=config) as sess:
            model_path = self.args.model_inst.ckpt_path
            saver.restore(sess, model_path)

            mpplot.subplots(nrows=1, ncols=2, figsize=(6, 6 * 2))
            while True:
                depth = self.read_frame_from_device(dev)
                depth = self.preprocess_input(depth)
                ops = {
                    'batch_frame': frames_tf,
                    'is_training': is_training_tf,
                    'pred': pred
                }
                cube, index, confidence = self.detect_region(
                    depth, sess, ops)
                # cube = iso_cube(np.array([-200, 20, 400]), 120)
                # show results
                self.show_results_stream(
                    depth, cube, index, confidence)
                # self.show_results(depth, cube)
                # sys.stdout.write('.')
                # sys.stdout.flush()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    def capture_loop(self):
        # Initialize service
        with pyrs.Service() as serv:
            depth_stream = pyrs.stream.DepthStream()
            # Initialize device
            with serv.Device(streams=(depth_stream,)) as dev:
                # Wait for device to initialize
                time.sleep(1.)
                self.capture_detect(serv, dev)


if __name__ == '__main__':
    with args_holder() as argsholder:
        argsholder.parse_args()
        ARGS = argsholder.args
        ARGS.mode = 'detect'
        argsholder.create_instance()
        cap = capture(ARGS)
        cap.capture_loop()
