import os
import sys
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as mpplot
from matplotlib.animation import FuncAnimation
import tensorflow as tf
# from tensorflow.contrib import slim
from colour import Color
import cv2
import pyrealsense2 as pyrs
import time
# from multiprocessing import Queue, Pool
from args_holder import args_holder
from utils.iso_boxes import iso_cube
from camera.hand_finder import hand_finder


class capture:
    class caminfo_ir:
        image_size = (480, 640)
        region_size = 120
        crop_size = 128  # input image size to models (may changed)
        crop_range = 480  # only operate within this range
        z_range = (100., 1060.)
        anchor_num = 8
        # intrinsic paramters of Intel Realsense SR300
        focal = (463.889, 463.889)
        centre = (320, 240)
        # joints description
        join_name = [
            'Wrist',
            'TMCP', 'IMCP', 'MMCP', 'RMCP', 'PMCP',
            'TPIP', 'TDIP', 'TTIP',
            'IPIP', 'IDIP', 'ITIP',
            'MPIP', 'MDIP', 'MTIP',
            'RPIP', 'RDIP', 'RTIP',
            'PPIP', 'PDIP', 'PTIP'
        ]
        join_num = 21
        join_type = ('W', 'T', 'I', 'M', 'R', 'P')
        join_color = (
            # Color('cyan'),
            Color('black'),
            Color('magenta'),
            Color('blue'),
            Color('lime'),
            Color('yellow'),
            Color('red')
        )
        join_id = (
            (1, 6, 7, 8),
            (2, 9, 10, 11),
            (3, 12, 13, 14),
            (4, 15, 16, 17),
            (5, 18, 19, 20)
        )
        bone_id = (
            ((0, 1), (1, 6), (6, 11), (11, 16)),
            ((0, 2), (2, 7), (7, 12), (12, 17)),
            ((0, 3), (3, 8), (8, 13), (13, 18)),
            ((0, 4), (4, 9), (9, 14), (14, 19)),
            ((0, 5), (5, 10), (10, 15), (15, 20))
        )
        bbox_color = Color('orange')

        def __init__():
            pass

    # helper to define the rendering canvas
    Canvas = namedtuple("Canvas", "fig ims axes")

    def create_canvas(self):
        # Create the figure canvas
        fig, _ = mpplot.subplots(nrows=1, ncols=2, figsize=(2 * 6, 1 * 6))
        ax1 = mpplot.subplot(1, 2, 1)
        # ax1.set_axis_off()
        ax1.set_xlim(0, 640)
        ax1.set_ylim(480, 0)
        ax2 = mpplot.subplot(1, 2, 2)
        # ax2.set_axis_off()
        # mpplot.subplots_adjust(left=0, right=1, top=1, bottom=0)
        mpplot.tight_layout()
        # need to define vmax, otherwise cannot update
        im1 = ax1.imshow(
            np.zeros(self.caminfo_ir.image_size, dtype=np.uint16),
            vmin=0., vmax=1., cmap=mpplot.cm.bone_r)
        im2 = ax2.imshow(np.zeros([480, 640, 3], dtype=np.uint8))
        # ax1.invert_xaxis()  # mirror the horizontal direction
        # ax2.invert_xaxis()
        # axes = fig.get_axes()
        canvas = self.Canvas(
            fig=fig, ims=(im1, im2), axes=fig.get_axes())
        return canvas

    def __init__(self, args):
        self.args = args

        # create the rendering canvas
        def close(event):
            if event.key == 'q':
                mpplot.close(event.canvas.figure)

        self.canvas = self.create_canvas()
        self.canvas.fig.canvas.mpl_connect(
            "key_press_event", close)
        self.pplot = []

        # for test purpose
        self.test_depth = np.zeros(self.caminfo_ir.image_size, dtype=np.uint16)
        self.test_depth[10:20, 10:20] = 240

    @staticmethod
    def read_frame_from_device(
            pipeline, align, depth_scale, z_range=caminfo_ir.z_range):
        frames = pipeline.wait_for_frames()

        # Get frameset of only depth
        # depth = frames.get_depth_frame()
        # if not depth:
        #     continue
        # depth_image = np.asanyarray(depth)

        # Get aligned color and depth frameset
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not aligned_depth_frame or not color_frame:
            return
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        depth_image = depth_image * depth_scale / 0.001  # scale to metric
        color_image = np.asanyarray(color_frame.get_data())
        # color_image = np.asanyarray(color_frame.get_data())[..., ::-1]
        # mirror the horizontal direction
        depth_image = np.flip(depth_image, 1)
        color_image = np.flip(color_image, 1)

        # Remove background - Set to grey
        grey_color = 159
        depth_image_3d = np.dstack(
            (depth_image, depth_image, depth_image))
        bg_removed = np.where(
            (depth_image_3d > z_range[1]) | (depth_image_3d <= 0),
            grey_color, color_image)
        np.clip(
            depth_image,
            z_range[0], z_range[1],
            out=depth_image)

        return depth_image, color_image, bg_removed

    def show_results(
        self, canvas,
            cube=iso_cube(np.array([-200, 20, 400]), 120),
            pose_det=None):
        ax = canvas.axes[0]
        rects = cube.proj_rects_3(
            self.args.data_ops.raw_to_2d,
            self.caminfo_ir
        )
        colors = [Color('orange').rgb, Color('red').rgb, Color('lime').rgb]
        for ii, rect in enumerate(rects):
            rect.draw(ax, colors[ii])
        if pose_det is None:
            return
        self.pplot = self.args.data_draw.draw_pose2d(
            ax, self.caminfo_ir,
            self.args.data_ops.raw_to_2d(pose_det, self.caminfo_ir)
        )

    def detect_region(self, depth, cube, sess, ops):
        depth_prow = self.args.model_inst.prow_one(
            depth, cube, self.args, self.caminfo_ir)
        depth_prow = np.expand_dims(depth_prow, -1)
        depth_prow = np.expand_dims(depth_prow, 0)
        feed_dict = {
            ops['batch_frame']: depth_prow,
            ops['is_training']: False
        }
        pred_val = sess.run(
            ops['pred'],
            feed_dict=feed_dict)
        pose_det = self.args.model_inst.rece_one(
            pred_val.reshape(3, -1).T, cube, self.caminfo_ir)
        return pose_det

    def show_detection(
            self, pipeline, align, depth_scale, sess, ops):
        hfinder = hand_finder(self.args, self.caminfo_ir)

        def update(i):
            canvas = self.canvas
            ax = canvas.axes[0]
            [p.remove() for p in reversed(ax.patches)]  # remove previews Rectangle drawings
            for i, line in enumerate(ax.lines):
                ax.lines.pop(i)  # remove all lines
            depth_image, color_image, bg_removed = self.read_frame_from_device(
                pipeline, align, depth_scale)
            # depth_image = self.test_depth  # TEST!!
            canvas.ims[0].set_data(
                depth_image / self.caminfo_ir.z_range[1])
            canvas.ims[1].set_data(bg_removed)
            cube = hfinder.simp_crop(depth_image)
            if cube is False:
                return
            pose_det = self.detect_region(
                depth_image, cube, sess, ops)
            self.show_results(canvas, cube, pose_det)

        # assign return value is necessary! Otherwise no updates.
        anim = FuncAnimation(
            self.canvas.fig, update, blit=False, interval=1)
        mpplot.show()

    def capture_detect(
            self, pipeline, align, depth_scale):
        tf.reset_default_graph()
        with tf.Graph().as_default(), \
                tf.device('/gpu:' + str(self.args.gpu_id)):
            frames_op, _ = \
                self.args.model_inst.placeholder_inputs(1)
            is_training_tf = tf.placeholder(
                tf.bool, shape=(), name='is_training')
            pred_op, end_points = self.args.model_inst.get_model(
                frames_op, is_training_tf,
                self.args.bn_decay, self.args.regu_scale)
            saver = tf.train.Saver()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            with tf.Session(config=config) as sess:
                model_path = self.args.model_inst.ckpt_path
                saver.restore(sess, model_path)
                ops = {
                    'batch_frame': frames_op,
                    'is_training': is_training_tf,
                    'pred': pred_op
                }
                self.show_detection(
                    pipeline, align, depth_scale, sess, ops)

    def capture_test(
            self, pipeline, align, depth_scale):
        def update(i):
            canvas = self.canvas
            depth_image, color_image, bg_removed = self.read_frame_from_device(
                pipeline, align, depth_scale)
            canvas.ims[0].set_data(
                depth_image / self.caminfo_ir.z_range[1])
            canvas.ims[1].set_data(bg_removed)
            cube = iso_cube(np.array([0, 0, 400]), 120)
            # cube=iso_cube(np.array([-200, 20, 400]), 120)
            self.show_results(canvas, cube)

        # assign return value is necessary! Otherwise no updates.
        anim = FuncAnimation(
            self.canvas.fig, update, blit=False, interval=1)
        mpplot.show()
        anim.save(
            os.path.join(self.args.out_dir, "capture.mp4"),
            fps=30, extra_args=['-vcodec', 'libx264'])

    def capture_loop(self):
        # Create a pipeline
        pipeline = pyrs.pipeline()

        # Create a config and configure the stream
        config = pyrs.config()
        config.enable_stream(pyrs.stream.depth, 640, 480, pyrs.format.z16, 30)
        config.enable_stream(pyrs.stream.color, 640, 480, pyrs.format.rgb8, 30)
        # config.enable_stream(pyrs.stream.color, 640, 480, pyrs.format.bgr8, 30)

        # Start streaming
        profile = pipeline.start(config)

        # Getting the depth sensor's depth scale
        depth_sensor = profile.get_device().first_depth_sensor()
        # depth_sensor.set_option(pyrs.option.depth_units, 0.001); # RuntimeError: This option is read-only! - Not supported for SR300.
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: ", depth_scale)

        # clip the background
        # clipping_distance_in_meters = 1 #1 meter
        # clipping_distance = clipping_distance_in_meters / depth_scale
        # z_range = self.caminfo_ir.z_range  # in metric system

        # Create an align object
        align_to = pyrs.stream.color
        align = pyrs.align(align_to)

        try:
            # self.capture_test(pipeline, align, depth_scale)
            self.capture_detect(pipeline, align, depth_scale)
        finally:
            pipeline.stop()


def test_camera(cap):
    # test the camera projection: center should align with the image dimension
    cube = iso_cube(np.array([0, 0, 400]), 120)
    rects = cube.proj_rects_3(
        cap.args.data_ops.raw_to_2d,
        cap.caminfo_ir
    )
    np.set_printoptions(formatter={'float': '{:6.4f}'.format})
    for ii, rect in enumerate(rects):
        rect.show_dims()


if __name__ == '__main__':
    with args_holder() as argsholder:
        argsholder.parse_args()
        ARGS = argsholder.args
        ARGS.mode = 'detect'
        argsholder.create_instance()
        cap = capture(ARGS)
        test_camera(cap)
        cap.capture_loop()
