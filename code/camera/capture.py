""" Hand in Depth
    https://github.com/xkunwu/depth-hand
"""
import os
import sys
import numpy as np
# from copy import deepcopy
import matplotlib.pyplot as mpplot
from matplotlib.animation import FuncAnimation
from skimage import io as skimio
import warnings
import tensorflow as tf
from colour import Color
import time
# from multiprocessing import Queue, Pool
from args_holder import args_holder
from utils.iso_boxes import iso_cube
from camera.hand_locator import hand_locator
mpplot.switch_backend("TkAgg")  # sometimes does not work
# mpplot.switch_backend("Qt5Agg")  # VERY, VERY slow


# helper to define the rendering canvas
class DetCanvas:
    def __init__(self, fig, ims, axes):
        self.fig = fig
        self.ims = ims
        self.axes = axes

    def save_raw(self, filename, axi=0):
        if 0 > axi:
            raise ValueError('can only save subplot')
            return
        if len(self.axes) <= axi:
            raise ValueError('there are only {} subplots'.format(len(self.axes)))
            return
        img = self.ims[axi].get_array()  # float64
        img = img.astype(np.uint16)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skimio.imsave(filename, img)
        return img

    def save_det(self, filename, axi=0):
        if 0 > axi:
            self.fig.savefig(filename)
            return
        if len(self.axes) <= axi:
            raise ValueError('there are only {} subplots'.format(len(self.axes)))
            return
        extent = self.axes[axi].get_window_extent().transformed(
            self.fig.dpi_scale_trans.inverted())
        self.fig.savefig(filename, bbox_inches=extent)

    @staticmethod
    def create_canvas(caminfo):
        # Create the figure canvas
        fig, _ = mpplot.subplots(nrows=1, ncols=2, figsize=(2 * 6, 1 * 6))
        ax1 = mpplot.subplot(1, 2, 1)
        ax1.set_axis_off()
        ax1.set_xlim(0, 640)
        ax1.set_ylim(480, 0)
        ax2 = mpplot.subplot(1, 2, 2)
        ax2.set_axis_off()
        # ax3 = mpplot.subplot(1, 3, 3)
        # ax3.set_axis_off()
        # mpplot.subplots_adjust(left=0, right=1, top=1, bottom=0)
        im1 = ax1.imshow(
            np.zeros(caminfo.image_size, dtype=np.uint16),
            vmin=0, vmax=2000, cmap=mpplot.cm.bone_r)
        # im1 = ax1.imshow(
        #     np.zeros(caminfo.image_size, dtype=np.float),
        #     vmin=0., vmax=1., cmap=mpplot.cm.bone_r)
        im2 = ax2.imshow(np.zeros([480, 640, 3], dtype=np.uint8))
        # im3 = ax3.imshow(
        #     np.zeros((128, 128), dtype=np.float),
        #     vmin=0., vmax=1., cmap=mpplot.cm.bone_r)
        mpplot.tight_layout()
        canvas = DetCanvas(
            fig=fig, ims=(im1, im2), axes=(ax1, ax2))
        return canvas

    @staticmethod
    def create_debug_canvas(caminfo):
        # Create the figure canvas
        fig, _ = mpplot.subplots(nrows=1, ncols=3, figsize=(3 * 4, 1 * 4))
        ax1 = mpplot.subplot(1, 3, 1)
        ax2 = mpplot.subplot(1, 3, 2)
        ax3 = mpplot.subplot(1, 3, 3)
        im1 = ax1.imshow(
            np.zeros((128, 128), dtype=np.float),
            vmin=0., vmax=1., cmap=mpplot.cm.bone_r)
        im2 = ax2.imshow(
            np.zeros((128, 128), dtype=np.float),
            vmin=0., vmax=1., cmap=mpplot.cm.bone_r)
        im3 = ax3.imshow(
            np.zeros((128, 128), dtype=np.float),
            vmin=0., vmax=1., cmap=mpplot.cm.bone_r)
        mpplot.tight_layout()
        canvas = DetCanvas(
            fig=fig, ims=(im1, im2, im3), axes=(ax1, ax2, ax3))
        return canvas


class capture:
    def show_debug_fig(self, img, cube):
        points3_pick = cube.pick(
            self.args.data_ops.img_to_raw(img, self.caminfo))
        points3_norm = cube.transform_center_shrink(points3_pick)
        # print(points3_pick.shape, points3_norm.shape)
        coord, depth = cube.project_ortho(points3_norm, roll=0)
        img_crop = cube.print_image(coord, depth, self.caminfo.crop_size)
        self.debug_fig.ims[0].set_data(img_crop)
        coord, depth = cube.project_ortho(points3_norm, roll=1)
        img_crop = cube.print_image(coord, depth, self.caminfo.crop_size)
        self.debug_fig.ims[1].set_data(img_crop)
        coord, depth = cube.project_ortho(points3_norm, roll=2)
        img_crop = cube.print_image(coord, depth, self.caminfo.crop_size)
        self.debug_fig.ims[2].set_data(img_crop)

    def __init__(self, args, camera):
        self.args = args
        self.cam = camera
        self.caminfo = camera.caminfo
        self.debug_fig = args.show_debug
        self.save_dir = os.path.join(self.args.out_dir, 'capture')
        self.save_det = args.save_det
        if self.save_det:
            self.save_det = os.path.join(
                self.save_dir,
                "detection_{}".format(time.time()))
            if not os.path.exists(self.save_det):
                os.makedirs(self.save_det)
            print('save detection at: ', self.save_det)
        self.save_raw = args.save_stream
        if self.save_raw:
            self.save_raw = args.stream_dir
            print('save detection at: ', self.save_raw)

        # create the rendering canvas
        def close(event):
            if event.key == 'q':
                mpplot.close(event.canvas.figure)
            if event.key == 'b':
                mpplot.savefig(os.path.join(
                    self.args.out_dir,
                    'capture_{}.png'.format(time.time())))

        self.canvas = DetCanvas.create_canvas(self.caminfo)
        self.canvas.fig.canvas.mpl_connect(
            "key_press_event", close)
        if self.debug_fig:
            self.debug_fig = DetCanvas.create_debug_canvas(self.caminfo)

    def show_results(
        self, canvas,
            cube=iso_cube(np.array([-200, 20, 400]), 120),
            pose_det=None):
        ax = canvas.axes[0]
        rects = cube.proj_rects_3(
            self.args.data_ops.raw_to_2d,
            self.caminfo
        )
        colors = [Color('orange').rgb, Color('red').rgb, Color('lime').rgb]
        for ii, rect in enumerate(rects):
            rect.draw(ax, colors[ii])
        if pose_det is None:
            return
        self.args.data_draw.draw_pose2d(
            ax, self.caminfo,
            self.args.data_ops.raw_to_2d(pose_det, self.caminfo)
        )

    def detect_region(self, depth, cube, sess, ops):
        depth_prow = self.args.model_inst.prow_one(
            depth, cube, self.args, self.caminfo)
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
            pred_val, cube, self.caminfo)
        return pose_det

    def show_detection(self, sess, ops):
        hfinder = hand_locator(self.args, self.caminfo)

        def update(i):
            print("==== Frame: ", i, "====")
            canvas = self.canvas
            ax = canvas.axes[0]
            [p.remove() for p in reversed(ax.patches)]  # remove previews Rectangle drawings
            for artist in ax.lines + ax.collections:
                artist.remove()  # remove all lines
            camframes = self.cam.provide()
            if camframes is None:
                return
            depth_image = camframes.depth
            color_image = camframes.color
            # canvas.ims[0].set_data(
            #     depth_image / self.caminfo.z_range[1])
            canvas.ims[0].set_data(depth_image)
            canvas.ims[1].set_data(color_image)
            cube = hfinder.simp_crop(depth_image)
            if cube is False:
                return
            # cube = camframes.extra  # FetchHands17
            pose_det = self.detect_region(
                depth_image, cube, sess, ops)
            self.show_results(canvas, cube, pose_det)
            if self.debug_fig:
                self.show_debug_fig(depth_image, cube)
            if self.save_det is not False:
                filename = os.path.join(
                    self.save_det,
                    self.args.data_io.index2imagename(i))
                self.canvas.save_det(filename)
            if self.save_raw is not False:
                filename = os.path.join(
                    self.save_raw,
                    self.args.data_io.index2imagename(i))
                self.canvas.save_raw(filename)
                # img = self.canvas.save_raw(filename)
                # print(np.max(abs(img - depth_image)), np.max(img), np.max(depth_image))

        # assign return value is necessary! Otherwise no updates.
        anim = FuncAnimation(
            self.canvas.fig, update, blit=False, interval=1)
        if self.debug_fig:
            anim_debug = FuncAnimation(
                self.debug_fig.fig, update, blit=False, interval=1)
        mpplot.show()

    def capture_detect(self):
        tf.reset_default_graph()
        with tf.Graph().as_default(), \
                tf.device('/gpu:' + str(self.args.gpu_id)):
            placeholders = \
                self.args.model_inst.placeholder_inputs(1)
            frames_op = placeholders.frames_tf
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
                print('restoring model from: {} ...'.format(
                    model_path))
                saver.restore(sess, model_path)
                print('model restored.')
                ops = {
                    'batch_frame': frames_op,
                    'is_training': is_training_tf,
                    'pred': pred_op
                }
                self.show_detection(sess, ops)

    def capture_test(self):
        def update(i):
            print("==== Frame: ", i, "====")
            canvas = self.canvas
            camframes = self.cam.provide()
            if camframes is None:
                return
            depth_image = camframes.depth
            color_image = camframes.color
            canvas.ims[0].set_data(depth_image)
            canvas.ims[1].set_data(color_image)
            cube = iso_cube(np.array([0, 0, 400]), 120)
            # cube=iso_cube(np.array([-200, 20, 400]), 120)
            self.show_results(canvas, cube)

        # assign return value is necessary! Otherwise no updates.
        anim = FuncAnimation(
            self.canvas.fig, update, blit=False, interval=1)
        mpplot.show()
        if self.save_raw is not False:
            filename = os.path.join(
                self.save_raw,
                "animcap_{}.mp4".format(time.time()))
            anim.save(
                filename, fps=30,
                extra_args=['-vcodec', 'libx264'])

    def capture_loop(self):
        # self.capture_test()
        self.capture_detect()


def test_camera(cap):
    # test the camera projection: center should align with the image dimension
    cube = iso_cube(np.array([0, 0, 400]), 120)
    rects = cube.proj_rects_3(
        cap.args.data_ops.raw_to_2d,
        cap.caminfo
    )
    np.set_printoptions(formatter={'float': '{:6.4f}'.format})
    for ii, rect in enumerate(rects):
        rect.show_dims()


def test_smooth(args):
    from mpl_toolkits.mplot3d import Axes3D
    from camera.realsense_cam import FetchHands17
    with FetchHands17(args) as cam:
        depthimg = cam.depth_image
        smoothed = cam.smooth_data()
        cube = cam.cube
        caminfo = cam.caminfo
        mpplot.subplots(nrows=1, ncols=2, figsize=(2 * 5, 2 * 5))
        ax = mpplot.subplot(1, 2, 1, projection='3d')
        points3 = args.data_ops.img_to_raw(depthimg, caminfo)
        points3_trans = points3
        # points3_trans = cube.pick(points3)
        # points3_trans = cube.transform_to_center(points3_trans)
        numpts = points3_trans.shape[0]
        if 10000 < numpts:
            points3_trans = points3_trans[
                np.random.choice(numpts, 1000, replace=False), :]
        ax.scatter(
            points3_trans[:, 0], points3_trans[:, 1], points3_trans[:, 2],
            color=Color('lightsteelblue').rgb)
        corners = cube.transform_to_center(cube.get_corners())
        cube.draw_cube_wire(ax, corners)
        ax.view_init(azim=-120, elev=-150)
        ax = mpplot.subplot(1, 2, 2, projection='3d')
        points3 = args.data_ops.img_to_raw(smoothed, caminfo)
        points3_trans = points3
        # points3_trans = cube.pick(points3)
        # points3_trans = cube.transform_to_center(points3_trans)
        numpts = points3_trans.shape[0]
        if 10000 < numpts:
            points3_trans = points3_trans[
                np.random.choice(numpts, 1000, replace=False), :]
        ax.scatter(
            points3_trans[:, 0], points3_trans[:, 1], points3_trans[:, 2],
            color=Color('lightsteelblue').rgb)
        corners = cube.transform_to_center(cube.get_corners())
        cube.draw_cube_wire(ax, corners)
        ax.view_init(azim=-120, elev=-150)
        mpplot.show()


if __name__ == '__main__':
    # import pdb; pdb.set_trace()
    with args_holder() as argsholder:
        if not argsholder.parse_args():
            os._exit(0)
        ARGS = argsholder.args
        ARGS.mode = 'detect'
        # ARGS.model_name = 'super_edt2m'
        if not argsholder.create_instance():
            os._exit(0)
        # test_smooth(ARGS)
        if ARGS.read_stream:
            from camera.realsense_cam import FileStreamer
            with FileStreamer(ARGS) as cam:
                ## FetchHands17!! {
                # cam.caminfo = ARGS.data_inst
                ## }
                cap = capture(ARGS, cam)
                test_camera(cap)
                cap.capture_loop()
        else:
            ## FetchHands17!! {
            # from camera.realsense_cam import FetchHands17
            # with FetchHands17(ARGS) as cam:
            ## }
            from camera.realsense_cam import RealsenceCam
            with RealsenceCam(ARGS) as cam:
                cap = capture(ARGS, cam)
                test_camera(cap)
                cap.capture_loop()
