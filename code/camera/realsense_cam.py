""" Hand in Depth
    https://github.com/xkunwu/depth-hand
"""
import os
import sys
import numpy as np
import re
from collections import namedtuple
from colour import Color
import pyrealsense2 as pyrs


class caminfo_ir:
    image_size = (480, 640)
    region_size = 120
    crop_size = 128  # input image size to models (may changed)
    crop_range = 480  # only operate within this range
    z_range = (100., 1060.)
    anchor_num = 8
    # intrinsic paramters of Intel Realsense SR300
    focal = (475.857, 475.856)
    centre = (310.982, 246.123)
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


CamFrames = namedtuple("CamFrames", "depth, color, extra")


class DummyCamFrame:
    def __init__(self):
        self.caminfo = caminfo_ir
        # for test purpose
        self.test_depth = np.zeros(self.caminfo.image_size, dtype=np.uint16)
        self.test_depth[10:20, 10:20] = 240

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is not None:
            import traceback
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            os._exit(0)
        return self

    def provide(self):
        return CamFrames(self.test_depth, self.test_depth, self.test_depth)


class FetchHands17:
    def __init__(self, args):
        if args is None:
            raise ValueError('need to provide valid args')
        self.caminfo = args.data_inst
        args.model_inst.check_dir(args.data_inst, args)
        self.depth_image, self.cube = \
            args.model_inst.fetch_random(args)

    def smooth_data(self, scale=5):
        import cv2
        return cv2.bilateralFilter(
            self.depth_image.astype(np.float32),
            5, 30, 30)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is not None:
            import traceback
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            os._exit(0)
        return self

    def provide(self):
        return CamFrames(self.depth_image, self.depth_image, self.cube)


class FileStreamer:
    def __init__(self, args):
        if args is None:
            raise ValueError('need to provide valid args')
        self.caminfo = caminfo_ir
        self.args = args
        outdir = args.stream_dir
        print('reading path: ', outdir)
        filelist = [f for f in os.listdir(outdir) if re.match(r'image_D(\d+)\.png', f)]
        if 0 == len(filelist):
            raise ValueError('no stream data found!')
        filelist.sort(key=lambda f: int(args.data_io.imagename2index(f)))
        self.filelist = [
            os.path.join(outdir, f) for f in filelist]
        self.clid = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is not None:
            import traceback
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            os._exit(0)
        return self

    def provide(self):
        if self.clid >= len(self.filelist):
            return None
        # print('reading: {}'.format(self.filelist[self.clid]))
        depth_image = self.args.data_io.read_image(
            self.filelist[self.clid]
        )
        self.clid += 1
        return CamFrames(depth_image, depth_image, None)


class RealsenceCam:
    def __init__(self, args):
        self.caminfo = caminfo_ir

        # Create a pipeline
        pipeline = pyrs.pipeline()

        # Create a config and configure the stream
        config = pyrs.config()
        image_size = caminfo_ir.image_size
        config.enable_stream(
            pyrs.stream.depth,
            image_size[1], image_size[0],
            pyrs.format.z16, 30)
        config.enable_stream(
            pyrs.stream.color,
            image_size[1], image_size[0],
            pyrs.format.rgb8, 30)

        # Start streaming
        profile = pipeline.start(config)
        depth_stream = profile.get_stream(pyrs.stream.depth)

        # Read intrinsics
        depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
        self.caminfo.focal = (
            depth_intrinsics.fx, depth_intrinsics.fy)
        self.caminfo.centre = (
            depth_intrinsics.ppx, depth_intrinsics.ppy)
        print("Depth intrinsics: ", depth_intrinsics)
        print("Intrinsics copied: ", self.caminfo.focal, self.caminfo.centre)

        # Getting the depth sensor's depth scale
        depth_sensor = profile.get_device().first_depth_sensor()
        # depth_sensor.set_option(pyrs.option.depth_units, 0.001); # RuntimeError: This option is read-only! - Not supported for SR300.
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: ", depth_scale)  # 0.00012498664727900177

        # clip the background
        # clipping_distance_in_meters = 1 #1 meter
        # clipping_distance = clipping_distance_in_meters / depth_scale
        # z_range = self.caminfo.z_range  # in metric system

        # Create an align object
        align_to = pyrs.stream.color
        align = pyrs.align(align_to)

        self.pipeline = pipeline
        self.align = align
        self.depth_scale = depth_scale * 1000  # scale to metric
        self.depth_scale *= 0.8  # scale tweak

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.pipeline.stop()
        if exc_type is not None:
            import traceback
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            os._exit(0)
        return self

    def provide(self):
        pipeline = self.pipeline
        align = self.align
        depth_scale = self.depth_scale
        z_range = self.caminfo.z_range

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
        depth_image = depth_image * depth_scale
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

        return CamFrames(depth_image, color_image, bg_removed)
