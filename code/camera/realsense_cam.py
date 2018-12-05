import os
import sys
import numpy as np
import re
from collections import namedtuple
import pyrealsense2 as pyrs


CamFrames = namedtuple("CamFrames", "depth, color, extra")


class DummyCamFrame:
    def __init__(self, caminfo):
        self.caminfo = caminfo
        # for test purpose
        self.test_depth = np.zeros(caminfo.image_size, dtype=np.uint16)
        self.test_depth[10:20, 10:20] = 240

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is not None:
            import traceback
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            sys.exit()
        return self

    def provide(self):
        return CamFrames(self.test_depth, self.test_depth, self.test_depth)


class FetchHands17:
    def __init__(self, caminfo, args=None):
        if args is None:
            raise ValueError('need to provide valid args')
            return 0
        args.model_inst.check_dir(args.data_inst, args)
        self.caminfo = caminfo
        self.depth_image, self.cube = \
            args.model_inst.fetch_random(args)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is not None:
            import traceback
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            sys.exit()
        return self

    def provide(self):
        return CamFrames(self.depth_image, self.depth_image, self.cube)


class FileStreamer:
    def __init__(self, caminfo, args=None, outdir=None):
        if (args is None) or (outdir is None):
            raise ValueError('need to provide valid output path')
            return 0
        self.args = args
        filelist = [f for f in os.listdir(outdir) if re.match(r'image_D(\d+)\.png', f)]
        # filelist = os.listdir(outdir)
        # print(outdir)
        # print(filelist)
        # retem = re.compile(r'image_D(\d+)\.png')
        # # filelist = filter(
        # #     retem.match,
        # #     filelist)
        # filelist = [f for f in filelist if re.match(r'image_D(\d+)\.png', f)]
        print(filelist)
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
            sys.exit()
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
    def __init__(self, caminfo):
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
        # z_range = self.caminfo.z_range  # in metric system

        # Create an align object
        align_to = pyrs.stream.color
        align = pyrs.align(align_to)

        self.pipeline = pipeline
        self.align = align
        self.depth_scale = depth_scale
        self.caminfo = caminfo

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.pipeline.stop()
        if exc_type is not None:
            import traceback
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            sys.exit()
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

        return CamFrames(depth_image, color_image, bg_removed)
