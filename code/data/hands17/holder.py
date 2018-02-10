import os
# import sys
import shutil
import numpy as np
from colour import Color
import progressbar
from . import ops as dataops
from . import io as dataio


class hands17holder:
    """ Pose class for Hands17 dataset
        wired coordinate system:
            ---> x
            |
            |y
        which is the same as image space.
        BUT np use row-major, so remember exchange (x, y) for each mapping
    """

    # # dataset info
    # data_dir = ''
    # out_dir = ''
    # predict_dir = ''
    # training_images = ''
    # frame_images = ''
    # training_annot_origin = ''
    # training_annot_cleaned = ''
    # training_annot_train = ''
    # training_annot_test = ''
    # frame_bbox = ''
    #
    # # num_training = int(957032)
    # num_training = int(992)
    # # num_training = int(96)

    # cropped & resized training images
    # world/image coordinates are reversed!!!
    # from colour import Color
    # p3 = np.array([
    #     [-20, -20, 400],
    #     [-20, 20, 400],
    #     [20, 20, 400],
    # ])
    # p2 = args.data_ops.raw_to_2d(p3, self.caminfo)
    # colors = [Color('orange').rgb, Color('red').rgb, Color('lime').rgb]
    # for ii, p in enumerate(p2):
    #     mpplot.plot(
    #         p[0], p[1],
    #         'o',
    #         color=colors[ii]
    #     )
    image_size = (480, 640)
    region_size = 120  # empirical spacial cropping radius
    crop_size = 128  # input image size to models (may changed)
    anchor_num = 16  # for attention model
    crop_range = 480.  # +/- spacial capture range
    z_range = (100., 1060.)  # empirical valid depth range
    z_max = 9999.  # max distance set to 10m
    # camera info
    focal = (475.065948, 475.065857)
    centre = (315.944855, 245.287079)
    # centre = (245.287079, 315.944855)
    # fx = 475.065948
    # fy = 475.065857
    # cx = 315.944855
    # cy = 245.287079

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

    def remove_out_frame_annot(self):
        self.num_training = int(0)
        pose_limit = np.array([
            [np.inf, np.inf, np.inf],
            [-np.inf, -np.inf, -np.inf],
        ])
        with open(self.training_annot_cleaned, 'w') as writer, \
                open(self.training_annot_origin, 'r') as reader:
            lines = reader.readlines()
            timerbar = progressbar.ProgressBar(
                maxval=len(lines),
                widgets=[
                    progressbar.Percentage(),
                    ' ', progressbar.Bar('=', '[', ']'),
                    ' ', progressbar.ETA()]
            ).start()
            for li, annot_line in enumerate(lines):
                _, pose_raw = dataio.parse_line_annot(annot_line)
                pose2d = dataops.raw_to_2d(pose_raw, self)
                if 0 > np.min(pose2d):
                    continue
                if 0 > np.min(self.image_size - pose2d):
                    continue
                writer.write(annot_line)
                pose_limit[0, :] = np.minimum(
                    pose_limit[0, :],
                    np.min(pose_raw, axis=0))
                pose_limit[1, :] = np.maximum(
                    pose_limit[1, :],
                    np.max(pose_raw, axis=0))
                self.num_training += 1
                if 0 == (li % 10e3):
                    timerbar.update(li)
            timerbar.finish()
        return pose_limit

    def shuffle_split(self):
        with open(self.training_annot_cleaned, 'r') as source:
            lines = source.readlines()
        # import random
        # np.random.shuffle(lines)
        with open(self.training_annot_train, 'w') as f:
            for line in lines[:self.train_test_split]:
                f.write(line)
        with open(self.training_annot_test, 'w') as f:
            for line in lines[self.train_test_split:]:
                # name = re.match(r'^(image_D\d+\.png)', line).group(1)
                # shutil.move(
                #     os.path.join(self.training_cropped, name),
                #     os.path.join(self.evaluate_cropped, name))
                f.write(line)

    def next_valid_split(self):
        """ split range for validation set """
        # split_beg = self.portion * self.split_id
        # self.split_id = (self.split_id + 1) % self.split_num
        # split_end = self.portion * self.split_id
        split_beg = self.portion * self.split_id
        split_end = 0
        return split_beg, split_end

    def init_data(self):
        if (not os.path.exists(self.training_annot_cleaned)):
            from timeit import default_timer as timer
            from datetime import timedelta
            time_s = timer()
            self.logger.info('cleaning data ...')
            pose_limit = self.remove_out_frame_annot()
            time_e = str(timedelta(seconds=timer() - time_s))
            self.logger.info('{:d} images after cleaning, time: {}'.format(
                self.num_training, time_e))
            self.logger.info('pose limit: {} --> {}'.format(
                pose_limit[0, :], pose_limit[1, :])
            )
        else:
            self.num_training = int(sum(
                1 for line in open(self.training_annot_cleaned, 'r')))

        # split the data into 10 portions
        split_num = int(10)
        portion = int(np.ceil(float(self.num_training) / split_num))
        # the last portion is used for test (compare models)
        self.train_test_split = int(portion * (split_num - 1))
        self.split_num = split_num - 1
        self.portion = portion
        # 1 out of (10 - 1) portions is picked out for validation
        self.split_id = -1 % self.split_num

        if ((not os.path.exists(self.training_annot_train)) or
                (not os.path.exists(self.training_annot_test))):
            self.shuffle_split()
            self.logger.info('splitted data: {} training, {} test ({:d} portions).'.format(
                self.train_test_split,
                self.num_training - self.train_test_split,
                split_num))
        test_file = os.path.basename(self.training_annot_test)
        if not os.path.exists(os.path.join(self.predict_dir, test_file)):
            shutil.copy2(self.training_annot_test, self.predict_dir)

    def __init__(self, args):
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.logger = args.logger
        self.predict_dir = args.predict_dir
        self.crop_size = args.crop_size
        self.anchor_num = args.anchor_num
        self.crop_range = args.crop_range
        # self.z_range[1] = self.crop_range * 2. + self.z_range[0]
        self.training_images = os.path.join(self.data_dir, 'training/images')
        self.frame_images = os.path.join(self.data_dir, 'frame/images')
        self.training_annot_origin = os.path.join(
            self.data_dir, 'training/Training_Annotation.txt')
        self.training_annot_cleaned = os.path.join(
            self.out_dir, 'annotation.txt')
        self.training_annot_train = os.path.join(
            self.out_dir, 'training_train.txt')
        self.training_annot_test = os.path.join(
            self.out_dir, 'training_test.txt')
        self.frame_bbox = os.path.join(self.data_dir, 'frame/BoundingBox.txt')
