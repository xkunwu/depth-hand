import os
# import sys
import shutil
import numpy as np
from colour import Color
import progressbar
from . import ops as dataops
from . import io as dataio


class hands17holder:
    """ Pose class for Hands17 dataset """

    # dataset info
    data_dir = ''
    out_dir = ''
    predict_dir = ''
    training_images = ''
    frame_images = ''
    training_annot_origin = ''
    training_annot_cleaned = ''
    training_annot_train = ''
    training_annot_test = ''
    frame_bbox = ''

    # num_training = int(957032)
    num_training = int(992)
    # num_training = int(96)
    tt_split = int(64)
    range_train = np.zeros(2, dtype=np.int)
    range_test = np.zeros(2, dtype=np.int)

    # cropped & resized training images
    image_size = [480, 640]
    region_size = 120
    crop_size = 128
    crop_range = 800.  # +/- limit
    z_range = (1e-4, 1600.)
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
                self.num_training += 1
                if 0 == (li % 10e3):
                    timerbar.update(li)
            timerbar.finish()

    def shuffle_split(self):
        with open(self.training_annot_cleaned, 'r') as source:
            lines = source.readlines()
        # import random
        # np.random.shuffle(lines)
        with open(self.training_annot_train, 'w') as f:
            for line in lines[self.range_train[0]:self.range_train[1]]:
                f.write(line)
        with open(self.training_annot_test, 'w') as f:
            for line in lines[self.range_test[0]:self.range_test[1]]:
                # name = re.match(r'^(image_D\d+\.png)', line).group(1)
                # shutil.move(
                #     os.path.join(self.training_cropped, name),
                #     os.path.join(self.evaluate_cropped, name))
                f.write(line)
        self.logger.info('splitted data: {} training, {} test ({:d} portions).'.format(
            self.range_train, self.range_test, self.tt_split))

    def init_data(self):
        if (not os.path.exists(self.training_annot_cleaned)):
            from timeit import default_timer as timer
            from datetime import timedelta
            time_s = timer()
            self.logger.info('cleaning data ...')
            self.remove_out_frame_annot()
            time_e = str(timedelta(seconds=timer() - time_s))
            self.logger.info('{:d} images after cleaning, time: {}'.format(
                self.num_training, time_e))
        else:
            self.num_training = int(sum(
                1 for line in open(self.training_annot_cleaned, 'r')))

        portion = int(self.num_training / self.tt_split)
        self.range_train[0] = int(0)
        self.range_train[1] = int(portion * (self.tt_split - 1))
        self.range_test[0] = self.range_train[1]
        self.range_test[1] = self.num_training

        if ((not os.path.exists(self.training_annot_train)) or
                (not os.path.exists(self.training_annot_test))):
            self.shuffle_split()
        test_file = os.path.basename(self.training_annot_test)
        if not os.path.exists(os.path.join(self.predict_dir, test_file)):
            shutil.copy2(self.training_annot_test, self.predict_dir)

    def __init__(self, args):
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.logger = args.logger
        self.predict_dir = args.predict_dir
        self.crop_size = args.crop_size
        self.crop_range = args.crop_range
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
