import os
# import sys
import shutil
import numpy as np
from colour import Color
import progressbar
import ops as dataops
import io as dataio


class hands17holder:
    """ Pose class for Hands17 dataset """

    # dataset info
    data_dir = ''
    out_dir = ''
    predict_dir = ''
    training_images = ''
    frame_images = ''
    training_cropped = ''
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
    image_size = [640, 480]
    crop_size = 96
    z_near = 1
    z_far = 3333
    z_max = 9999  # max distance set to 10m
    # camera info
    focal = (475.065948, 475.065857)
    centre = (315.944855, 245.287079)
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
        # random.shuffle(lines)
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

    # def crop_resize_save(self, annot_line, messages=None):
    #     img_name, img_crop, p3z_crop, resce = self.get_rect_crop_resize(
    #         annot_line)
    #     img_crop[self.z_near > img_crop] = self.z_max
    #     img_crop[self.z_far < img_crop] = self.z_max
    #     dataio.save_image(
    #         os.path.join(self.training_cropped, img_name),
    #         img_crop
    #     )
    #     # self.draw_hist_random(self.training_cropped, img_name)
    #     out_list = np.append(p3z_crop.flatten(), resce.flatten()).flatten()
    #     crimg_line = ''.join("%12.4f" % x for x in out_list)
    #     pose_l = img_name + crimg_line + '\n'
    #     if messages is not None:
    #         messages.put(pose_l)
    #     return pose_l

    def init_data(self, rebuild=False):
        if rebuild:
            shutil.rmtree(self.out_dir)
            os.makedirs(self.out_dir)
        if (not os.path.exists(self.training_annot_cleaned)):
            print('cleaning data ...')
            # from timeit import default_timer as timer
            self.remove_out_frame_annot()
            print('data cleaned, using: {}'.format(
                self.training_annot_cleaned))
        else:
            self.num_training = int(sum(
                1 for line in open(self.training_annot_cleaned, 'r')))
        print('total number of images: {:d}'.format(
            self.num_training))

        portion = int(self.num_training / self.tt_split)
        self.range_train[0] = int(0)
        self.range_train[1] = int(portion * (self.tt_split - 1))
        self.range_test[0] = self.range_train[1]
        self.range_test[1] = self.num_training
        print('splitted data: {} training, {} test.'.format(
            self.range_train, self.range_test))

        if ((not os.path.exists(self.training_annot_train)) or
                (not os.path.exists(self.training_annot_test))):
            self.shuffle_split()
        test_file = os.path.basename(self.training_annot_test)
        if not os.path.exists(os.path.join(self.predict_dir, test_file)):
            shutil.copy2(self.training_annot_test, self.predict_dir)
        print('images are splitted out for evaluation: {:d} portions'.format(
            self.tt_split))

    def __init__(self, args):
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.crop_size = args.crop_size
        self.predict_dir = os.path.join(self.out_dir, 'predict')
        if not os.path.exists(self.predict_dir):
            os.mkdir(self.predict_dir)
        self.training_images = os.path.join(self.data_dir, 'training/images')
        self.frame_images = os.path.join(self.data_dir, 'frame/images')
        self.training_cropped = os.path.join(self.out_dir, 'cropped')
        self.training_annot_origin = os.path.join(
            self.data_dir, 'training/Training_Annotation.txt')
        self.training_annot_cleaned = os.path.join(
            self.out_dir, 'annotation.txt')
        self.training_annot_train = os.path.join(
            self.out_dir, 'training_train.txt')
        self.training_annot_test = os.path.join(
            self.out_dir, 'training_test.txt')
        self.frame_bbox = os.path.join(self.data_dir, 'frame/BoundingBox.txt')
