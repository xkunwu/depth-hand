import os
# import sys
import shutil
import numpy as np
from colour import Color
import progressbar
import h5py
from . import ops as dataops
from . import io as dataio
from . import provider as datapro
from utils.iso_boxes import iso_cube


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
    region_size = 120.  # empirical spacial cropping radius
    crop_size = 128  # input image size to models (may changed)
    # hmap_size = 32  # for detection models
    # anchor_num = 16  # for attention model
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
        with h5py.File(self.training_annot_cleaned, 'w') as h5file, \
                open(self.training_annot_origin, 'r') as reader:
            lines = reader.readlines()
            num_line = len(lines)
            h5file.create_dataset(
                'index',
                (num_line,),
                maxshape=(num_line,),
                compression='lzf',
                dtype='i4'
            )
            h5file.create_dataset(
                'poses',
                (num_line, self.join_num * 3),
                maxshape=(num_line, self.join_num * 3),
                compression='lzf',
                dtype='f4')
            h5file.create_dataset(
                'resce',
                (num_line, 4),
                maxshape=(num_line, 4),
                compression='lzf',
                dtype='f4')
            timerbar = progressbar.ProgressBar(
                maxval=num_line,
                widgets=[
                    progressbar.Percentage(),
                    ' ', progressbar.Bar('=', '[', ']'),
                    ' ', progressbar.ETA()]
            ).start()
            for li, line in enumerate(lines):
                img_name, pose_raw = dataio.parse_line_annot(line)
                pose2d = dataops.raw_to_2d(pose_raw, self)
                if 0 > np.min(pose2d):
                    continue
                if 0 > np.min(self.image_size - pose2d):
                    continue
                index = dataio.imagename2index(img_name)
                cube = iso_cube(
                    (np.max(pose_raw, axis=0) + np.min(pose_raw, axis=0)) / 2,
                    self.region_size
                )
                h5file['index'][self.num_training] = index
                h5file['poses'][self.num_training, ...] = pose_raw.flatten()
                h5file['resce'][self.num_training, ...] = cube.dump()
                pose_limit[0, :] = np.minimum(
                    pose_limit[0, :],
                    np.min(pose_raw, axis=0))
                pose_limit[1, :] = np.maximum(
                    pose_limit[1, :],
                    np.max(pose_raw, axis=0))
                self.num_training += 1
                if 0 == (li % 10e2):
                    timerbar.update(li)
            timerbar.finish()
            h5file['index'].resize((self.num_training,))
            h5file['poses'].resize((self.num_training, self.join_num * 3))
            h5file['resce'].resize((self.num_training, 4))
        return pose_limit

    def remove_out_frame_annot_mt(self):
        from itertools import islice
        from utils.coder import file_pack
        from model.batch_allot import batch_index
        self.num_training = int(0)
        pose_limit = np.array([
            [np.inf, np.inf, np.inf],
            [-np.inf, -np.inf, -np.inf],
        ])
        with file_pack() as filepack:
            reader = filepack.push_file(self.training_annot_origin)
            num_line = int(sum(1 for line in reader))
            reader.seek(0)
            batchallot = batch_index(self, num_line)
            store_size = batchallot.store_size
            h5file, batch_data = batchallot.create_index(
                filepack, self.training_annot_cleaned, num_line)
            timerbar = progressbar.ProgressBar(
                maxval=num_line,
                widgets=[
                    progressbar.Percentage(),
                    ' ', progressbar.Bar('=', '[', ']'),
                    ' ', progressbar.ETA()]
            ).start()
            write_beg = 0
            li = 0
            while True:
                next_n_lines = list(islice(reader, store_size))
                if not next_n_lines:
                    break
                proc_size = len(next_n_lines)
                args = [range(proc_size), next_n_lines]
                batch_data['valid'] = np.full(
                    (store_size,), False)
                datapro.puttensor_mt(
                    args,
                    datapro.prow_index, self, batch_data
                )
                valid = batch_data['valid']
                valid_num = np.sum(valid)
                write_end = write_beg + valid_num
                if write_beg == write_end:
                    break
                h5file['index'][write_beg:write_end, ...] = \
                    batch_data['index'][valid, ...]
                poses = batch_data['poses'][valid, ...]
                h5file['poses'][write_beg:write_end, ...] = \
                    poses
                h5file['resce'][write_beg:write_end, ...] = \
                    batch_data['resce'][valid, ...]
                poses = poses.reshape(-1, 3)
                pose_limit[0, :] = np.minimum(
                    pose_limit[0, :],
                    np.min(poses, axis=0))
                pose_limit[1, :] = np.maximum(
                    pose_limit[1, :],
                    np.max(poses, axis=0))
                write_beg = write_end
                self.num_training += valid_num
                li += proc_size
                if 0 == (li % 10e2):
                    timerbar.update(li)
            timerbar.finish()
            batchallot.resize(h5file, self.num_training)
        return pose_limit

    # def shuffle_split(self):
    #     with open(self.training_annot_cleaned, 'r') as source:
    #         lines = source.readlines()
    #     # import random
    #     # np.random.shuffle(lines)
    #     with open(self.training_annot_train, 'w') as f:
    #         for line in lines[:self.train_test_split]:
    #             f.write(line)
    #     with open(self.training_annot_test, 'w') as f:
    #         for line in lines[self.train_test_split:]:
    #             # name = re.match(r'^(image_D\d+\.png)', line).group(1)
    #             # shutil.move(
    #             #     os.path.join(self.training_cropped, name),
    #             #     os.path.join(self.evaluate_cropped, name))
    #             f.write(line)

    # def next_valid_split(self):
    #     """ split range for validation set """
    #     # split_beg = self.portion * self.split_id
    #     # self.split_id = (self.split_id + 1) % self.split_num
    #     # split_end = self.portion * self.split_id
    #     split_beg = self.portion * self.split_id
    #     split_end = 0
    #     return split_beg, split_end

    def init_data(self):
        update_log = False
        if (not os.path.exists(self.training_annot_cleaned)):
            from timeit import default_timer as timer
            from datetime import timedelta
            time_s = timer()
            self.logger.info('cleaning data ...')
            # pose_limit = self.remove_out_frame_annot()
            pose_limit = self.remove_out_frame_annot_mt()
            time_e = str(timedelta(seconds=timer() - time_s))
            self.logger.info('{:d} images after cleaning, time: {}'.format(
                self.num_training, time_e))
            self.logger.info('pose limit: {} --> {}'.format(
                pose_limit[0, :], pose_limit[1, :])
            )
            update_log = True
        else:
            with h5py.File(self.training_annot_cleaned, 'r') as h5file:
                self.num_training = h5file['index'].shape[0]
            self.logger.info('collected {:d} images'.format(
                self.num_training))

        # split the data into 10 portions
        split_num = int(10)
        portion = int(np.ceil(float(self.num_training) / split_num))
        # the last portion is used for test (compare models)
        self.train_test_split = int(portion * (split_num - 1))
        # self.split_num = split_num - 1
        # self.portion = portion
        # # 1 out of (10 - 1) portions is picked out for validation
        # self.split_id = -1 % self.split_num
        self.train_valid_split = int(portion * (split_num - 2))

        # if ((not os.path.exists(self.training_annot_train)) or
        #         (not os.path.exists(self.training_annot_test))):
        #     self.shuffle_split()
        #     self.logger.info('splitted data: {} training, {} test ({:d} portions).'.format(
        #         self.train_test_split,
        #         self.num_training - self.train_test_split,
        #         split_num))
        test_file = self.training_annot_test
        if not os.path.exists(test_file):
            # shutil.copy2(self.training_annot_test, self.predict_dir)
            with h5py.File(self.training_annot_cleaned, 'r') as h5file:
                index = h5file['index'][self.train_test_split:, ...]
                poses = h5file['poses'][self.train_test_split:, ...]
                with h5py.File(test_file + '.h5', 'w') as writer:
                    dataio.write_h5(writer, index, poses)
                with open(test_file, 'w') as writer:
                    dataio.write_txt(writer, index, poses)
        # with h5py.File(test_file, 'r') as reader:
        #     with open(test_file + '_1.txt', 'w') as writer:
        #         dataio.h5_to_txt(reader, writer)

        return update_log

    def __init__(self, args):
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.logger = args.logger
        self.predict_dir = args.predict_dir
        self.crop_size = args.crop_size
        # self.anchor_num = args.anchor_num
        self.crop_range = args.crop_range
        # self.z_range[1] = self.crop_range * 2. + self.z_range[0]
        self.training_images = os.path.join(self.data_dir, 'training/images')
        self.frame_images = os.path.join(self.data_dir, 'frame/images')
        self.training_annot_origin = os.path.join(
            self.data_dir, 'training/Training_Annotation.txt')
        self.training_annot_cleaned = os.path.join(
            self.out_dir, 'annotation')
        # self.training_annot_train = os.path.join(
        #     self.out_dir, 'training_train')
        self.training_annot_train = self.training_annot_cleaned
        self.training_annot_test = os.path.join(
            self.predict_dir, 'training_test')
        self.frame_bbox = os.path.join(self.data_dir, 'frame/BoundingBox.txt')
        # self.store_file = {
        #     'index': self.training_annot_train,
        #     'poses': self.training_annot_train,
        #     'resce': self.training_annot_train
        # }
        self.store_prow = {
            'pose_c': datapro.prow_pose_c,
            'pose_c1': datapro.prow_pose_c1,
            'pose_hit': datapro.prow_pose_hit,
            'pose_lab': datapro.prow_pose_lab,
            'crop2': datapro.prow_crop2,
            'clean': datapro.prow_clean,
            'ortho3': datapro.prow_ortho3,
            'pcnt3': datapro.prow_pcnt3,
            'truncd': datapro.prow_truncd,
            'tsdf3': datapro.prow_tsdf3,
            'vxhit': datapro.prow_vxhit,
            'vxoff': datapro.prow_vxoff,
            'vxudir': datapro.prow_vxudir,
            'hmap2': datapro.prow_hmap2,
            'udir2': datapro.prow_udir2,
            'vxedt': datapro.prow_vxedt,
            'edt2': datapro.prow_edt2,
            'ov3edt2': datapro.prow_ov3edt2,
            'edt2m': datapro.prow_edt2m,
            'ov3dist2': datapro.prow_ov3dist2,
            'ov3edt2m': datapro.prow_ov3edt2m,
        }
