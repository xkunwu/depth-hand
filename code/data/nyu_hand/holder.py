import os
import numpy as np
from colour import Color
import progressbar
import h5py
import scipy.io
from . import ops as dataops
from . import io as dataio
from . import provider as datapro
from utils.iso_boxes import iso_cube
from utils.coder import file_pack
from model.batch_allot import batch_index


class nyu_handholder:
    """ Pose class for nyu hand dataset
        coordinate system:
            ---> y
            |
            |x
        which is the same as row-major np arrays.
    """

    image_size = (480, 640)
    region_size = 120.  # empirical spacial cropping radius
    crop_size = 128  # input image size to models (may changed)
    # hmap_size = 32  # for detection models
    # anchor_num = 16  # for attention model
    crop_range = 480.  # +/- spacial capture range
    z_range = (100., 1060.)  # empirical valid depth range
    z_max = 9999.  # max distance set to 10m
    # camera info
    focal = (588.036865, 587.075073)
    centre = (320, 240)
    # focal = (587.075073, 588.036865)
    # centre = (240, 320)

    # joints description
    join_name = [
        'Palm',
        'Wrist1', 'Wrist2',
        'Thumb.R1', 'Thumb.R2', 'Thumb.T',
        'Index.R', 'Index.T',
        'Mid.R', 'Mid.T',
        'Ring.R', 'Ring.T',
        'Pinky.R', 'Pinky.T'
    ]

    join_num = 14
    join_type = ('C', 'W', 'T', 'I', 'M', 'R', 'P')
    join_color = (
        Color('black'),
        Color('cyan'),
        Color('magenta'),
        Color('blue'),
        Color('lime'),
        Color('yellow'),
        Color('red')
    )
    join_keep = (0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 27, 30, 31, 32)
    join_id = (
        (0),
        (1, 2),
        (3, 4, 5),
        (6, 7),
        (8, 9),
        (10, 11),
        (12, 13)
    )
    bbox_color = Color('orange')

    def _prepare_data_full_path(
        self, target, precon_list, store_name,
            path_pre_fn, mode,
            filepack, batchallot):
        precon_h5 = {}
        num_line = 0
        for precon in precon_list:
            precon_h5[precon] = filepack.push_h5(
                path_pre_fn(mode, store_name[precon]))
            num_line = precon_h5[precon][precon].shape[0]
        store_size = batchallot.store_size
        print(
            'preparing data ({}): {:d} lines with store size {:d} ...'.format(
                path_pre_fn(mode, store_name[target]), num_line, store_size))
        target_h5, batch_data = batchallot.create_fn[target](
            filepack, path_pre_fn(mode, store_name[target]), num_line
        )
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
            write_end = min(write_beg + store_size, num_line)
            proc_size = write_end - write_beg
            if 0 >= proc_size:
                break
            args = [range(proc_size)]
            for precon in precon_list:
                args.append(precon_h5[precon][precon][
                    write_beg:write_end, ...])
            datapro.puttensor_mt(
                args, self.store_prow[target],
                self, mode, batch_data
            )
            target_h5[write_beg:write_end, ...] = \
                batch_data[0:proc_size, ...]
            write_beg = write_end
            li += proc_size
            timerbar.update(li)
        timerbar.finish()

    def prepare_data_recur(
            self, target, store_name, filepack, batchallot):
        precon_list = self.store_precon[target]
        if not precon_list:
            return
        for precon in precon_list:
            self.prepare_data_recur(
                precon, store_name, filepack, batchallot)
        path_train = self.prepared_join('train', store_name[target])
        if not os.path.exists(path_train):
            self._prepare_data_full_path(
                target, precon_list, store_name,
                self.prepared_join, 'train',
                filepack, batchallot
            )
        path_test = self.prepared_join('test', store_name[target])
        if not os.path.exists(path_test):
            self._prepare_data_full_path(
                target, precon_list, store_name,
                self.prepared_join, 'test',
                filepack, batchallot
            )

    def _remove_out_frame_mat(
        self, mat_name, h5_name, mode, filepack,
            shuffle=False, num_line=None):
        mat_reader = scipy.io.loadmat(mat_name)
        mat_xyz = mat_reader['joint_xyz']
        # mat_xyz_shape = mat_xyz.shape
        # mat_xyz = mat_xyz.transpose().reshape(mat_xyz_shape)
        if num_line is None:
            num_line = mat_xyz.shape[1]
        shuffleid = np.arange(num_line)
        if shuffle:
            np.random.shuffle(shuffleid)
        print(mat_xyz.flags)
        print(mat_xyz.shape)
        poses_sel = mat_xyz[0, ...]
        print(poses_sel.shape)
        poses_sel = poses_sel[:num_line, ...]
        print(poses_sel.shape)
        poses_sel = poses_sel[:, self.join_keep, :]
        print(poses_sel.flags)
        print(poses_sel.shape)
        # poses_sel = mat_xyz[0, :num_line, self.join_keep, :]
        poses_sel[..., 1] *= -1
        # poses_sel[..., [0, 1]] = poses_sel[..., [1, 0]]
        poses_sel = poses_sel[:, ::-1, :]
        poses_sel = poses_sel.flatten(order='A')
        poses_sel = poses_sel.reshape(-1, self.join_num * 3)
        if shuffle:
            poses_sel = poses_sel[shuffleid, ...]
        print(poses_sel.flags)
        print(poses_sel.shape)
        # import matplotlib.pyplot as mpplot
        # p0 = poses_sel[0, ...].reshape(-1, 3)
        # pose2d, pose_z = dataops.raw_to_2dz(p0, self)
        # # mpplot.imshow(dep, cmap=mpplot.cm.bone_r)
        # mpplot.plot(pose2d[:, 0], pose2d[:, 1], 'o')
        # mpplot.show()
        batchallot = batch_index(self, num_line)
        store_size = batchallot.store_size
        h5file, batch_data = batchallot.create_index(
            filepack, h5_name, num_line, None)
        pose_limit = np.array([
            [np.inf, np.inf, np.inf],
            [-np.inf, -np.inf, -np.inf],
        ])
        timerbar = progressbar.ProgressBar(
            maxval=num_line,
            widgets=[
                progressbar.Percentage(),
                ' ', progressbar.Bar('=', '[', ']'),
                ' ', progressbar.ETA()]
        ).start()
        read_beg = 0
        write_beg = 0
        li = 0
        while True:
            read_end = min(read_beg + store_size, num_line)
            proc_size = read_end - read_beg
            if 0 >= proc_size:
                break
            # index = (write_beg + 1) + np.arange(proc_size)
            index = shuffleid[read_beg:read_end] + 1
            poses = poses_sel[read_beg:read_end, :]
            args = [np.arange(proc_size), index, poses]
            batch_data['valid'] = np.full(
                (store_size,), False)
            datapro.puttensor_mt(
                args, datapro.prow_index,
                self, mode, batch_data
            )
            valid = batch_data['valid']
            valid_num = np.sum(valid)
            write_end = write_beg + valid_num
            if write_beg < write_end:
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
            read_beg = read_end
            write_beg = write_end
            li += proc_size
            if 0 == (li % 10e2):
                timerbar.update(li)
        timerbar.finish()
        batchallot.resize(h5file, write_beg)
        # if shuffle:
        #     batchallot.shuffle(h5file)
        return write_beg, pose_limit

    # def _merge_annotation(self):
    #     with file_pack() as filepack:
    #         batchallot = batch_index(self, self.num_annotation)
    #         h5out, _ = batchallot.create_index(
    #             filepack, self.annotation_cleaned,
    #             self.num_annotation, None)
    #         batchallot.write(
    #             filepack.push_h5(self.annotation_train),
    #             h5out, 0, self.num_training)
    #         batchallot.write(
    #             filepack.push_h5(self.annotation_test),
    #             h5out, self.num_training, self.num_annotation)

    def remove_out_frame_annot(self):
        annot_origin_train = os.path.join(
            self.data_dir, 'train', 'joint_data.mat')
        annot_origin_test = os.path.join(
            self.data_dir, 'test', 'joint_data.mat')
        annotation_train = self.prepared_join('train', self.annotation)
        annotation_test = self.prepared_join('test', self.annotation)
        self.num_training = int(0)
        pose_limit = np.array([
            [np.inf, np.inf, np.inf],
            [-np.inf, -np.inf, -np.inf],
        ])
        num_eval = self.args.num_eval
        with file_pack() as filepack:
            if num_eval is None:
                self.num_training, lim = self._remove_out_frame_mat(
                    annot_origin_train, annotation_train, 'train',
                    filepack, shuffle=True)
            else:
                self.num_training, lim = self._remove_out_frame_mat(
                    annot_origin_train, annotation_train, 'train',
                    filepack, shuffle=True, num_line=(num_eval * 10))
            pose_limit[0, :] = np.minimum(
                pose_limit[0, :],
                lim[0, :])
            pose_limit[1, :] = np.maximum(
                pose_limit[1, :],
                lim[1, :])
        with file_pack() as filepack:
            if num_eval is None:
                self.num_evaluate, lim = self._remove_out_frame_mat(
                    annot_origin_test, annotation_test, 'test',
                    filepack)
            else:
                self.num_evaluate, lim = self._remove_out_frame_mat(
                    annot_origin_test, annotation_test, 'test',
                    filepack, num_line=num_eval)
            pose_limit[0, :] = np.minimum(
                pose_limit[0, :],
                lim[0, :])
            pose_limit[1, :] = np.maximum(
                pose_limit[1, :],
                lim[1, :])
        self.num_annotation = self.num_training + self.num_evaluate
        # self._merge_annotation()
        return pose_limit

    def init_data(self):
        update_log = False
        annotation_train = self.prepared_join('train', self.annotation)
        annotation_test = self.prepared_join('test', self.annotation)
        if (
                (not os.path.exists(annotation_train)) or
                (not os.path.exists(annotation_test))):
            from timeit import default_timer as timer
            from datetime import timedelta
            time_s = timer()
            self.logger.info('cleaning data ...')
            pose_limit = self.remove_out_frame_annot()
            time_e = str(timedelta(seconds=timer() - time_s))
            self.logger.info('{:d} training images, {:d} evaluate images after cleaning, time: {}'.format(
                self.num_training, self.num_evaluate, time_e))
            self.logger.info('pose limit: {} --> {}'.format(
                pose_limit[0, :], pose_limit[1, :])
            )
            update_log = True
            dataio.h5_to_txt(
                annotation_test,
                annotation_test + '.txt')
        else:
            with h5py.File(annotation_train, 'r') as h5file:
                self.num_training = h5file['index'].shape[0]
            with h5py.File(annotation_test, 'r') as h5file:
                self.num_evaluate = h5file['index'].shape[0]
            self.num_annotation = self.num_training + self.num_evaluate

        # split the data into 10 portions
        split_num = int(10)
        portion = int(np.ceil(float(self.num_training) / split_num))
        # the last portion is used for test (compare models)
        self.train_test_split = 0
        self.train_valid_split = int(portion * (split_num - 1))
        self.logger.info('collected {:d} images: {:d} training, {:d} validation, {:d} evaluate'.format(
            self.num_annotation, self.train_valid_split,
            self.num_training - self.train_valid_split,
            self.num_evaluate))

        return update_log

    def images_join(self, mode, filename):
        return os.path.join(self.data_dir, mode, filename)

    def prepared_join(self, mode, filename):
        return os.path.join(self.out_dir, 'prepared', mode, filename)

    def __init__(self, args):
        self.args = args
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.logger = args.logger
        self.crop_size = args.crop_size
        # self.anchor_num = args.anchor_num
        self.crop_range = args.crop_range
        # self.z_range[1] = self.crop_range * 2. + self.z_range[0]
        # self.training_images = os.path.join(self.data_dir, 'train')
        # self.test_images = os.path.join(self.data_dir, 'test')
        self.annotation = 'annotation'
        # self.annotation_train = self.annotation_train_join(
        #     self.annotation)
        # self.annotation_test = self.annotation_test_join(
        #     self.annotation)
        prepared_train = self.prepared_join('train', '')
        if not os.path.exists(prepared_train):
            os.makedirs(prepared_train)
        prepared_test = self.prepared_join('test', '')
        if not os.path.exists(prepared_test):
            os.makedirs(prepared_test)
        self.store_precon = {
            'index': [],
            'poses': [],
            'resce': [],
            'pose_c': ['poses', 'resce'],
            'crop2': ['index', 'resce'],
            'clean': ['index', 'resce'],
        }
        self.store_prow = {
            'pose_c': datapro.prow_pose_c,
            # 'pose_c1': datapro.prow_pose_c1,
            # 'pose_hit': datapro.prow_pose_hit,
            # 'pose_lab': datapro.prow_pose_lab,
            'crop2': datapro.prow_crop2,
            'clean': datapro.prow_clean,
            'ortho3': datapro.prow_ortho3,
            # 'pcnt3': datapro.prow_pcnt3,
            # 'truncd': datapro.prow_truncd,
            # 'tsdf3': datapro.prow_tsdf3,
            # 'vxhit': datapro.prow_vxhit,
            # 'vxoff': datapro.prow_vxoff,
            # 'vxudir': datapro.prow_vxudir,
            # 'hmap2': datapro.prow_hmap2,
            'udir2': datapro.prow_udir2,
            # 'vxedt': datapro.prow_vxedt,
            'edt2': datapro.prow_edt2,
            # 'ov3edt2': datapro.prow_ov3edt2,
            'edt2m': datapro.prow_edt2m,
            # 'ov3dist2': datapro.prow_ov3dist2,
            # 'ov3edt2m': datapro.prow_ov3edt2m,
        }
