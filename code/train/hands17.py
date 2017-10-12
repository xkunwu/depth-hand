import os
import shutil
import sys
import numpy as np
import matplotlib.pyplot as mpplot
# import matplotlib.image as mpimg
import matplotlib.patches as mppatches
import scipy.ndimage as spndim
import scipy.misc as spmisc
from skimage import io as skimio
import imageio
import csv
import cv2
# import coder
from random import randint
from random import random
import linecache
import re
from colour import Color
from multiprocessing.dummy import Pool as ThreadPool
# import multiprocessing
# from multiprocessing import Manager as ThreadManager
from timeit import default_timer as timer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
sys.path.append(BASE_DIR)
from args_holder import args_holder
sys.path.append(os.path.join(BASE_DIR, 'utils'))
from image_ops import make_color_range, fig2data
BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))


class hands17:
    """ Pose class for Hands17 dataset """

    # dataset info
    data_dir = ''
    training_images = ''
    frame_images = ''
    training_cropped = ''
    training_annot_origin = ''
    training_annot_cleaned = ''
    training_annot_shuffled = ''
    training_annot_cropped = ''
    training_annot_training = ''
    training_annot_evaluation = ''
    training_annot_prediction = ''
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

    # camera info
    focal = (475.065948, 475.065857)
    centre = (315.944855, 245.287079)
    # fx = 475.065948
    # fy = 475.065857
    # cx = 315.944855
    # cy = 245.287079

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

    @staticmethod
    def shuffle_annot():
        with open(hands17.training_annot_cleaned, 'r') as source:
            data = [(random(), line) for line in source]
        data.sort()
        with open(hands17.training_annot_shuffled, 'w') as target:
            for _, line in data:
                target.write(line)

    @classmethod
    def pre_provide(cls, data_dir, out_dir, rebuild=False):
        cls.data_dir = data_dir
        cls.training_images = os.path.join(data_dir, 'training/images')
        cls.frame_images = os.path.join(data_dir, 'frame/images')
        cls.training_cropped = os.path.join(out_dir, 'cropped')
        cls.training_annot_origin = os.path.join(
            data_dir, 'training/Training_Annotation.txt')
        cls.training_annot_cleaned = os.path.join(
            out_dir, 'annotation.txt')
        cls.training_annot_shuffled = os.path.join(
            out_dir, 'annotation_shuffled.txt')
        cls.training_annot_cropped = os.path.join(
            out_dir, 'annotation_cropped.txt')
        cls.training_annot_training = os.path.join(
            out_dir, 'training_training.txt')
        cls.training_annot_evaluation = os.path.join(
            out_dir, 'training_evaluate.txt')
        cls.training_annot_prediction = os.path.join(
            out_dir, 'training_predict.txt')
        cls.frame_bbox = os.path.join(data_dir, 'frame/BoundingBox.txt')

        if rebuild or (not os.path.exists(hands17.training_annot_cleaned)):
            hands17.remove_out_frame_annot()
        else:
            hands17.num_training = int(sum(
                1 for line in open(hands17.training_annot_cleaned, 'r')))

        portion = int(hands17.num_training / hands17.tt_split)
        hands17.range_train[0] = int(0)
        hands17.range_train[1] = int(portion * (hands17.tt_split - 1))
        hands17.range_test[0] = hands17.range_train[1]
        hands17.range_test[1] = hands17.num_training
        print('splitted data: {} training, {} test.'.format(
            hands17.range_train, hands17.range_test))

        if rebuild or (not os.path.exists(hands17.training_annot_shuffled)):
            hands17.shuffle_annot()
        print('using shuffled data: {}'.format(
            hands17.training_annot_shuffled))

        # if rebuild:  # just over-write, this detete operation is slow
        #     if os.path.exists(hands17.training_cropped):
        #         shutil.rmtree(hands17.training_cropped)
        if (rebuild or (not os.path.exists(hands17.training_annot_cropped)) or
                (not os.path.exists(hands17.training_cropped))):
            print('running cropping code (be patient) ...')
            # time_s = timer()
            # hands17.crop_resize_training_images()
            # print('single tread time: {:.4f}'.format(timer() - time_s))
            time_s = timer()
            hands17.crop_resize_training_images_mt()
            print('multiprocessing time: {:.4f}'.format(timer() - time_s))
        print('using cropped and resized images: {}'.format(
            hands17.training_cropped))

        if (rebuild or (not os.path.exists(hands17.training_annot_training)) or
                (not os.path.exists(hands17.training_annot_evaluation))):
            hands17.split_evaluation_images()
        print('images are splitted out for evaluation: {:d} portions'.format(
            hands17.tt_split))

    @staticmethod
    def remove_out_frame_annot():
        hands17.num_training = int(0)
        with open(hands17.training_annot_cleaned, 'w') as writer, \
                open(hands17.training_annot_origin, 'r') as reader:
            for annot_line in reader.readlines():
                _, pose_mat, rescen = hands17.parse_line_pose(annot_line)
                pose2d = hands17.get2d(pose_mat)
                if 0 > np.min(pose2d):
                    continue
                if 0 > np.min(hands17.image_size - pose2d):
                    continue
                writer.write(annot_line)
                hands17.num_training += 1

    @staticmethod
    def split_evaluation_images():
        with open(hands17.training_annot_cropped, 'r') as f:
            lines = [x.strip() for x in f.readlines()]
        with open(hands17.training_annot_training, 'w') as f:
            for line in lines[hands17.range_train[0]:hands17.range_train[1]]:
                f.write(line + '\n')
        with open(hands17.training_annot_evaluation, 'w') as f:
            for line in lines[hands17.range_test[0]:hands17.range_test[1]]:
                # name = re.search(r'(image_D\d+\.png)', line).group(1)
                # shutil.move(
                #     os.path.join(hands17.training_cropped, name),
                #     os.path.join(hands17.evaluate_cropped, name))
                f.write(line + '\n')

    @staticmethod
    def get3d(pose_mat, rescen=None):
        """ reproject 2d poses to 3d.
            pose_mat: nx3 array
        """
        if rescen is None:
            rescen = np.array([1, 0, 0])
        pose2d = pose_mat[:, 0:2] / rescen[0] + rescen[1:3]
        pose_z = np.array(pose_mat[:, 2]).reshape(-1, 1)
        pose3d = (pose2d - hands17.centre) / hands17.focal * pose_z
        return np.hstack((pose3d, pose_z))

    @staticmethod
    def compare_error(fname_echt, fname_pred):
        """ NOTE: the number of predictions might be smaller
            return: FxJ, l2 error matrix
        """
        error_l = []
        with open(fname_echt, 'r') as file_s, \
                open(fname_pred, 'r') as file_t:
            sour_lines = [x.strip() for x in file_s.readlines()]
            targ_lines = [x.strip() for x in file_t.readlines()]
            for li, line_t in enumerate(targ_lines):
                name_s, pose_s, scen_s = hands17.parse_line_pose(sour_lines[li])
                name_t, pose_t, scen_t = hands17.parse_line_pose(line_t)
                if name_s != name_t:
                    print('different names: {} - {}'.format(name_s, name_t))
                    return
                p3d_s = hands17.get3d(pose_s, scen_s)
                p3d_t = hands17.get3d(pose_t, scen_t)
                error_l.append(np.sqrt(
                    np.sum((p3d_s - p3d_t) ** 2, axis=1)
                ))
        return np.array(error_l)

    @staticmethod
    def draw_error_percentage_curve(errors, ax):
        err_max = np.max(errors, axis=1).tolist()
        num_v = len(err_max)
        percent = np.arange(num_v + 1) * 100 / num_v
        err_max = np.concatenate(([0], np.sort(err_max)))
        # fig, ax = mpplot.subplots()
        mpplot.plot(
            err_max, percent,
            '-',
            linewidth=2.0
        )
        ax.set_ylabel('Percentage (%)')
        ax.set_ylim([0, 100])
        ax.set_xlabel('Maximal error of single joint (mm)')
        ax.set_xlim(left=0)
        # ax.set_xlim(right=50)
        # mpplot.show()

    @staticmethod
    def draw_error_per_joint(errors, ax):
        err_mean = np.mean(errors, axis=0)
        err_max = np.max(errors, axis=0)
        err_min = np.min(errors, axis=0)
        err_mean = np.append(err_mean, np.mean(err_mean))
        err_max = np.append(err_max, np.mean(err_max))
        err_min = np.append(err_min, np.mean(err_min))
        err_m2m = [
            (err_mean - err_min).tolist(),
            (err_max - err_mean).tolist()
        ]
        err_mean = err_mean.tolist()
        jid = range(len(err_mean))
        jtick = hands17.join_name
        jtick.append('Mean')
        # fig, ax = mpplot.subplots()
        mpplot.bar(
            jid, err_mean, yerr=err_m2m, align='center',
            error_kw=dict(ecolor='gray', lw=1, capsize=3, capthick=2)
        )
        ax.set_ylabel('Mean error (mm)')
        ax.set_ylim(bottom=0)
        ax.set_xlim([-1, 22])
        mpplot.xticks(jid, jtick, rotation='vertical')
        mpplot.margins(0.1)
        # mpplot.show()

    @staticmethod
    def get2d(points3):
        """ project 3D point onto image plane using camera info
            Args:
                points3: nx3 array
        """
        pose_z = np.array(points3[:, 2]).reshape(-1, 1)
        return points3[:, 0:2] / pose_z * hands17.focal + hands17.centre

    @staticmethod
    def getbm(base_z, base_margin=20):
        """ return margin (x, y) accroding to projective-z of MMCP.
            Args:
                base_z: base z-value in mm
                base_margin: base margin in mm
        """
        marg = np.tile(base_margin, (2, 1)) * hands17.focal / base_z
        m = max(marg)
        return m

    @staticmethod
    def get_rect_crop_resize(annot_line):
        """
            Returns:
                p3z_crop: projected 2d coordinates, and original z on the 3rd column
        """
        img_name, pose_mat, rescen = hands17.parse_line_pose(annot_line)
        img = hands17.read_image(
            os.path.join(hands17.training_images, img_name))
        pose2d = hands17.get2d(pose_mat)
        rect = hands17.get_rect(pose2d, 0.25)
        rs = hands17.crop_size / rect[1, 1]
        rescen = np.append(rs, rect[0, :])
        p2d_crop = (pose2d - rect[0, :]) * rs
        p3z_crop = np.hstack((
            p2d_crop, np.array(pose_mat[:, 2].reshape(-1, 1)) * rs
        ))
        img_crop = img[
            int(np.floor(rect[0, 1])):int(np.ceil(rect[0, 1] + rect[1, 1])),
            int(np.floor(rect[0, 0])):int(np.ceil(rect[0, 0] + rect[1, 0]))
        ]
        # try:
        # img_crop_resize = spmisc.imresize(
        #     img_crop, (hands17.crop_size, hands17.crop_size), interp='bilinear')
        # img_crop_resize = spndim.interpolation.zoom(img_crop, rs)
        # img_crop_resize = img_crop_resize[0:hands17.crop_size, 0:hands17.crop_size]
        img_crop_resize = cv2.resize(
            img_crop, (hands17.crop_size, hands17.crop_size))
        # except:
        #     print(np.hstack((pose_mat, pose2d)))
        # print(np.max(img_crop), np.max(img_crop_resize), img_crop_resize.shape)

        return img_name, img_crop_resize, p3z_crop, rescen

    @staticmethod
    def crop_resize_save(annot_line, messages=None):
        img_name, img_crop, p3z_crop, rescen = hands17.get_rect_crop_resize(
            annot_line)
        img_crop[1 > img_crop] = 9999  # max distance set to 10m
        hands17.save_image(
            os.path.join(hands17.training_cropped, img_name),
            img_crop
        )
        # hands17.draw_hist_random(hands17.training_cropped, img_name)
        out_list = np.append(p3z_crop.flatten(), rescen.flatten()).flatten()
        crimg_line = ''.join("%12.4f" % x for x in out_list)
        pose_l = img_name + crimg_line + '\n'
        if messages is not None:
            messages.put(pose_l)
        return pose_l

    @staticmethod
    def crop_resize_training_images():
        if not os.path.exists(hands17.training_cropped):
            os.makedirs(hands17.training_cropped)
        with open(hands17.training_annot_cropped, 'w') as crop_writer:
            with open(hands17.training_annot_shuffled, 'r') as fanno:
                for line_number, annot_line in enumerate(fanno):
                    pose_l = hands17.crop_resize_save(annot_line)
                    crop_writer.write(pose_l)
                    # break

    class mt_queue_writer:
        @staticmethod
        def listener(file_name, messages):
            with open(file_name, 'w') as writer:
                while 1:
                    m = messages.get()
                    if m == '\n':
                        break
                    print(m)
                    writer.write(m)

    @staticmethod
    def crop_resize_training_images_mt():
        if not os.path.exists(hands17.training_cropped):
            os.makedirs(hands17.training_cropped)
        # thread_manager = ThreadManager()
        # messages = thread_manager.Queue()
        # thread_pool = ThreadPool(multiprocessing.cpu_count() + 2)
        # watcher = thread_pool.apply_async(
        #     hands17.mt_queue_writer.listener,
        #     (hands17.training_annot_cropped, messages))
        # jobs = []
        # with open(hands17.training_annot_shuffled, 'r') as fanno:
        #     annot_line = fanno.readline()
        #     job = thread_pool.apply_async(
        #         hands17.crop_resize_save, (annot_line, messages))
        #     jobs.append(job)
        # for job in jobs:
        #     job.get()
        # messages.put('\n')
        # thread_pool.close()
        thread_pool = ThreadPool()
        with open(hands17.training_annot_shuffled, 'r') as fanno:
            annot_list = [line for line in fanno if line]
        with open(hands17.training_annot_cropped, 'w') as writer:
            for result in thread_pool.map(hands17.crop_resize_save, annot_list):
                # (item, count) tuples from worker
                writer.write(result)
        thread_pool.close()
        thread_pool.join()

    @staticmethod
    def get_rect(pose2d, bm):
        """ return a rectangle with margin that contains 2d point set
        """
        # bm = 300
        ptp = np.ptp(pose2d, axis=0)
        ctl = np.min(pose2d, axis=0)
        cen = ctl + ptp / 2
        ptp_m = max(ptp)
        if 1 > bm:
            bm = ptp_m * bm
        ptp_m = ptp_m + 2 * bm
        # clip to image border
        ctl = cen - ptp_m / 2
        cbr = ctl + ptp_m
        obm = np.min([ctl, hands17.image_size - cbr])
        if 0 > obm:
            # print(ctl, hands17.image_size - cbr, obm, ptp_m)
            ptp_m = ptp_m + 2 * obm
        ctl = cen - ptp_m / 2
        return np.vstack((ctl, np.array([ptp_m, ptp_m])))

    @staticmethod
    def draw_pose_rect(img, pose_vec):
        """ Draw pose in a shifted window.
            Args:
        """
    @staticmethod
    def draw_pose3(img, pose_mat, show_margin=False):
        """ Draw 3D pose onto 2D image domain: using only (x, y).
            Args:
                pose_mat: nx3 array
        """
        pose2d = hands17.get2d(pose_mat)

        # draw bounding box
        # bm = hands17.getbm(pose_mat[3, 2])
        bm = 0.25
        rect = hands17.get_rect(pose2d, bm)
        mpplot.gca().add_patch(mppatches.Rectangle(
            rect[0, :], rect[1, 0], rect[1, 1],
            linewidth=1, facecolor='none',
            edgecolor=hands17.bbox_color.rgb)
        )

        img_posed = hands17.draw_pose2d(img, pose2d, show_margin)
        return img_posed

    @staticmethod
    def draw_pose2d(img, pose2d, show_margin=False):
        """ Draw 2D pose on the image domain.
            Args:
                pose2d: nx2 array
        """
        p2wrist = np.array([pose2d[0, :]])
        for fii, joints in enumerate(hands17.join_id):
            p2joints = pose2d[joints, :]
            # color_list = make_color_range(
            #     Color('black'), hands17.join_color[fii + 1], 4)
            # color_range = [C.rgb for C in make_color_range(
            #     color_list[-2], hands17.join_color[fii + 1], len(p2joints) + 1)]
            color_v0 = Color(hands17.join_color[fii + 1])
            color_v0.luminance = 0.3
            color_range = [C.rgb for C in make_color_range(
                color_v0, hands17.join_color[fii + 1], len(p2joints) + 1)]
            for jj, joint in enumerate(p2joints):
                mpplot.plot(
                    p2joints[jj, 0], p2joints[jj, 1],
                    'o',
                    color=color_range[jj + 1]
                )
            p2joints = np.vstack((p2wrist, p2joints))
            mpplot.plot(
                p2joints[:, 0], p2joints[:, 1],
                '-',
                linewidth=2.0,
                color=hands17.join_color[fii + 1].rgb
            )
            # path = mpath.Path(p2joints)
            # verts = path.interpolated(steps=step).vertices
            # x, y = verts[:, 0], verts[:, 1]
            # z = np.linspace(0, 1, step)
            # colorline(x, y, z, cmap=mpplot.get_cmap('jet'))
        # mpplot.gcf().gca().add_artist(
        #     mpplot.Circle(
        #         p2wrist[0, :],
        #         20,
        #         color=[i / 255 for i in hands17.join_color[0]]
        #     )
        # )
        mpplot.plot(
            p2wrist[0, 0], p2wrist[0, 1],
            'o',
            color=hands17.join_color[0].rgb
        )
        # for fii, bone in enumerate(hands17.bone_id):
        #     for jj in range(4):
        #         p0 = pose2d[bone[jj][0], :]
        #         p2 = pose2d[bone[jj][1], :]
        #         mpplot.plot(
        #             (int(p0[0]), int(p0[1])), (int(p2[0]), int(p2[1])),
        #             color=[i / 255 for i in hands17.join_color[fii + 1]],
        #             linewidth=2.0
        #         )
        #         # cv2.line(img,
        #         #          (int(p0[0]), int(p0[1])),
        #         #          (int(p2[0]), int(p2[1])),
        #         #          hands17.join_color[fii + 1], 1)

        return fig2data(mpplot.gcf(), show_margin)

    @staticmethod
    def read_image(image_name):
        # img = mpimg.imread(image_name)
        # img = spndim.imread(image_name)
        # img = (img - img.min()) / (img.max() - img.min()) * 255
        img = skimio.imread(image_name)
        return img

    @staticmethod
    def save_image(image_name, img):
        # mpimg.imsave(image_name, img, cmap=mpplot.cm.gray)
        # spmisc.imsave(image_name, img)
        skimio.imsave(image_name, img)

    @staticmethod
    def parse_line_pose(annot_line):
        line_l = re.split(r'\s+', annot_line.strip())
        rescen = None
        if 64 == len(line_l):
            pose_mat = np.reshape(
                [float(i) for i in line_l[1:64]],
                (hands17.join_num, 3)
            )
        elif 67 == len(line_l):
            pose_mat = np.reshape(
                [float(i) for i in line_l[1:64]],
                (hands17.join_num, 3)
            )
            rescen = np.array([float(i) for i in line_l[64:67]])
        else:
            print('error - wrong pose annotation: {}'.format(line_l))
        return line_l[0], pose_mat, rescen

    @staticmethod
    def get_line(filename, n):
        with open(filename, 'r') as f:
            for line_number, line in enumerate(f):
                if line_number == n:
                    return line

    @staticmethod
    def draw_pred_random(image_dir, annot_echt, annot_pred):
        img_id = randint(1, sum(1 for _ in open(annot_pred, 'r')))
        line_echt = linecache.getline(annot_echt, img_id)
        line_pred = linecache.getline(annot_pred, img_id)
        img_name, pose_echt, rescen_echt = hands17.parse_line_pose(line_echt)
        _, pose_pred, rescen_pred = hands17.parse_line_pose(line_pred)
        img_path = os.path.join(image_dir, img_name)
        print('drawing pose #{:d}: {}'.format(img_id, img_path))
        img = hands17.read_image(img_path)

        fig, ax = mpplot.subplots(nrows=1, ncols=2)
        mpplot.subplot(1, 2, 1)
        mpplot.imshow(img, cmap='bone')
        if rescen_echt is None:
            hands17.draw_pose3(img, pose_echt, show_margin=True)
        else:
            hands17.draw_pose3(
                img,
                hands17.get3d(pose_echt, rescen_echt),
                show_margin=True
            )
        mpplot.gcf().gca().set_title('Ground truth')
        mpplot.subplot(1, 2, 2)
        mpplot.imshow(img, cmap='bone')
        if rescen_pred is None:
            hands17.draw_pose3(img, pose_pred, show_margin=True)
        else:
            hands17.draw_pose3(
                img,
                hands17.get3d(pose_pred, rescen_pred),
                show_margin=True
            )
        mpplot.gcf().gca().set_title('Prediction')
        mpplot.show()

    @staticmethod
    def draw_pose_random(image_dir, annot_txt, img_id=-1):
        """ Draw 3D pose of a randomly picked image.
        """
        if 0 > img_id:
            # img_id = randint(1, hands17.num_training)
            img_id = randint(1, hands17.num_training)
        # Notice that linecache counts from 1
        annot_line = linecache.getline(annot_txt, img_id)
        # annot_line = linecache.getline(annot_txt, 652)  # palm
        # annot_line = linecache.getline(annot_txt, 465)  # the finger
        # print(annot_line)

        img_name, pose_mat, rescen = hands17.parse_line_pose(annot_line)
        img_path = os.path.join(image_dir, img_name)
        print('drawing pose #{:d}: {}'.format(img_id, img_path))
        img = hands17.read_image(img_path)

        mpplot.imshow(img, cmap='bone')
        if rescen is None:
            hands17.draw_pose3(img, pose_mat)
        else:
            hands17.draw_pose2d(img, pose_mat[:, 0:2])
        mpplot.show()
        return img_id

    @staticmethod
    def draw_pose_stream(gif_file, max_draw=100):
        """ Draw 3D poses and streaming output as GIF file.
        """
        with imageio.get_writer(gif_file, mode='I', duration=0.2) as gif_writer:
            with open(hands17.training_annot_cleaned, 'r') as fa:
                csv_reader = csv.reader(fa, delimiter='\t')
                for lii, annot_line in enumerate(csv_reader):
                    if lii >= max_draw:
                        return
                        # raise coder.break_with.Break
                    img_name, pose_mat, rescen = hands17.parse_line_pose(annot_line)
                    img = hands17.read_image(os.path.join(hands17.training_images, img_name))
                    mpplot.imshow(img, cmap='bone')
                    img_posed = hands17.draw_pose3(img, pose_mat)
                    # mpplot.show()
                    gif_writer.append_data(img_posed)
                    mpplot.gcf().clear()

    @staticmethod
    def parse_line_bbox(annot_line):
        line_l = re.split(r'\s+', annot_line.strip())
        bbox = np.reshape(
            [float(i) for i in line_l[1:5]],
            (2, 2)
        )
        return line_l[0], bbox

    @staticmethod
    def draw_bbox_random():
        """ Draw 3D pose of a randomly picked image.
        """
        # img_id = randint(1, hands17.num_training)
        img_id = randint(1, hands17.num_training)
        # Notice that linecache counts from 1
        annot_line = linecache.getline(hands17.frame_bbox, img_id)
        # annot_line = linecache.getline(hands17.frame_bbox, 652)

        img_name, bbox = hands17.parse_line_bbox(annot_line)
        img_path = os.path.join(hands17.frame_images, img_name)
        print('drawing BoundingBox #{:d}: {}'.format(img_id, img_path))
        img = hands17.read_image(img_path)
        mpplot.imshow(img, cmap='bone')
        # rect = bbox.astype(int)
        rect = bbox
        mpplot.gca().add_patch(mppatches.Rectangle(
            rect[0, :], rect[1, 0], rect[1, 1],
            linewidth=1, facecolor='none',
            edgecolor=hands17.bbox_color.rgb)
        )
        mpplot.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        mpplot.gcf().gca().axis('off')
        mpplot.show()

    @staticmethod
    def draw_hist_random(image_dir, img_id=-1):
        if 0 > img_id:
            img_id = randint(1, hands17.num_training)
        img_name = 'image_D{:08d}.png'.format(img_id)
        img_path = os.path.join(image_dir, img_name)
        print('drawing hist: {}'.format(img_path))
        img = hands17.read_image(img_path)

        fig, ax = mpplot.subplots(nrows=2, ncols=2)
        mpplot.subplot(2, 2, 1)
        mpplot.imshow(img, cmap='bone')
        mpplot.subplot(2, 2, 2)
        img_val = img.flatten()
        # img_val = [v for v in img.flatten() if (10 > v)]
        mpplot.hist(img_val)
        mpplot.subplot(2, 2, 3)
        img_matt = img
        img_matt[2 > img_matt] = 9999
        mpplot.imshow(img_matt, cmap='bone')
        mpplot.subplot(2, 2, 4)
        img_val = [v for v in img_matt.flatten() if (10 > v)]
        mpplot.hist(img_val)
        mpplot.show()
        return img_id


def test(args):
    hands17.pre_provide(
        args.data_dir,
        args.out_dir,
        rebuild=True
    )
    # hands17.draw_pose_stream(
    #     os.path.join(args.data_dir, 'training/pose.gif'),
    #     10
    # )
    # hands17.draw_bbox_random()
    # hands17.draw_pose_random(
    #     hands17.training_images,
    #     hands17.training_annot_cleaned
    # )
    # hands17.draw_pose_random(
    #     hands17.training_cropped,
    #     hands17.training_annot_cropped
    # )
    # hands17.draw_hist_random(
    #     # hands17.training_images
    #     hands17.training_cropped
    # )


if __name__ == '__main__':
    argsholder = args_holder()
    argsholder.parse_args()
    test(argsholder.args)
