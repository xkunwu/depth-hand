import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import imageio
import csv
# import coder
from random import randint
from random import random
import linecache
import re
from colour import Color
import scipy.misc
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Manager as ThreadManager
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(BASE_DIR, '..')
sys.path.append(BASE_DIR)
from args_holder import args_holder
sys.path.append(os.path.join(BASE_DIR, 'utils'))
from image_ops import make_color_range, fig2data


class hands17:
    """ Pose class for Hands17 dataset """

    # dataset info
    data_dir = ''
    training_images = ''
    training_cropped = ''
    training_annot = ''
    training_annot_shuffled = ''
    training_annot_cropped = ''
    evaluate_cropped = ''
    evaluate_annot_cropped = ''
    frame_images = ''
    frame_bbox = ''

    # num_training = int(957032)
    num_training = int(992)
    tt_split = int(8)
    range_train = np.zeros(2, dtype=np.int)
    range_test = np.zeros(2, dtype=np.int)

    # cropped & resized training images
    image_size = [640, 480]
    crop_size = 96

    # camera info
    fx = 475.065948
    fy = 475.065857
    cx = 315.944855
    cy = 245.287079

    # [Wrist,
    # TMCP, IMCP, MMCP, RMCP, PMCP,
    # TPIP, TDIP, TTIP,
    # IPIP, IDIP, ITIP,
    # MPIP, MDIP, MTIP,
    # RPIP, RDIP, RTIP,
    # PPIP, PDIP, PTIP]

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
        with open(hands17.training_annot, 'r') as source:
            data = [(random(), line) for line in source]
        data.sort()
        with open(hands17.training_annot_shuffled, 'w') as target:
            for _, line in data:
                target.write(line)

    @classmethod
    def pre_provide(cls, data_dir, rebuild=False):
        cls.data_dir = data_dir
        cls.training_images = os.path.join(data_dir, 'training/images')
        cls.training_cropped = os.path.join(data_dir, 'training/cropped')
        cls.training_annot = os.path.join(
            data_dir, 'training/Training_Annotation.txt')
        cls.training_annot_shuffled = os.path.join(
            data_dir, 'training/annotation_shuffled.txt')
        cls.training_annot_cropped = os.path.join(
            data_dir, 'training/annotation_cropped.txt')
        cls.frame_images = os.path.join(data_dir, 'frame/images')
        cls.frame_bbox = os.path.join(data_dir, 'frame/BoundingBox.txt')
        cls.evaluate_cropped = os.path.join(data_dir, 'training/cropped_eval')
        cls.evaluate_annot_cropped = os.path.join(
            data_dir, 'training/annotation_cropped_eval.txt')

        portion = int(hands17.num_training / hands17.tt_split)
        hands17.range_train[0] = int(0)
        hands17.range_train[1] = int(portion * 7 + 1)
        hands17.range_test[0] = hands17.range_train[1]
        hands17.range_test[1] = hands17.num_training + 1
        print('splitted data: {} training, {} test.'.format(
            hands17.range_train, hands17.range_test))

        if (not os.path.exists(hands17.training_annot_shuffled) or rebuild):
            hands17.shuffle_annot()
        print('using shuffled data: {}'.format(
            hands17.training_annot_shuffled))

        # if os.path.exists(hands17.training_annot_cropped):
        #     os.remove(hands17.training_annot_cropped)
        if (not os.path.exists(hands17.training_annot_cropped) or
                not os.path.exists(hands17.training_cropped) or rebuild):
            # hands17.crop_resize_training_images()
            hands17.crop_resize_training_images_mt()
        print('using cropped and resized images: {}'.format(
            hands17.training_cropped))

        # if (not os.path.exists(hands17.evaluate_annot_cropped) or
        #         not os.path.exists(hands17.evaluate_cropped) or rebuild):
        #     hands17.split_evaluation_images()
        # print('images for evaluation are: {}'.format(
        #     hands17.evaluate_cropped))

    @staticmethod
    def split_evaluation_images():
        if not os.path.exists(hands17.evaluate_cropped):
            os.makedirs(hands17.evaluate_cropped)

    @staticmethod
    def get2d(points3):
        """ project 3D point onto image plane using camera info
            Args:
                points3: nx3 array
        """
        r = points3[:, 0] * hands17.fx / points3[:, 2] + hands17.cx
        c = points3[:, 1] * hands17.fy / points3[:, 2] + hands17.cy
        return np.vstack((r, c)).T

    @staticmethod
    def getbm(base_z, base_margin=20):
        """ return margin (x, y) accroding to projective-z of MMCP.
            Args:
                base_z: base z-value in mm
                base_margin: base margin in mm
        """
        r = base_margin * hands17.fx / base_z
        c = base_margin * hands17.fy / base_z
        # return np.ceil(np.array([r, c]))
        m = max(r, c)
        return m

    @staticmethod
    def get_rect_crop_resize(annot_line):
        img_name, pose3 = hands17.parse_line_pose(annot_line)
        img = hands17.read_image(
            os.path.join(hands17.training_images, img_name))
        pose2d = hands17.get2d(pose3)
        rect = hands17.get_rect(pose2d, 0.25)
        rs = hands17.crop_size / rect[1, 1]
        p2d_crop = (pose2d - rect[0, :]) * rs
        img_crop = img[
            int(np.floor(rect[0, 1])):int(np.ceil(rect[0, 1] + rect[1, 1])),
            int(np.floor(rect[0, 0])):int(np.ceil(rect[0, 0] + rect[1, 0]))
        ]  # reverse (x, y) while cropping
        img_crop_resize = scipy.misc.imresize(img_crop, rs, interp='bilinear')

        # fig, ax = plt.subplots(nrows=1, ncols=2)
        # plt.subplot(1, 2, 1)
        # plt.imshow(img, cmap='bone')
        # hands17.draw_pose3(img, pose3)
        # plt.subplot(1, 2, 2)
        # plt.imshow(img_crop_resize, cmap='bone')
        # hands17.draw_pose2d(img_crop_resize, p2d_crop)
        # plt.show()
        return img_name, img_crop_resize, p2d_crop

    @staticmethod
    def crop_resize_save(annot_line, messages=None):
        img_name, img_crop, p2d_crop = hands17.get_rect_crop_resize(
            annot_line)
        crimg_line = ''.join("%10.4f" % x for x in p2d_crop.flatten())
        hands17.save_image(
            os.path.join(hands17.training_cropped, img_name),
            img_crop
        )
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
    def get_rect(points2d, bm):
        """ return a rectangle with margin that contains 2d point set
        """
        # bm = 300
        ptp = np.ptp(points2d, axis=0)
        ctl = np.min(points2d, axis=0)
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
    def draw_pose3(img, pose3):
        """ Draw 3D pose onto 2D image domain: using only (x, y).
            Args:
                pose3: nx3 array
        """
        pose2d = hands17.get2d(pose3)

        # draw bounding box
        # bm = hands17.getbm(pose3[3, 2])
        bm = 0.25
        rect = hands17.get_rect(pose2d, bm)
        plt.gca().add_patch(patches.Rectangle(
            rect[0, :], rect[1, 0], rect[1, 1],
            linewidth=1, facecolor='none',
            edgecolor=hands17.bbox_color.rgb)
        )

        img_posed = hands17.draw_pose2d(img, pose2d)
        return img_posed

    @staticmethod
    def draw_pose2d(img, pose2d):
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
                plt.plot(
                    p2joints[jj, 0], p2joints[jj, 1],
                    'o',
                    color=color_range[jj + 1]
                )
            p2joints = np.vstack((p2wrist, p2joints))
            plt.plot(
                p2joints[:, 0], p2joints[:, 1],
                '-',
                linewidth=2.0,
                color=hands17.join_color[fii + 1].rgb
            )
            # path = mpath.Path(p2joints)
            # verts = path.interpolated(steps=step).vertices
            # x, y = verts[:, 0], verts[:, 1]
            # z = np.linspace(0, 1, step)
            # colorline(x, y, z, cmap=plt.get_cmap('jet'))
        # plt.gcf().gca().add_artist(
        #     plt.Circle(
        #         p2wrist[0, :],
        #         20,
        #         color=[i / 255 for i in hands17.join_color[0]]
        #     )
        # )
        plt.plot(
            p2wrist[0, 0], p2wrist[0, 1],
            'o',
            color=hands17.join_color[0].rgb
        )
        # for fii, bone in enumerate(hands17.bone_id):
        #     for jj in range(4):
        #         p0 = pose2d[bone[jj][0], :]
        #         p2 = pose2d[bone[jj][1], :]
        #         plt.plot(
        #             (int(p0[0]), int(p0[1])), (int(p2[0]), int(p2[1])),
        #             color=[i / 255 for i in hands17.join_color[fii + 1]],
        #             linewidth=2.0
        #         )
        #         # cv2.line(img,
        #         #          (int(p0[0]), int(p0[1])),
        #         #          (int(p2[0]), int(p2[1])),
        #         #          hands17.join_color[fii + 1], 1)

        # return fig2data(plt.gcf(), show_axis=True)
        return fig2data(plt.gcf())

    @staticmethod
    def read_image(image_name):
        img = mpimg.imread(image_name)
        # img = (img - img.min()) / (img.max() - img.min()) * 255
        return img

    @staticmethod
    def save_image(image_name, img):
        mpimg.imsave(image_name, img)

    @staticmethod
    def parse_line_pose(annot_line):
        line_l = re.split(r'\s+', annot_line.strip())
        if 64 == len(line_l):
            pose3 = np.reshape(
                [float(i) for i in line_l[1:64]],
                (21, 3)
            )
        elif 43 == len(line_l):
            pose3 = np.reshape(
                [float(i) for i in line_l[1:43]],
                (21, 2)
            )
        else:
            print('error - wrong pose annotation!\n')
        return line_l[0], pose3

    @staticmethod
    def get_line(filename, n):
        with open(filename, 'r') as f:
            for line_number, line in enumerate(f):
                if line_number == n:
                    return line

    @staticmethod
    def draw_pose2_random(image_dir, annot_txt):
        """ Draw 3D pose of a randomly picked image.
        """
        # img_id = randint(1, hands17.num_training)
        img_id = randint(1, 999)
        print('drawing pose: # {}'.format(img_id))
        # Notice that linecache counts from 1
        annot_line = linecache.getline(annot_txt, img_id)
        # annot_line = linecache.getline(annot_txt, 652)
        # print(annot_line)

        img_name, pose3 = hands17.parse_line_pose(annot_line)
        img = hands17.read_image(os.path.join(image_dir, img_name))

        plt.imshow(img, cmap='bone')
        if 3 == pose3.shape[1]:
            hands17.draw_pose3(img, pose3)
        else:
            hands17.draw_pose2d(img, pose3)
        plt.show()

    @staticmethod
    def draw_pose2_stream(gif_file, max_draw=100):
        """ Draw 3D poses and streaming output as GIF file.
        """
        with imageio.get_writer(gif_file, mode='I', duration=0.2) as gif_writer:
            with open(hands17.training_annot, 'r') as fa:
                csv_reader = csv.reader(fa, delimiter='\t')
                for lii, annot_line in enumerate(csv_reader):
                    if lii >= max_draw:
                        return
                        # raise coder.break_with.Break
                    img_name, pose3 = hands17.parse_line_pose(annot_line)
                    img = hands17.read_image(os.path.join(hands17.training_images, img_name))
                    plt.imshow(img, cmap='bone')
                    img_posed = hands17.draw_pose3(img, pose3)
                    # plt.show()
                    gif_writer.append_data(img_posed)
                    plt.gcf().clear()

    @staticmethod
    def parse_line_bbox(annot_line):
        line_l = re.split(r'\s+', annot_line.strip())
        pose3 = np.reshape(
            [float(i) for i in line_l[1:5]],
            (2, 2)
        )
        return line_l[0], pose3

    @staticmethod
    def draw_bbox_random():
        """ Draw 3D pose of a randomly picked image.
        """
        # img_id = randint(1, hands17.num_training)
        img_id = randint(1, 999)
        print('drawing BoundingBox: # {}'.format(img_id))
        # Notice that linecache counts from 1
        annot_line = linecache.getline(hands17.frame_bbox, img_id)
        # annot_line = linecache.getline(hands17.frame_bbox, 652)
        # print(annot_line)

        img_name, bbox = hands17.parse_line_bbox(annot_line, hands17.frame_images)
        img = hands17.read_image(os.path.join(hands17.frame_images, img_name))
        plt.imshow(img, cmap='bone')
        # rect = bbox.astype(int)
        rect = bbox
        plt.gca().add_patch(patches.Rectangle(
            rect[0, :], rect[1, 0], rect[1, 1],
            linewidth=1, facecolor='none',
            edgecolor=hands17.bbox_color.rgb)
        )
        plt.show()


def test(args):
    hands17.pre_provide(args.data_dir)
    # hands17.draw_pose2_stream(
    #     os.path.join(args.data_dir, 'training/pose.gif'),
    #     10
    # )
    # hands17.draw_bbox_random()
    # hands17.draw_pose2_random(
    #     hands17.training_images,
    #     hands17.training_annot
    # )
    # hands17.draw_pose2_random(
    #     hands17.training_cropped,
    #     hands17.training_annot_cropped
    # )


if __name__ == '__main__':
    argsholder = args_holder()
    argsholder.parse_args()
    test(argsholder.args)
