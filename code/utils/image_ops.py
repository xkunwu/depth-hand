import os
import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio
import csv
import coder
from random import randint
import linecache
import re


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 3D numpy array with RGBA channels
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.gcf().gca().axis('off')
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    ncols, nrows = fig.canvas.get_width_height()
    buf = np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


class pose_hands17:
    """ Pose class for Hands17 dataset """

    # dataset info
    num_training = 957032

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
        # (64, 64, 64),
        (0, 255, 255),
        (255, 0, 255),
        (0, 0, 255),
        (0, 255, 0),
        (255, 255, 0),
        (255, 0, 0)
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

    @classmethod
    def get2d(cls, points3):
        """ project 3D point onto image plane using camera info
            Args:
                points3: nx3 array
        """
        r = points3[:, 0] * cls.fx / points3[:, 2] + cls.cx
        c = points3[:, 1] * cls.fy / points3[:, 2] + cls.cy
        return np.vstack((r, c)).T.astype(int)

    @classmethod
    def draw_pose2(cls, img, pose3):
        """ Draw 3D pose onto 2D image domain: using only (x, y).
            Args:
                pose3: nx3 array
        """
        p2wrist = cls.get2d(np.array([pose3[0, :]]))
        for fii, joints in enumerate(cls.join_id):
            p2joints = cls.get2d(pose3[joints, :])
            plt.plot(
                p2joints[:, 0], p2joints[:, 1],
                'o',
                color=[i / 255 for i in cls.join_color[fii + 1]]
            )
            p2joints = np.vstack((p2wrist, p2joints))
            plt.plot(
                p2joints[:, 0], p2joints[:, 1],
                '-',
                linewidth=2.0,
                color=[i / 255 for i in cls.join_color[fii + 1]]
            )
        # plt.gcf().gca().add_artist(
        #     plt.Circle(
        #         p2wrist[0, :],
        #         20,
        #         color=[i / 255 for i in cls.join_color[0]]
        #     )
        # )
        plt.plot(
            p2wrist[0, 0], p2wrist[0, 1],
            'o',
            color=[i / 255 for i in cls.join_color[0]]
        )
        # for fii, bone in enumerate(cls.bone_id):
        #     for jj in range(4):
        #         p0 = cls.get2d(pose3[bone[jj][0], :])
        #         p2 = cls.get2d(pose3[bone[jj][1], :])
        #         plt.plot(
        #             (int(p0[0]), int(p0[1])), (int(p2[0]), int(p2[1])),
        #             color=[i / 255 for i in cls.join_color[fii + 1]],
        #             linewidth=2.0
        #         )
        #         # cv2.line(img,
        #         #          (int(p0[0]), int(p0[1])),
        #         #          (int(p2[0]), int(p2[1])),
        #         #          cls.join_color[fii + 1], 1)
        return fig2data(plt.gcf())

    @staticmethod
    def parse_line(line_l, image_dir):
        image_name = line_l[0]
        img = mpimg.imread(os.path.join(image_dir, image_name))
        img = (img - img.min()) / (img.max() - img.min()) * 255
        # img = cv2.imread(os.path.join(image_dir, image_name), 0)
        # img[img == 0] = img.max()
        # img = img.astype(np.float32)
        # max16 = 2 ** 16 - 1
        # # img = (img - img.min()) / (img.max() - img.min()) * max16
        # img = (img - 100) / 3000 * max16
        # img = img.astype(np.uint16)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        pose3 = np.reshape(
            [float(i) for i in line_l[1:64]],
            (21, 3)
        )
        return img, pose3

    @staticmethod
    def get_line(filename, n):
        with open(filename, 'r') as f:
            for line_number, line in enumerate(f):
                if line_number == n:
                    return line

    @classmethod
    def draw_pose2_random(cls, train_dir):
        """ Draw 3D pose of a randomly picked image.
        """
        image_dir = os.path.join(train_dir, 'images')
        annot_txt = os.path.join(train_dir, 'Training_Annotation.txt')
        # img_id = randint(1, cls.num_training)
        img_id = randint(1, 99)
        print('drawing pose: # {}'.format(img_id))
        # Notice that linecache counts from 1
        # line = linecache.getline(annot_txt, img_id)
        line = linecache.getline(annot_txt, 97)
        # print(line)

        poser = re.split(r'\t+', line.rstrip('\t'))
        img, pose3 = cls.parse_line(poser, image_dir)

        cls.draw_pose2(img, pose3)
        plt.imshow(img, cmap='bone')
        plt.show()

    @classmethod
    def draw_pose2_stream(cls, train_dir, gif_file, max_draw=100):
        """ Draw 3D poses and streaming output as GIF file.
        """
        image_dir = os.path.join(train_dir, 'images')
        annot_txt = os.path.join(train_dir, 'Training_Annotation.txt')
        with imageio.get_writer(gif_file, mode='I', duration=0.2) as gif_writer:
            with open(annot_txt, 'r') as fa:
                csv_reader = csv.reader(fa, delimiter='\t')
                for lii, poser in enumerate(csv_reader):
                    if lii >= max_draw:
                        return
                        # raise coder.break_with.Break
                    img, pose3 = cls.parse_line(poser, image_dir)
                    plt.imshow(img, cmap='bone')
                    img = cls.draw_pose2(img, pose3)
                    # plt.show()
                    gif_writer.append_data(img)
                    plt.gcf().clear()


def show_depth(file_name):
    """ show a depth image """
    img = mpimg.imread(file_name)
    # img = (img - img.min()) / (img.max() - img.min()) * 255
    plt.imshow(img, cmap='gray')
    plt.show()


def test():
    # show_depth('/home/xwu/data/hands17/frame/images/image_D00084480.png')
    # image_dir = '/home/xwu/data/hands17/frame/images'
    # os.system('convert -delay 20 -resize 320x240 -loop 0 {0}/*.png {0}/loop.gif'.format(image_dir))

    # pose_hands17.draw_pose2_stream(
    #     '/home/xwu/data/hands17/training',
    #     '/home/xwu/data/hands17/training/pose.gif',
    #     10
    # )
    # show_depth('/home/xwu/data/hands17/training/pose.gif')
    pose_hands17.draw_pose2_random(
        '/home/xwu/data/hands17/training'
    )


if __name__ == '__main__':
    test()
