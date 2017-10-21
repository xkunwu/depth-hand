import numpy as np
from skimage import io as skimio
import re


def read_image(image_name):
    # img = mpimg.imread(image_name)
    # img = spndim.imread(image_name)
    # img = (img - img.min()) / (img.max() - img.min()) * 255
    img = skimio.imread(image_name)
    return img


def save_image(image_name, img):
    # mpimg.imsave(image_name, img, cmap=mpplot.cm.gray)
    # spmisc.imsave(image_name, img)
    skimio.imsave(image_name, img)


def parse_line_pose(annot_line):
    line_l = re.split(r'\s+', annot_line.strip())
    rescen = None
    if 64 == len(line_l):
        pose_mat = np.reshape(
            [float(i) for i in line_l[1:64]],
            (21, 3)
        )
    elif 67 == len(line_l):
        pose_mat = np.reshape(
            [float(i) for i in line_l[1:64]],
            (21, 3)
        )
        rescen = np.array([float(i) for i in line_l[64:67]])
    else:
        print('error - wrong pose annotation: {}'.format(line_l))
    return line_l[0], pose_mat, rescen


def get_line(filename, n):
    with open(filename, 'r') as f:
        for line_number, line in enumerate(f):
            if line_number == n:
                return line


def parse_line_bbox(annot_line):
    line_l = re.split(r'\s+', annot_line.strip())
    bbox = np.reshape(
        [float(i) for i in line_l[1:5]],
        (2, 2)
    )
    return line_l[0], bbox
