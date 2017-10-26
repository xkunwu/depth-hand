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


def parse_line_pose(annot_line, append_line=None):
    """ parse raw annotation, and appendix for crop-resize """
    annot_list = re.split(r'\s+', annot_line.strip())
    if 64 == len(annot_list):
        pose_mat = np.reshape(
            [float(i) for i in annot_list[1:64]],
            (21, 3)
        )
    else:
        print('error - wrong pose annotation: {}'.format(annot_list))
    if append_line is None:
        rescen = np.array([1, 0, 0])
    else:
        append_list = re.split(r'\s+', annot_line.strip())
        rescen = np.array([float(i) for i in append_list])
    return annot_list[0], pose_mat, rescen


def get_line(filename, img_id):
    img_name_id = 'image_D{:08d}.png'.format(img_id)
    with open(filename, 'r') as f:
        for line_number, line in enumerate(f):
            img_name, _, _ = parse_line_pose(line)
            if img_name_id == img_name:
                return line


def parse_line_bbox(annot_line):
    line_l = re.split(r'\s+', annot_line.strip())
    bbox = np.reshape(
        [float(i) for i in line_l[1:5]],
        (2, 2)
    )
    return line_l[0], bbox
