""" Hand in Depth
    https://github.com/xkunwu/depth-hand
"""
import re
from skimage import io as skimio
from data.io_abc import io_abc


class io(io_abc):
    @classmethod
    def read_image(cls, image_name):
        img = skimio.imread(image_name).astype(float)
        return (img[:, :, 1] * 256) + img[:, :, 2]

    @classmethod
    def imagename2index(cls, image_name):
        return int(re.match(r'^depth_1_(\d+)\.png', image_name).group(1))

    @classmethod
    def index2imagename(cls, index):
        return 'depth_1_{:07d}.png'.format(index)
