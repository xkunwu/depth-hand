import numpy as np
from skimage import io as skimio
import re
import h5py


def read_image(image_name):
    img = skimio.imread(image_name).astype(float)
    return (img[:, :, 1] * 256) + img[:, :, 2]


def save_image(image_name, img):
    skimio.imsave(image_name, img)


def imagename2index(image_name):
    return int(re.match(r'^depth_1_(\d+)\.png', image_name).group(1))


def index2imagename(index):
    return 'depth_1_{:07d}.png'.format(index)


def write_h5(writer, index, poses):
    writer.create_dataset(
        'index',
        index.shape,
        compression='lzf',
        dtype='i4'
    )
    writer.create_dataset(
        'poses',
        poses.shape,
        compression='lzf',
        dtype='f4')
    writer['index'][:] = index
    writer['poses'][:] = poses


def write_txt(writer, index, poses):
    for ii, pp in zip(index, poses):
        writer.write(
            index2imagename(ii) +
            '\t' + '\t'.join("%.4f" % x for x in pp.flatten()) +
            '\n')


def h5_to_txt(h5_name, txt_name):
    with h5py.File(h5_name, 'r') as reader, \
            open(txt_name, 'w') as writer:
        write_txt(writer, reader['index'], reader['poses'])
