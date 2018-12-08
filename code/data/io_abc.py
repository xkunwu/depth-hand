""" Hand in Depth
    https://github.com/xkunwu/depth-hand
"""
import numpy as np
from skimage import io as skimio
import h5py


class io_abc(object):
    @classmethod
    def read_image(cls, image_name):
        # img = mpimg.imread(image_name)
        # img = spndim.imread(image_name)
        # img = (img - img.min()) / (img.max() - img.min()) * 255
        img = skimio.imread(image_name)
        return img

    @classmethod
    def save_image(cls, image_name, img):
        # mpimg.imsave(image_name, img, cmap=mpplot.cm.gray)
        # spmisc.imsave(image_name, img)
        skimio.imsave(image_name, img)

    @classmethod
    def write_h5(cls, writer, index, poses):
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

    @classmethod
    def write_txt(cls, writer, index, poses):
        for ii, pp in zip(index, poses):
            writer.write(
                cls.index2imagename(ii) +
                '\t' + '\t'.join("%.4f" % x for x in pp.flatten()) +
                '\n')

    @classmethod
    def h5_to_txt(cls, h5_name, txt_name):
        with h5py.File(h5_name, 'r') as reader, \
                open(txt_name, 'w') as writer:
            cls.write_txt(writer, reader['index'], reader['poses'])

    @classmethod
    def txt_to_h5(cls, txt_name, h5_name):
        with open(txt_name, 'r') as reader, \
                h5py.File(h5_name, 'w') as writer:
            lines = [x.strip() for x in reader.readlines()]
            # num_line = len(lines)
            index = []
            poses = []
            for line in lines:
                name, pose = cls.parse_line_annot(line)
                index.append(cls.imagename2index(name))
                poses.append(pose.flatten())
            cls.write_h5(writer, np.array(index), np.vstack(poses))
