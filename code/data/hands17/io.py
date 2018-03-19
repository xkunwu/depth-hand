import numpy as np
import re
from data.io_abc import io_abc


class io(io_abc):
    @classmethod
    def imagename2index(cls, image_name):
        return int(re.match(r'^image_D(\d+)\.png', image_name).group(1))

    @classmethod
    def index2imagename(cls, index):
        return 'image_D{:08d}.png'.format(index)

    @classmethod
    def parse_line_annot(cls, line):
        """ parse raw annotation, and appendix for crop-resize """
        annot_list = re.split(r'\s+', line.strip())
        if 64 == len(annot_list):
            pose_raw = np.reshape(
                [float(i) for i in annot_list[1:64]],
                (21, 3)
            )
        else:
            print('error - wrong pose annotation: {} --> {}'.format(
                line, annot_list))
            print(annot_list)
        return annot_list[0], pose_raw

    @classmethod
    def parse_line_appen2(cls, line):
        append_list = re.split(r'\s+', line.strip())
        resce = np.array([float(i) for i in append_list])
        return resce

    @classmethod
    def get_line(cls, filename, img_id):
        img_name_id = cls.index2imagename(img_id)
        with open(filename, 'r') as f:
            for _, line in enumerate(f):
                img_name, _ = cls.parse_line_annot(line)
                if img_name_id == img_name:
                    return line

    @classmethod
    def parse_line_bbox(cls, annot_line):
        line_l = re.split(r'\s+', annot_line.strip())
        bbox = np.reshape(
            [float(i) for i in line_l[1:5]],
            (2, 2)
        )
        return line_l[0], bbox
