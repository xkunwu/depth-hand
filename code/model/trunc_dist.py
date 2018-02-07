import os
import numpy as np
from model.base_conv3 import base_conv3


class trunc_dist(base_conv3):
    """ This class holds baseline training approach using 3d CNN.
    """
    def __init__(self, args):
        super(trunc_dist, self).__init__(args)

    def provider_worker(self, line, image_dir, caminfo):
        img_name, pose_raw = self.data_module.io.parse_line_annot(line)
        img = self.data_module.io.read_image(os.path.join(image_dir, img_name))
        pcnt, resce = self.data_module.ops.fill_grid(
            img, pose_raw, caminfo.crop_size, caminfo)
        tdf = self.data_module.ops.prop_dist(pcnt)
        resce3 = resce[0:4]
        pose_pca = self.data_module.ops.raw_to_pca(pose_raw, resce3)
        index = self.data_module.io.imagename2index(img_name)
        return (index, np.expand_dims(tdf, axis=-1),
                pose_pca.flatten().T, resce)

    def yanker(self, pose_local, resce, caminfo):
        resce3 = resce[0:4]
        return self.data_module.ops.pca_to_raw(pose_local, resce3)
