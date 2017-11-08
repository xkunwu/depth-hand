# import tensorflow as tf
import os
# import sys
# from importlib import import_module
# import numpy as np
# import h5py
from base_conv3 import base_conv3

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
# sys.path.append(BASE_DIR)
# sys.path.append(os.path.join(BASE_DIR, 'utils'))
# tf_util = import_module('tf_util')
# file_pack = getattr(
#     import_module('coder'),
#     'file_pack'
# )
# iso_cube = getattr(
#     import_module('iso_boxes'),
#     'iso_cube'
# )


class trunc_dist(base_conv3):
    """ This class holds baseline training approach using 3d CNN.
    """
    def __init__(self, out_dir):
        super(trunc_dist, self).__init__(out_dir)
        self.train_dir = os.path.join(out_dir, 'truncdf')

    def receive_data(self, thedata, args):
        """ Receive parameters specific to the data """
        self.pose_dim = thedata.join_num * 3
        self.image_dir = thedata.training_images
        self.caminfo = thedata
        self.provider = args.data_provider
        self.provider_worker = args.data_provider.prow_truncdf
        self.yanker = self.provider.yank_truncdf
        self.check_dir(thedata, args)
