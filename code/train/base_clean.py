import os
from base_regre import base_regre


class base_clean(base_regre):
    """ This class use cleaned data from 3D PCA bounding box.
    """
    def __init__(self, out_dir):
        super(base_clean, self).__init__(out_dir)
        self.train_dir = os.path.join(out_dir, 'cleaned')

    def receive_data(self, thedata, args):
        """ Receive parameters specific to the data """
        self.pose_dim = thedata.join_num * 3
        self.image_dir = thedata.training_images
        self.provider = args.data_provider
        self.provider_worker = args.data_provider.prow_cleaned
        self.check_dir(thedata, args)
