from base_conv3 import base_conv3


class trunc_dist(base_conv3):
    """ This class holds baseline training approach using 3d CNN.
    """
    def __init__(self):
        super(trunc_dist, self).__init__()

    def receive_data(self, thedata, args):
        """ Receive parameters specific to the data """
        super(trunc_dist, self).receive_data(thedata, args)
        self.provider_worker = args.data_provider.prow_truncdf
        self.yanker = self.provider.yank_truncdf
