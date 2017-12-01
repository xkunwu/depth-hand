import os
from .incept_resnet import incept_resnet
from .base_clean import base_clean
from .base_regre import base_regre


class base_regre_inres(base_regre):
    """ This class use cleaned data from 3D PCA bounding cube.
    """
    def __init__(self):
        super(base_regre_inres, self).__init__()

    def receive_data(self, thedata, args):
        """ Receive parameters specific to the data """
        self.prepare_dir = args.prepare_dir
        self.appen_train = os.path.join(
            self.prepare_dir, 'train_{}'.format(self.__class__.__base__.__name__))
        self.appen_test = os.path.join(
            self.prepare_dir, 'test_{}'.format(self.__class__.__base__.__name__))
        self.predict_dir = args.predict_dir
        self.predict_file = os.path.join(
            self.predict_dir, 'predict_{}'.format(self.__class__.__name__))
        self.batch_size = args.batch_size
        self.out_dim = thedata.join_num * 3
        self.image_dir = thedata.training_images
        self.caminfo = thedata
        self.provider = args.data_provider
        # self.provider_worker = args.data_provider.prow_cleaned
        # self.yanker = self.provider.yank_cleaned
        self.provider_worker = self.provider.prow_cropped
        self.yanker = self.provider.yank_cropped

    def get_model(
            self, input_tensor, is_training,
            scope=None, final_endpoint='stage_out'):
        """ input_tensor: BxHxWxC
            out_dim: BxJ, where J is flattened 3D locations
        """
        self.end_point_list = []
        return incept_resnet.get_net(
            input_tensor, self.out_dim, is_training, self.end_point_list
        )


class base_clean_inres(base_clean):
    """ This class use cleaned data from 3D PCA bounding cube.
    """
    def __init__(self):
        super(base_clean_inres, self).__init__()

    def receive_data(self, thedata, args):
        """ Receive parameters specific to the data """
        self.prepare_dir = args.prepare_dir
        self.appen_train = os.path.join(
            self.prepare_dir, 'train_{}'.format(self.__class__.__base__.__name__))
        self.appen_test = os.path.join(
            self.prepare_dir, 'test_{}'.format(self.__class__.__base__.__name__))
        self.predict_dir = args.predict_dir
        self.predict_file = os.path.join(
            self.predict_dir, 'predict_{}'.format(self.__class__.__name__))
        self.batch_size = args.batch_size
        self.out_dim = thedata.join_num * 3
        self.image_dir = thedata.training_images
        self.caminfo = thedata
        self.provider = args.data_provider
        self.provider_worker = args.data_provider.prow_cleaned
        self.yanker = self.provider.yank_cleaned

    def get_model(
            self, input_tensor, is_training,
            scope=None, final_endpoint='stage_out'):
        """ input_tensor: BxHxWxC
            out_dim: BxJ, where J is flattened 3D locations
        """
        self.end_point_list = []
        return incept_resnet.get_net(
            input_tensor, self.out_dim, is_training, self.end_point_list
        )
