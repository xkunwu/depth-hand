import os
from .incept_resnet import incept_resnet
from .base_clean import base_clean
from .base_regre import base_regre


class base_regre_inres(base_regre):
    def __init__(self, args):
        super(base_regre_inres, self).__init__(args)
        self.appen_train = os.path.join(
            self.prepare_dir,
            'train_{}'.format(self.__class__.__base__.__name__))
        self.appen_test = os.path.join(
            self.prepare_dir,
            'test_{}'.format(self.__class__.__base__.__name__))

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
    def __init__(self, args):
        super(base_clean_inres, self).__init__(args)
        self.appen_train = os.path.join(
            self.prepare_dir,
            'train_{}'.format(self.__class__.__base__.__name__))
        self.appen_test = os.path.join(
            self.prepare_dir,
            'test_{}'.format(self.__class__.__base__.__name__))

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
