import os
from .hourglass import hourglass
from .base_clean import base_clean
from .base_regre import base_regre


class base_regre_hg(base_regre):
    def __init__(self, args):
        super(base_regre_hg, self).__init__(args)
        self.appen_train = os.path.join(
            self.prepare_dir,
            'train_{}'.format(self.__class__.__base__.__name__))
        self.appen_test = os.path.join(
            self.prepare_dir,
            'test_{}'.format(self.__class__.__base__.__name__))

    def get_model(
            self, input_tensor, is_training,
            bn_decay, regu_scale,
            scope=None, final_endpoint='stage_out'):
        """ input_tensor: BxHxWxC
            out_dim: Bx(Jx3), where J is number of joints
        """
        self.end_point_list = []
        return hourglass.get_net(
            input_tensor, self.out_dim, is_training,
            bn_decay, regu_scale, self.end_point_list
        )


class base_clean_hg(base_clean):
    def __init__(self, args):
        super(base_clean_hg, self).__init__(args)
        self.appen_train = os.path.join(
            self.prepare_dir,
            'train_{}'.format(self.__class__.__base__.__name__))
        self.appen_test = os.path.join(
            self.prepare_dir,
            'test_{}'.format(self.__class__.__base__.__name__))

    def get_model(
            self, input_tensor, is_training,
            bn_decay, regu_scale,
            scope=None, final_endpoint='stage_out'):
        """ input_tensor: BxHxWxC
            out_dim: Bx(Jx3), where J is number of joints
        """
        self.end_point_list = []
        return hourglass.get_net(
            input_tensor, self.out_dim, is_training,
            bn_decay, regu_scale, self.end_point_list
        )
