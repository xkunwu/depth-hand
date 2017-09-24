import argparse
import os
import sys


class args_holder:
    """ this class holds all arguments, and provides parsing functionality """
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # directories
        self.home_dir = os.path.expanduser('~')
        self.parser.add_argument(
            '--data_root', default=os.path.join(self.home_dir, '/data/hands17/'),
            help='root dir of data set [default: hands17]')
        self.parser.add_argument(
            '--log_dir', default='log',
            help='Log dir [default: log]')

        # system parameters
        self.parser.add_argument(
            '--gpu_id', type=int, default=0,
            help='GPU to use [default: GPU 0]')
        self.parser.add_argument(
            '--model', default='pointnet_cls', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')

        # learning parameters
        self.parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
        self.parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
        self.parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
        self.parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
        self.parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
        self.parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
        self.parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')

    def parse_args(self):
        self.args = self.parser.parse_args()
