import argparse
import os
from datetime import datetime
import multiprocessing


class args_holder:
    """ this class holds all arguments, and provides parsing functionality """
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # directories
        home_dir = os.path.expanduser('~')
        self.parser.add_argument(
            '--data_root', default=os.path.join(home_dir, 'data'),
            help='root dir of all data sets [default: data]')
        self.parser.add_argument(
            '--data_name', default='hands17',
            help='name of data set and its dir [default: hands17]')
        self.parser.add_argument(
            '--log_dir', default='log',
            help='Log dir [default: log]')
        self.parser.add_argument(
            '--log_file', default='univue.log',
            help='Log file name [default: univue.log]')

        # system parameters
        self.parser.add_argument(
            '--gpu_id', type=int, default=0,
            help='GPU to use [default: GPU 0]')
        self.parser.add_argument(
            '--model', default='base_regre',
            help='Model name [default: base_regre]')

        # input preprocessing
        self.parser.add_argument(
            '--img_size', type=int, default=96,
            help='Resized input image size [default: 96]')
        self.parser.add_argument(
            '--pose_dim', type=int, default=42,
            help='Output tensor length of pose [default: 42]')

        # learning parameters
        # self.parser.add_argument(
        #     '--feature_length', type=int, default=2048,
        #     help='network output feature length [default: 2048]')
        self.parser.add_argument(
            '--max_epoch', type=int, default=250,
            help='Epoch to run [default: 250]')
        self.parser.add_argument(
            '--batch_size', type=int, default=64,
            help='Batch Size during training [default: 64]')
        self.parser.add_argument(
            '--learning_rate', type=float, default=0.001,
            help='Initial learning rate [default: 0.001]')
        self.parser.add_argument(
            '--momentum', type=float, default=0.9,
            help='Initial learning rate [default: 0.9]')
        self.parser.add_argument(
            '--optimizer', default='adam',
            help='adam or momentum [default: adam]')
        self.parser.add_argument(
            '--decay_step', type=int, default=200000,
            help='Decay step for lr decay [default: 200000]')
        self.parser.add_argument(
            '--decay_rate', type=float, default=0.7,
            help='Decay rate for lr decay [default: 0.8]')

    def parse_args(self):
        self.args = self.parser.parse_args()
        self.args.data_dir = os.path.join(self.args.data_root, self.args.data_name)
        self.args.log_file = self.args.log_file + datetime.now().strftime('-%y-%m-%d-%H:%M:%S')
        self.cpu_count = multiprocessing.cpu_count()
        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)
