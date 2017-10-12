import argparse
import os
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
            '--out_dir', default='output',
            help='Output dir [default: output]')
        self.parser.add_argument(
            '--log_file', default='univue.log',
            help='Log file name [default: univue.log]')
        self.parser.add_argument(
            '--model_ckpt', default='model.ckpt',
            help='Model checkpoint name [default: model.ckpt]')

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
            '--pose_dim', type=int, default=63,
            help='Output tensor length of pose [default: 63]')

        # learning parameters
        # self.parser.add_argument(
        #     '--feature_length', type=int, default=2048,
        #     help='network output feature length [default: 2048]')
        self.parser.add_argument(
            '--max_epoch', type=int, default=100,
            help='Epoch to run [default: 100]')
        self.parser.add_argument(
            '--batch_size', type=int, default=64,
            help='Batch Size during training [default: 64]')
        self.parser.add_argument(
            '--optimizer', default='adam',
            help='Only using adam currently [default: adam]')
        self.parser.add_argument(
            '--bn_momentum', type=float, default=0.8,
            help='Initial batch normalization momentum [default: 0.8]')
        self.parser.add_argument(
            '--learning_rate', type=float, default=0.001,
            help='Initial learning rate [default: 0.001]')
        self.parser.add_argument(
            '--decay_step', type=int, default=200000,
            # twice of 1M dataset
            help='Decay step for lr decay [default: 200000]')
        self.parser.add_argument(
            '--decay_rate', type=float, default=0.9,
            # fast decay, as using adaptive optimizer
            help='Decay rate for lr decay [default: 0.9]')

    def parse_args(self):
        self.args = self.parser.parse_args()
        self.args.data_dir = os.path.join(self.args.data_root, self.args.data_name)
        this_dir = os.path.dirname(os.path.abspath(__file__))
        self.args.proj_root = os.path.abspath(os.path.join(this_dir, os.pardir))
        self.args.out_dir = os.path.join(self.args.proj_root, self.args.out_dir)
        if not os.path.exists(self.args.out_dir):
            os.makedirs(self.args.out_dir)
        self.args.out_dir = os.path.join(self.args.out_dir, self.args.data_name)
        if not os.path.exists(self.args.out_dir):
            os.makedirs(self.args.out_dir)
        self.args.log_dir = os.path.join(self.args.out_dir, 'log')
        self.args.cpu_count = multiprocessing.cpu_count()
