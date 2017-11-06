import os
from importlib import import_module
import argparse
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
            '--rebuild_data', default=False,
            help='rebuild data structure and preprocess [default: False]')
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
        # [base_regre, base_clean, ortho3view, base_conv3, trunc_dist]
        self.parser.add_argument(
            # '--model_name', default='ortho3view',
            '--model_name', default='base_clean',
            help='Model name [default: base_clean], from \
            [base_regre, base_clean, ortho3view, base_conv3, trunc_dist]'
        )

        # input preprocessing
        # self.parser.add_argument(
        #     '--crop_size', type=int, default=96,
        #     help='Resized image size of cropped area [default: 96]')
        # self.parser.add_argument(
        #     '--pose_dim', type=int, default=63,
        #     help='Output tensor length of pose [default: 63]')

        # learning parameters
        self.parser.add_argument(
            '--max_epoch', type=int, default=2,
            help='Epoch to run [default: 10]')
        self.parser.add_argument(
            '--batch_size', type=int, default=128,
            help='Batch size during training [default: 64]')
        self.parser.add_argument(
            '--store_level', type=int, default=16,
            help='Batch size for storage [default: 16, means 2<<16 frames]')
        # self.parser.add_argument(
        #     '--optimizer', default='adam',
        #     help='Only using adam currently [default: adam]')
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
        self.args.store_level = (2 << self.args.store_level)

    def create_instance(self):
        self.args.model_class = getattr(
            import_module('train.' + self.args.model_name),
            self.args.model_name
        )
        self.args.model_inst = self.args.model_class(self.args.out_dir)
        self.args.model_inst.tweak_args(self.args)
        self.args.data_module = import_module(
            'data.' + self.args.data_name)
        self.args.data_provider = import_module(
            'data.' + self.args.data_name + '.provider')
        self.args.data_draw = import_module(
            'data.' + self.args.data_name + '.draw')
        self.args.data_ops = import_module(
            'data.' + self.args.data_name + '.ops')
        self.args.data_io = import_module(
            'data.' + self.args.data_name + '.io')
        self.args.data_class = getattr(
            import_module('data.' + self.args.data_name + '.holder'),
            self.args.data_name + 'holder'
        )
        self.args.data_inst = self.args.data_class(self.args)
        self.args.data_inst.init_data(
            self.args.rebuild_data
        )
        self.args.model_inst.receive_data(self.args.data_inst, self.args)

    def write_args(self, log_dir):
        with open(os.path.join(log_dir, 'args.txt'), 'w') as writer:
            for arg in vars(self.args):
                writer.write('--{}={}'.format(arg, getattr(self.args, arg)))
                # print(arg, getattr(self.args, arg))


if __name__ == "__main__":
    # python args_holder.py --batch_size=16 --store_level=6
    argsholder = args_holder()
    argsholder.parse_args()
    ARGS = argsholder.args
    ARGS.rebuild_data = True  # this is very slow
    argsholder.create_instance()
