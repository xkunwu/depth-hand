import os
import sys
from importlib import import_module
import argparse
import logging


class args_holder:
    """ this class holds all arguments, and provides parsing functionality """
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # directories
        home_dir = os.path.expanduser('~')
        this_dir = os.path.dirname(os.path.abspath(__file__))
        proj_root = os.path.abspath(os.path.join(this_dir, os.pardir))
        self.parser.add_argument(
            '--data_root', default=os.path.join(home_dir, 'data'),
            help='root dir of all data sets [default: data]')
        self.parser.add_argument(
            '--data_name', default='hands17',
            help='name of data set and its dir [default: hands17]')
        self.parser.add_argument(
            '--out_root', default=os.path.join(proj_root, 'output'),
            help='Output dir [default: output]')
        self.parser.add_argument(
            '--retrain', default=False,
            help='retrain the model [default: False]')
        # self.parser.add_argument(
        #     '--model_ckpt', default='model.ckpt',
        #     help='Model checkpoint name [default: model.ckpt]')

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

        # learning parameters
        self.parser.add_argument(
            '--max_epoch', type=int, default=2,
            help='Epoch to run [default: 10]')
        self.parser.add_argument(
            '--batch_size', type=int, default=128,
            help='Batch size during training [default: 64]')
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

    def make_new_log(self):
        log_dir = os.path.join(self.args.out_dir, 'log')
        blinks = os.path.join(log_dir, 'blinks')
        if not os.path.exists(blinks):
            os.makedirs(blinks)
        log_dir_ln = os.path.join(
            blinks, self.args.model_name)
        if (not os.path.exists(log_dir_ln)):
            from datetime import datetime
            log_time = datetime.now().strftime('%y%m%d-%H%M%S')
            # git_hash = subprocess.check_output(
            #     ['git', 'rev-parse', '--short', 'HEAD'])
            self.args.log_dir_t = os.path.join(
                log_dir, 'log-{}'.format(log_time)
            )
            os.makedirs(self.args.log_dir_t)
            os.symlink(self.args.log_dir_t, log_dir_ln + '-tmp')
            os.rename(log_dir_ln + '-tmp', log_dir_ln)
        else:
            self.args.log_dir_t = os.readlink(log_dir_ln)

    def make_logging(self):
        logFormatter = logging.Formatter(
            # '%(asctime)s [%(levelname)-5.5s]  %(message)s (%(filename)s:%(lineno)s)',
            '%(asctime)s [%(levelname)-5.5s]  %(message)s',
            datefmt='%y-%m-%d %H:%M:%S')
        logger = logging.getLogger('univue')
        logger.setLevel(logging.INFO)
        fileHandler = logging.FileHandler(
            os.path.join(
                self.args.out_dir, 'log', 'univue.log'),
            mode='a'
        )
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)
        consoleHandler = logging.StreamHandler(stream=sys.stdout)
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)
        self.args.logger = logger
        logger = logging.getLogger('train')
        logger.setLevel(logging.INFO)
        if self.args.retrain:
            fileHandler = logging.FileHandler(
                os.path.join(self.args.log_dir_t, 'train.log'),
                mode='w'
            )
        else:
            fileHandler = logging.FileHandler(
                os.path.join(self.args.log_dir_t, 'train.log'),
                mode='a'
            )
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)
        consoleHandler = logging.StreamHandler(stream=sys.stdout)
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)

    @staticmethod
    def write_args(args):
        import inspect
        with open(os.path.join(args.log_dir_t, 'args.txt'), 'w') as writer:
            for arg in vars(args):
                att = getattr(args, arg)
                if inspect.ismodule(att) or inspect.isclass(att):
                    continue
                writer.write('--{}={}\n'.format(arg, att))
                # print(arg, getattr(args, arg))

    def parse_args(self):
        self.args = self.parser.parse_args()
        self.args.data_dir = os.path.join(
            self.args.data_root,
            self.args.data_name)
        self.args.out_dir = os.path.join(
            self.args.out_root,
            self.args.data_name
        )
        self.args.prepare_dir = os.path.join(
            self.args.out_dir,
            'prepared'
        )
        if not os.path.exists(self.args.prepare_dir):
            os.mkdir(self.args.prepare_dir)
        self.args.predict_dir = os.path.join(
            self.args.out_dir,
            'predict'
        )
        if not os.path.exists(self.args.predict_dir):
            os.mkdir(self.args.predict_dir)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        logger = logging.getLogger('univue')
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        logger = logging.getLogger('train')
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        logging.shutdown()
        # if exc_type is not None:
        #     print(exc_type, exc_value, exc_traceback)
        return self

    def create_instance(self):
        self.make_new_log()
        if not os.path.exists(os.path.join(
                self.args.log_dir_t, 'model.ckpt.meta')):
            self.args.retrain = True
        self.make_logging()
        self.args.logger.info('######## {} [{}] ########'.format(
            self.args.data_name, self.args.model_name))
        self.args.model_class = getattr(
            import_module('train.' + self.args.model_name),
            self.args.model_name
        )
        self.args.model_inst = self.args.model_class()
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
        self.args.data_inst.init_data()
        self.args.model_inst.receive_data(self.args.data_inst, self.args)
        self.args.model_inst.check_dir(self.args.data_inst, self.args)
        self.write_args(self.args)


if __name__ == "__main__":
    # python args_holder.py --batch_size=16
    with args_holder() as argsholder:
        argsholder.parse_args()
        ARGS = argsholder.args
        argsholder.create_instance()
