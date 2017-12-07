import os
import sys
from importlib import import_module
from shutil import copyfile
import numpy as np
import re
from train_abc import train_abc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
sys.path.append(BASE_DIR)
args_holder = getattr(
    import_module('args_holder'),
    'args_holder'
)


def run_one(args, mpplot, with_train=False, with_eval=False):
    predict_file = args.model_inst.predict_file
    trainer = train_abc(args, False)
    if with_train or (not os.path.exists(os.path.join(
            args.log_dir_t, 'model.ckpt.meta'))):
        trainer.train()
    if with_eval or (not os.path.exists(predict_file)):
        trainer.evaluate()
    # trainer.evaluate()


def draw_compare(args, mpplot, predict_dir=None):
    dataeval = import_module(
        'data.' + args.data_name + '.eval')
    if predict_dir is None:
        predict_dir = args.predict_dir
    predictions = []
    methods = []
    for file in os.listdir(predict_dir):
        m = re.match(r'^predict_(.+)', file)
        if m:
            predictions.append(os.path.join(predict_dir, file))
            methods.append(m.group(1))
    annot_test = args.data_inst.training_annot_test
    error_l = []
    for predict in predictions:
        error_l.append(dataeval.compare_error(
            args.data_inst,
            annot_test,
            predict
        ))
    errors = np.stack(error_l, axis=0)
    mpplot.figure(figsize=(2 * 5, 1 * 5))
    dataeval.draw_error_percentage_curve(
        errors, methods, mpplot.gca())
    mpplot.savefig(os.path.join(predict_dir, 'error_rate.png'))
    mpplot.gcf().clear()
    dataeval.draw_error_per_joint(
        errors, methods, mpplot.gca(), args.data_inst.join_name)
    mpplot.savefig(os.path.join(predict_dir, 'error_bar.png'))

    maxmean = np.max(np.mean(errors, axis=1), axis=1)
    idx = np.argsort(maxmean)
    restr = 'maximal per-joint mean error summary:'
    for ii in idx:
        restr += ' {}({:.2f})'.format(methods[ii], maxmean[ii])
    args.logger.info(restr)
    print('figures saved: error summary')


def test_dataops(args):
    data_inst = args.data_inst

    args.model_inst.draw_random(data_inst, args)

    # datadraw = import_module(
    #     'data.' + args.data_name + '.draw')
    # datadraw.draw_raw3d_random(
    #     data_inst,
    #     data_inst.training_images,
    #     data_inst.training_annot_cleaned
    # )
    # sys.exit()

    # mpplot.gcf().clear()
    # fig_size = (2 * 5, 5)
    # mpplot.subplots(nrows=1, ncols=2, figsize=fig_size)
    # mpplot.subplot(1, 2, 1)
    # datadraw.draw_pose_raw_random(
    #     data_inst,
    #     data_inst.training_images,
    #     data_inst.training_annot_cleaned,
    #     219  # palm
    # )
    # mpplot.subplot(1, 2, 2)
    # datadraw.draw_pose_raw_random(
    #     data_inst,
    #     data_inst.training_cropped,
    #     data_inst.training_annot_cropped,
    #     219  # palm
    # )
    # mpplot.show()


if __name__ == "__main__":
    # python evaluate.py --max_epoch=1 --batch_size=16 --model_name=base_clean
    # import pdb; pdb.set_trace()

    # with_train = True
    with_train = False
    # with_eval = True
    with_eval = False

    # mpl = import_module('matplotlib')
    # mpl.use('Agg')
    # mpplot = import_module('matplotlib.pyplot')
    # with args_holder() as argsholder:
    #     argsholder.parse_args()
    #     args = argsholder.args
    #     argsholder.create_instance()
    #     # import shutil
    #     # shutil.rmtree(args.out_dir)
    #     # os.makedirs(args.out_dir)
    #
    #     test_dataops(args)
    #
    #     run_one(args, mpplot, with_train, with_eval)
    #
    #     # draw_compare(args, mpplot)
    # sys.exit()

    mpl = import_module('matplotlib')
    mpl.use('Agg')
    mpplot = import_module('matplotlib.pyplot')
    methlist = [
        # 'localizer2',
        # 'direc_tsdf',
        # 'trunc_dist',
        # 'base_conv3',
        # 'ortho3view',
        'base_regre',
        # 'base_clean',
        # 'base_regre_inres',
        # 'base_clean_inres',
    ]
    for meth in methlist:
        with args_holder() as argsholder:
            argsholder.parse_args()
            args = argsholder.args
            args.model_name = meth
            argsholder.create_instance()
            test_dataops(args)
            # run_one(args, mpplot, with_train, with_eval)
            # run_one(args, mpplot, True, True)
            run_one(args, mpplot, False, False)
    draw_compare(args, mpplot)
    copyfile(
        os.path.join(args.out_dir, 'log', 'univue.log'),
        os.path.join(args.predict_dir, 'univue.log')
    )
