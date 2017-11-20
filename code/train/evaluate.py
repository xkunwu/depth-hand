import os
import sys
from importlib import import_module
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


def run_one(args, with_train=False):
    import matplotlib as mpl
    mpl.use('Agg')
    mpplot = import_module('matplotlib.pyplot')

    argsholder.create_instance()
    data_inst = args.data_inst
    predict_file = os.path.join(
        data_inst.predict_dir, args.model_inst.predict_file)

    trainer = train_abc(args, False)
    if with_train or (not os.path.exists(os.path.join(
            args.log_dir_t, 'model.ckpt.meta'))):
        trainer.train()
    if with_train or (not os.path.exists(predict_file)):
        trainer.evaluate()

    print('evaluating {} ...'.format(args.model_name))

    datadraw = import_module(
        'data.' + args.data_name + '.draw')
    # mpplot.gcf().clear()
    mpplot.figure()
    datadraw.draw_pred_random(
        data_inst,
        data_inst.training_images,
        data_inst.training_annot_test,
        predict_file
    )
    fname = 'prediction_{}.png'.format(args.model_name)
    mpplot.savefig(os.path.join(args.data_inst.predict_dir, fname))

    dataeval = import_module(
        'data.' + args.data_name + '.eval')
    errors = dataeval.compare_error(
        data_inst,
        data_inst.training_annot_test,
        predict_file
    )
    mpplot.gcf().clear()
    dataeval.draw_error_percentage_curve(
        errors, [args.model_name], mpplot.gca())
    fname = '{}_error_rate.png'.format(args.model_name)
    mpplot.savefig(os.path.join(args.data_inst.predict_dir, fname))
    mpplot.gcf().clear()
    dataeval.draw_error_per_joint(
        errors, [args.model_name], mpplot.gca(), data_inst.join_name)
    fname = '{}_error_bar.png'.format(args.model_name)
    mpplot.savefig(os.path.join(args.data_inst.predict_dir, fname))
    mpplot.gcf().clear()
    dataeval.draw_mean_error_distribution(
        errors, mpplot.gca())
    fname = '{}_error_dist.png'.format(args.model_name)
    mpplot.savefig(os.path.join(args.data_inst.predict_dir, fname))
    args.logger.info('per-joint mean error maximum: {}'.format(
        np.max(np.mean(errors, axis=1))
    ))
    print('figures saved')

    # draw_sum = 3
    # draw_i = 1
    # fig_size = (draw_sum * 5, 5)
    # mpplot.subplots(nrows=1, ncols=draw_sum, figsize=fig_size)
    # mpplot.subplot(1, draw_sum, draw_i)
    # draw_i += 1
    # dataeval.draw_error_percentage_curve(errors, mpplot.gca())
    # mpplot.subplot(1, draw_sum, draw_i)
    # dataeval.draw_error_per_joint(errors, mpplot.gca(), data_inst.join_name)
    # draw_i += 1
    # mpplot.subplot(1, draw_sum, draw_i)
    # dataeval.draw_mean_error_distribution(errors, mpplot.gca())
    # draw_i += 1
    # mpplot.show()


def draw_compare(args, predict_dir=None):
    import matplotlib as mpl
    mpl.use('Agg')
    mpplot = import_module('matplotlib.pyplot')

    argsholder.create_instance()
    dataeval = import_module(
        'data.' + args.data_name + '.eval')
    if predict_dir is None:
        predict_dir = args.data_inst.predict_dir
    predictions = []
    methods = []
    for file in os.listdir(predict_dir):
        m = re.match(r'^predict_(.+)', file)
        if m:
            predictions.append(os.path.join(predict_dir, file))
            methods.append(m.group(1))
    annot_test = args.data_inst.training_annot_test
    error_l = None
    for predict in predictions:
        errors = dataeval.compare_error(
            args.data_inst,
            annot_test,
            predict
        )
        if error_l is None:
            error_l = errors
        else:
            error_l = np.concatenate((error_l, errors), axis=0)
    mpplot.gcf().clear()
    dataeval.draw_error_percentage_curve(
        error_l, methods, mpplot.gca())
    mpplot.savefig(os.path.join(args.data_inst.predict_dir, 'error_rate.png'))
    # mpplot.show()
    mpplot.gcf().clear()
    dataeval.draw_error_per_joint(
        error_l, methods, mpplot.gca(), args.data_inst.join_name)
    mpplot.savefig(os.path.join(args.data_inst.predict_dir, 'error_bar.png'))
    # mpplot.show()
    print('figures saved')


def test_dataops(args):
    argsholder.create_instance()
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
    with args_holder() as argsholder:
        argsholder.parse_args()
        args = argsholder.args

        # test_dataops(args)
        # sys.exit()

        # scp_t = os.path.join(args.out_dir, 'predict_sipadan')
        # draw_compare(args, scp_t)
        # sys.exit()

        # run_one(args)
        # sys.exit()

    methlist = [
        'direc_tsdf',
        'trunc_dist',
        'base_conv3',
        'ortho3view',
        'base_clean',
        'base_regre',
    ]
    for meth in methlist:
        with args_holder() as argsholder:
            argsholder.parse_args()
            args = argsholder.args
            args.model_name = meth
            run_one(args, True)
            # test_dataops(args)

    draw_compare(args)
