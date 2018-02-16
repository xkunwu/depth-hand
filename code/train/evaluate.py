import os
import sys
from importlib import import_module
from shutil import copyfile
import numpy as np
import re


def run_one(args, with_train=False, with_eval=False):
    predict_file = args.model_inst.predict_file
    trainer = args.model_inst.get_trainer(args, new_log=False)
    if with_train or (not os.path.exists(os.path.join(
            args.log_dir_t, 'model.ckpt.meta'))):
        trainer.train()
    if with_eval or (not os.path.exists(predict_file)):
        trainer.evaluate()
    # trainer.evaluate()


def draw_compare(args, predict_dir=None):
    mpplot = import_module('matplotlib.pyplot')
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
        error = dataeval.compare_error(
            args.data_inst,
            annot_test,
            predict
        )
        # print(error.shape)
        error_l.append(error)
    errors = np.stack(error_l, axis=0)
    fig = mpplot.figure(figsize=(2 * 5, 1 * 5))
    dataeval.draw_error_percentage_curve(
        errors, methods, mpplot.gca())
    mpplot.savefig(os.path.join(predict_dir, 'error_rate.png'))
    mpplot.close(fig)
    # mpplot.gcf().clear()
    fig = mpplot.figure(figsize=(2 * 5, 1 * 5))
    err_mean = dataeval.draw_error_per_joint(
        errors, methods, mpplot.gca(), args.data_inst.join_name)
    mpplot.savefig(os.path.join(predict_dir, 'error_bar.png'))
    mpplot.close(fig)

    maxmean = np.max(np.mean(errors, axis=1), axis=1)
    idx = np.argsort(maxmean)
    restr = 'maximal per-joint mean error summary:'
    for ii in idx:
        restr += ' {}({:.2f})'.format(methods[ii], maxmean[ii])
    args.logger.info(restr)
    idx = np.argsort(err_mean)
    restr = 'mean error summary:'
    for ii in idx:
        restr += ' {}({:.2f})'.format(methods[ii], err_mean[ii])
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
    # python -m train.evaluate --max_epoch=1 --batch_size=10 --bn_decay=0.94 --model_name=base_clean
    # import pdb; pdb.set_trace()

    from args_holder import args_holder
    # import tfplot
    # with_train = True
    with_train = False
    with_eval = True
    # with_eval = False

    # mpl = import_module('matplotlib')
    # mpl.use('Agg')
    with args_holder() as argsholder:
        argsholder.parse_args()
        args = argsholder.args
        argsholder.create_instance()
        # import shutil
        # shutil.rmtree(args.out_dir)
        # os.makedirs(args.out_dir)

        test_dataops(args)

        run_one(args, with_train, with_eval)
        argsholder.append_log()

        # draw_compare(args)
    sys.exit()

    mpl = import_module('matplotlib')
    mpl.use('Agg')
    methlist = [
        # # 'localizer2',
        # 'voxel_regre',
        # 'voxel_detect',
        'dense_regre',
        # 'direc_tsdf',
        # 'trunc_dist',
        # 'base_conv3',
        # 'base_conv3_inres',
        # 'ortho3view',
        # 'base_regre',
        # 'base_clean',
        # 'base_regre_inres',
        # 'base_clean_inres',
        # 'base_regre_hg',
        # 'base_clean_hg',
    ]
    for meth in methlist:
        with args_holder() as argsholder:
            argsholder.parse_args()
            args = argsholder.args
            args.model_name = meth
            argsholder.create_instance()
            test_dataops(args)
            run_one(args, with_train, with_eval)
            # run_one(args, True, True)
            # run_one(args, False, False)
            argsholder.append_log()
    with args_holder() as argsholder:
        argsholder.parse_args()
        argsholder.create_instance()
        args = argsholder.args
        draw_compare(args)
        # args.model_inst.detect_write_images()
    copyfile(
        os.path.join(args.log_dir, 'univue.log'),
        os.path.join(args.predict_dir, 'univue.log')
    )
