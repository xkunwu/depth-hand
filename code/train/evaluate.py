import os
import sys
import progressbar
from importlib import import_module
from shutil import copyfile
import numpy as np
import re
from args_holder import model_ref


def run_one(args, with_train=False, with_eval=False):
    predict_file = args.model_inst.predict_file
    trainer = args.model_inst.get_trainer(args, new_log=False)
    if with_train or (not os.path.exists(os.path.join(
            args.log_dir_t, 'model.ckpt.meta'))):
        trainer.train()
    if with_eval or (not os.path.exists(predict_file)):
        trainer.evaluate()
    # trainer.evaluate()


def convert_legacy_txt(predictions):
    dataio = getattr(
        import_module('data.' + args.data_name + '.io'),
        'io')
    for predict in predictions:
        txt_back = predict + '.txt'
        os.rename(predict, txt_back)
        dataio.txt_to_h5(txt_back, predict)
        # dataio.h5_to_txt(predict + '.h5', predict + '_1.txt')


def draw_compare(args, predict_dir=None):
    mpplot = import_module('matplotlib.pyplot')
    dataeval = getattr(
        import_module('data.' + args.data_name + '.eval'),
        'eval')
    if predict_dir is None:
        predict_dir = args.model_inst.predict_dir
    predictions = []
    methods = []
    for file in os.listdir(predict_dir):
        m = re.match(r'^predict_(.+)', file)
        if m:
            predictions.append(os.path.join(predict_dir, file))
            methods.append(m.group(1))
    # convert_legacy_txt(predictions)  # convert legacy txt files
    # return
    num_method = len(methods)
    print('{:d} methods collected for comparison ...'.format(num_method))
    error_l = []
    timerbar = progressbar.ProgressBar(
        maxval=num_method,
        widgets=[
            progressbar.Percentage(),
            ' ', progressbar.Bar('=', '[', ']'),
            ' ', progressbar.ETA()]
    ).start()
    mi = 0
    for predict in predictions:
        error = dataeval.compare_error_h5(
            args.data_inst,
            args.data_inst.annotation_test,
            predict
        )
        # print(error.shape)
        error_l.append(error)
        mi += 1
        timerbar.update(mi)
    timerbar.finish()
    errors = np.stack(error_l, axis=0)
    args.logger.info('{:d} methods, {:d} test frames'.format(
        errors.shape[0], errors.shape[1]
    ))
    print('drawing figures ...')
    fig = mpplot.figure(figsize=(2 * 5, 1 * 5))
    meanmean = dataeval.draw_error_per_joint(
        errors, methods, mpplot.gca(),
        args.data_inst.join_name,
        [model_ref[m] for m in methods])
    fig.tight_layout()
    mpplot.savefig(os.path.join(predict_dir, 'error_bar.png'))
    mpplot.gcf().clear()
    maxmean = dataeval.draw_error_percentage_curve(
        errors, methods, mpplot.gca(),
        [model_ref[m] for m in methods])
    fig.tight_layout()
    mpplot.savefig(os.path.join(predict_dir, 'error_rate.png'))
    if args.show_draw:
        mpplot.show()
    mpplot.close(fig)

    # maxmean = np.max(np.mean(errors, axis=1), axis=1)

    idx_max = np.argsort(maxmean)
    idx_mean = np.argsort(meanmean)
    restr = 'mean error summary:'
    for ii in idx_mean:
        restr += ' {}({:.2f})'.format(methods[ii], meanmean[ii])
    args.logger.info(restr)
    restr = 'maximal per-joint mean error summary:'
    for ii in idx_max:
        restr += ' {}({:.2f})'.format(methods[ii], maxmean[ii])
    args.logger.info(restr)

    # restr = 'mean error summary:'
    # for ii in idx_mean:
    #     restr += ' {} & {:.2f} &'.format(
    #         model_ref[methods[ii]], meanmean[ii])
    # print(restr)
    # restr = 'maximal per-joint mean error summary:'
    # for ii in idx_max:
    #     restr += ' {} & {:.2f} &'.format(
    #         model_ref[methods[ii]], maxmean[ii])
    # print(restr)
    restr = 'error summary:'
    for ii in idx_mean:
        restr += ' {} & {:.2f} & {:.2f} \\\\'.format(
            model_ref[methods[ii]], meanmean[ii], maxmean[ii])
    print(restr)
    print('figures saved: error summary')

# python -m train.evaluate --max_epoch=1 --batch_size=5 --bn_decay=0.9 --show_draw=True --model_name=base_clean
# python -m train.evaluate --max_epoch=1 --batch_size=5 --model_name=super_edt2m
# python -m train.evaluate --out_root=${HOME}/data/univue/palau --max_epoch=1 --batch_size=5 --bn_decay=0.9 --show_draw=True --model_name=base_clean
# python -m train.evaluate --data_name=nyu_hand --num_eval=99 --max_epoch=1 --batch_size=5 --bn_decay=0.9 --show_draw=True --model_name=base_clean

# python -m train.evaluate --data_name=nyu_hand --max_epoch=20 --valid_stop=0.5 --bn_decay=0.995 --decay_step=100000
# python -m train.evaluate --gpu_id=1 --data_name=nyu_hand --out_root=${HOME}/data/out0219 --max_epoch=20 --valid_stop=0.5 --bn_decay=0.995 --decay_step=100000

# import pdb; pdb.set_trace()


if __name__ == "__main__":
    from args_holder import args_holder
    # with_train = True
    with_train = False
    # with_eval = True
    with_eval = False

    # mpl = import_module('matplotlib')
    # mpl.use('Agg')
    with args_holder() as argsholder:
        if not argsholder.parse_args():
            os._exit(0)
        args = argsholder.args
        if not argsholder.create_instance():
            os._exit(0)
        # import shutil
        # shutil.rmtree(args.out_dir)
        # os.makedirs(args.out_dir)

        run_one(args, with_train, with_eval)
        argsholder.append_log()

        # draw_compare(args)
    os._exit(0)

    mpl = import_module('matplotlib')
    mpl.use('Agg')
    methlist = [
        # 'super_edt3',
        # 'super_ov3edt2m',
        # 'super_ov3dist2',
        # 'super_ov3edt2',
        # 'super_edt2m',
        # 'super_edt2',
        # 'super_dist3',
        # 'voxel_regre',
        # 'voxel_offset',
        # 'super_vxhit',
        # 'voxel_detect',
        # 'super_dist2',
        # 'super_udir2',
        # 'super_hmap2',
        # 'dense_regre',
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
        # # 'localizer2',
    ]
    for meth in methlist:
        with args_holder() as argsholder:
            if not argsholder.parse_args():
                os._exit(0)
            args = argsholder.args
            args.model_name = meth
            if not argsholder.create_instance():
                os._exit(0)
            run_one(args, with_train, with_eval)
            # os._exit(0)
            # run_one(args, True, True)
            # run_one(args, False, False)
            args.model_inst.detect_write_images()
            argsholder.append_log()
    with args_holder() as argsholder:
        if not argsholder.parse_args():
            os._exit(0)
        if not argsholder.create_instance():
            os._exit(0)
        args = argsholder.args
        draw_compare(args)
        # argsholder.append_log()
        # args.model_inst.check_dir(args.data_inst, args)
        # args.model_inst.detect_write_images()
    # copyfile(
    #     os.path.join(args.log_dir, 'univue.log'),
    #     os.path.join(args.model_inst.predict_dir, 'univue.log')
    # )
