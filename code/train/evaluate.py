import os
import sys
from importlib import import_module
import matplotlib.pyplot as mpplot
from train_abc import train_abc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
sys.path.append(BASE_DIR)
args_holder = getattr(
    import_module('args_holder'),
    'args_holder'
)


def run_one(args):
    argsholder.create_instance()
    data_inst = args.data_inst
    # args.model_inst.draw_random(data_inst, args)
    # datadraw = import_module(
    #     'data.' + args.data_name + '.draw')
    # datadraw.draw_raw3d_random(
    #     data_inst,
    #     data_inst.training_images,
    #     data_inst.training_annot_cleaned
    # )
    # sys.exit()

    trainer = train_abc(args, False)
    if (not os.path.exists(os.path.join(
            trainer.log_dir_t, args.model_ckpt + '.index'))):
        trainer.train()
    # trainer.train()
    if (not os.path.exists(args.model_inst.predict_file)):
        trainer.evaluate()
    # trainer.evaluate()

    print('evaluating {} ...'.format(args.model_name))
    datadraw = import_module(
        'data.' + args.data_name + '.draw')
    dataeval = import_module(
        'data.' + args.data_name + '.eval')

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

    datadraw.draw_pred_random(
        data_inst,
        data_inst.training_images,
        data_inst.training_annot_test,
        args.model_inst.predict_file
    )

    errors = dataeval.compare_error(
        data_inst,
        data_inst.training_annot_test,
        args.model_inst.predict_file
    )

    mpplot.figure()
    dataeval.draw_error_percentage_curve(errors, mpplot.gca())
    fname = '{}_error_rate.png'.format(args.model_name)
    mpplot.savefig(os.path.join(args.model_inst.predict_dir, fname))
    mpplot.figure()
    dataeval.draw_error_per_joint(errors, mpplot.gca(), data_inst.join_name)
    fname = '{}_error_bar.png'.format(args.model_name)
    mpplot.savefig(os.path.join(args.model_inst.predict_dir, fname))
    mpplot.figure()
    dataeval.draw_mean_error_distribution(errors, mpplot.gca())
    fname = '{}_error_dist.png'.format(args.model_name)
    mpplot.savefig(os.path.join(args.model_inst.predict_dir, fname))
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

if __name__ == "__main__":
    argsholder = args_holder()
    argsholder.parse_args()
    args = argsholder.args
    # args.rebuild_data = True  # this is very slow
    run_one(args)
    # sys.exit()
    methlist = [
        'base_regre',
        'base_clean',
        'ortho3view',
        'base_conv3',
        'trunc_dist'
    ]
    for meth in methlist:
        args.model_name = meth
        run_one(args)
