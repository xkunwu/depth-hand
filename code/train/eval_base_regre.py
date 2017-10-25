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
argsholder = args_holder()
argsholder.parse_args()
ARGS = argsholder.args
ARGS.batch_size = 16
ARGS.max_epoch = 1
argsholder.create_instance()
data_inst = ARGS.data_inst

trainer = train_abc(ARGS, False)
if (not os.path.exists(data_inst.training_annot_predict)):
    trainer.evaluate()
# trainer.evaluate()

datadraw = import_module(
    'data.' + ARGS.data_name + '.draw')
dataeval = import_module(
    'data.' + ARGS.data_name + '.eval')

fig_size = (2 * 5, 5)
mpplot.subplots(nrows=1, ncols=2, figsize=fig_size)
mpplot.subplot(1, 2, 1)
datadraw.draw_pose_xy_random(
    data_inst,
    data_inst.training_images,
    data_inst.training_annot_cleaned,
    219  # palm
)
mpplot.subplot(1, 2, 2)
datadraw.draw_pose_xy_random(
    data_inst,
    data_inst.training_cropped,
    data_inst.training_annot_cropped,
    219  # palm
)
mpplot.show()

datadraw.draw_pred_random(
    data_inst,
    data_inst.training_images,
    data_inst.training_annot_test,
    data_inst.training_annot_predict
)

errors = dataeval.compare_error(
    data_inst,
    data_inst.training_annot_test,
    data_inst.training_annot_predict
)
draw_sum = 3
draw_i = 1
fig_size = (draw_sum * 5, 5)
mpplot.subplots(nrows=1, ncols=draw_sum, figsize=fig_size)
mpplot.subplot(1, draw_sum, draw_i)
draw_i += 1
dataeval.draw_error_percentage_curve(errors, mpplot.gca())
mpplot.subplot(1, draw_sum, draw_i)
dataeval.draw_error_per_joint(errors, mpplot.gca(), data_inst.join_name)
draw_i += 1
mpplot.subplot(1, draw_sum, draw_i)
dataeval.draw_mean_error_distribution(errors, mpplot.gca())
draw_i += 1
mpplot.show()
