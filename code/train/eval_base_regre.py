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
# dataops = import_module(
#     'data.' + argsholder.args.data_name + '.ops')
# dataio = import_module(
#     'data.' + argsholder.args.data_name + '.io')
datadraw = import_module(
    'data.' + argsholder.args.data_name + '.draw')
dataeval = import_module(
    'data.' + argsholder.args.data_name + '.eval')
argsholder.args.batch_size = 16
argsholder.args.max_epoch = 1
data_inst = argsholder.args.data_inst

trainer = train_abc(argsholder.args, False)
if (not os.path.exists(data_inst.training_annot_predict)):
    trainer.evaluate()

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
dataeval.draw_error_per_joint(data_inst, errors, mpplot.gca())
draw_i += 1
mpplot.subplot(1, draw_sum, draw_i)
dataeval.draw_mean_error_distribution(errors, mpplot.gca())
draw_i += 1
mpplot.show()
