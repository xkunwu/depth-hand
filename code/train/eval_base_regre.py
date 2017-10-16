import os
import sys
import importlib
import matplotlib.pyplot as mpplot
from train_abc import train_abc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
sys.path.append(BASE_DIR)
args_holder_class = getattr(
    importlib.import_module('args_holder'),
    'args_holder'
)
argsholder = args_holder_class()
argsholder.parse_args()
ARGS = argsholder.args
ARGS.batch_size = 16
ARGS.max_epoch = 1
ARGS.data_class = getattr(
    importlib.import_module(ARGS.data_name),
    ARGS.data_name
)
ARGS.model_class = getattr(
    importlib.import_module(ARGS.model_name),
    ARGS.model_name
)

trainer = train_abc(False)
if (not os.path.exists(ARGS.data_class.training_annot_prediction)):
    trainer.evaluate()

ARGS.data_class.draw_pred_random(
    ARGS.data_class.training_images,
    ARGS.data_class.training_annot_evaluation,
    ARGS.data_class.training_annot_prediction
)

errors = ARGS.data_class.compare_error(
    ARGS.data_class.training_annot_evaluation,
    ARGS.data_class.training_annot_prediction
)
draw_sum = 3
draw_i = 1
fig_size = (draw_sum * 5, 5)
mpplot.subplots(nrows=1, ncols=draw_sum, figsize=fig_size)
mpplot.subplot(1, draw_sum, draw_i)
draw_i += 1
ARGS.data_class.draw_error_percentage_curve(errors, mpplot.gca())
mpplot.subplot(1, draw_sum, draw_i)
ARGS.data_class.draw_error_per_joint(errors, mpplot.gca())
draw_i += 1
mpplot.subplot(1, draw_sum, draw_i)
ARGS.data_class.draw_mean_error_distribution(errors, mpplot.gca())
draw_i += 1
mpplot.show()
