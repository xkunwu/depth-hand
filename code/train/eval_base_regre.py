import os
import sys
import matplotlib.pyplot as mpplot
from hands17 import hands17
from base_regre import base_regre
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
sys.path.append(BASE_DIR)
from args_holder import args_holder


argsholder = args_holder()
argsholder.parse_args()
argsholder.args.batch_size = 16

trainer = base_regre(argsholder.args, False)
if (not os.path.exists(hands17.training_annot_prediction)):
    trainer.evaluate()

hands17.draw_pred_random(
    hands17.training_images,
    hands17.training_annot_evaluation,
    hands17.training_annot_prediction
)

errors = hands17.compare_error(
    hands17.training_annot_evaluation,
    hands17.training_annot_prediction
)
draw_sum = 3
draw_i = 1
fig_size = (draw_sum * 5, 5)
mpplot.subplots(nrows=1, ncols=draw_sum, figsize=fig_size)
mpplot.subplot(1, draw_sum, draw_i)
draw_i += 1
hands17.draw_error_percentage_curve(errors, mpplot.gca())
mpplot.subplot(1, draw_sum, draw_i)
hands17.draw_error_per_joint(errors, mpplot.gca())
draw_i += 1
mpplot.subplot(1, draw_sum, draw_i)
hands17.draw_mean_error_distribution(errors, mpplot.gca())
draw_i += 1
mpplot.show()
