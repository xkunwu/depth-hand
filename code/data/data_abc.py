import os
import sys
from importlib import import_module

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
sys.path.append(BASE_DIR)
args_holder = getattr(
    import_module('args_holder'),
    'args_holder'
)

with args_holder() as argsholder:
    argsholder.parse_args()
    ARGS = argsholder.args
    argsholder.create_instance()
    data_name = ARGS.data_name
    data_inst = ARGS.data_inst

    datadraw = import_module(
        'data.' + ARGS.data_name + '.draw')
    datadraw.draw_raw3d_random(
        data_inst,
        data_inst.training_images,
        data_inst.training_annot_cleaned
    )
