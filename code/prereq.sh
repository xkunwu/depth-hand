#!/bin/bash

conda create -n hand python=3.6 numpy pip
conda install notebook jupyterlab
conda install -c pyviz holoviews bokeh
pip install -r requirements.txt
