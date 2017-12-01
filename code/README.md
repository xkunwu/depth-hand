# install requirements
conda install --file requirements.txt
pip install pyquaternion
pip install colour
pip install pycparser
pip install cython
pip install pyrealsense
conda install --channel menpo opencv
conda install --channel menpo mayavi
conda install pyside
pip install mayavi

# remote management
tensorboard --logdir log
jupyter-notebook --no-browser --port=8888
ssh ${1:-sipadan} -L localhost:${2:-1}6006:localhost:6006 -L localhost:${2:-1}8888:localhost:8888

# get hardware info
cat /proc/meminfo
hwinfo --short
