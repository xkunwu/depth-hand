conda install --file requirements.txt

python code/train/base_regre.py --batch_size=16 --max_epoch=10
python code/train/base_regre.py --batch_size=128 --max_epoch=100

tensorboard --logdir log
jupyter-notebook --no-browser --port=8888
ssh ${1:-sipadan} -L localhost:${2:-1}6006:localhost:6006 -L localhost:${2:-1}8888:localhost:8888
