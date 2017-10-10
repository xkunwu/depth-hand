conda install --file requirements.txt

python code/train/base_regre.py --batch_size=16 --max_epoch=10

tensorboard --logdir log
ssh -L localhost:16006:localhost:6006 ${1:-sipadan}
ssh -L localhost:26006:localhost:6006 logan
