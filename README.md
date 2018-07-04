docker build -t handpose .
nvidia-docker run -ti --rm \
    -v ${HOME}/projects/univue-hand-pose/code:/workspace/code:ro \
    -v ${HOME}/data:/data:ro \
    -v ${HOME}/data/univue/output:/output \
    -e "TERM=xterm-256color" \
    --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
    handpose
python -m train.evaluate \
    --data_root=/data \
    --out_root=/output \
    --max_epoch=1 --batch_size=5 --bn_decay=0.9 \
    --show_draw=True --model_name=base_clean
