#!/bin/bash

# --ip=0.0.0.0 \

nohup jupyter-lab \
    --ip=127.0.0.1 \
    --port=8888 \
    --no-browser \
    /workspace >$HOME/jupyter.out 2>&1 &

sleep 1
/startup/jupyter_list.sh
