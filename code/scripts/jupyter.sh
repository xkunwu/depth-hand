#!/bin/bash

jupyter-start()
{
    nohup jupyter-lab \
        --ip=127.0.0.1 \
        --port=8888 \
        --no-browser \
        $(pwd) >./jupyter.out 2>&1 &
}

jupyter-list()
{
	jupyter-notebook list | sed -n "s/^.*token=\(\S\+\).*$/\1/p"
}

jupyter-kill()
{
    proc_id=$(ps aux | grep "[j]upyter-lab" | tee /dev/tty | awk '{print $2}')
    if [ -z "$proc_id" ]; then
        exit 0
    fi
    kill -9 $proc_id
    printf "processes: [$proc_id] killed\n"
}
