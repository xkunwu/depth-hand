#!/bin/bash

#jupyter-start()
#{
#    nohup jupyter-lab \
#        --ip=127.0.0.1 \
#        --port=8888 \
#        --no-browser \
#        $(pwd) >./jupyter.out 2>&1 &
#}
#
#jupyter-list()
#{
#	jupyter-notebook list | sed -n "s/^.*token=\(\S\+\).*$/\1/p"
#}
#
#jupyter-kill()
#{
#    proc_id=$(ps aux | grep "[j]upyter-lab" | tee /dev/tty | awk '{print $2}')
#    if [ -z "$proc_id" ]; then
#        exit 0
#    fi
#    kill -9 $proc_id
#    printf "processes: [$proc_id] killed\n"
#}
source ./scripts/jupyter.sh

# Check if the function exists (bash specific)
if declare -f "$1" > /dev/null
then
  # call arguments verbatim
  "$@"
else
  # Show a helpful error
  echo "'$1' is not a known function name" >&2
  exit 1
fi
