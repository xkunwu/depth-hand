#!/bin/bash

# default user should be you
# USER=$(whoami)
USER=${USER:-root}
if [ "$USER" == "root" ]; then
    HOME=/root
fi

umask 0002

# start all the services
/startup/jupyter_start.sh

