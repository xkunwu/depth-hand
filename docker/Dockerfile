# Use Tensorflow from Tensorflow image as parent image
FROM tensorflow/tensorflow:1.12.0-gpu-py3

## essential packages
RUN apt-get update && apt-get install -y \
    software-properties-common sudo bash-completion apt-utils \
    vim tmux net-tools build-essential cmake git ffmpeg \
    libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

## add the default user
ARG USER_ME=me
ARG USER_UID=1000
ARG USER_GID=1000
COPY ./docker/image /
RUN \
    umask 0002 && \
    echo "create user $USER_ME ($USER_UID:$USER_GID)" && \
    # useradd --uid $USER_UID --user-group --create-home --shell /bin/bash --groups adm,sudo,root $USER_ME && \
    groupadd --gid $USER_GID $USER_ME && \
    useradd --uid $USER_UID --gid $USER_GID --create-home --shell /bin/bash --groups adm,sudo,root $USER_ME && \
    echo "$USER_ME:123123" | chpasswd && \
    \
    cp -t /home/$USER_ME /root/.vimrc /root/.tmux.conf && \
    mkdir -p /home/$USER_ME/.vnc && \
    cp -t /home/$USER_ME/.vnc /root/.vnc/xstartup && \
    chmod 755 /home/$USER_ME/.vnc/xstartup && \
    \
    mkdir -p /workspace && \
    chown -R $USER_UID:$USER_GID /workspace && \
    ln -s /workspace /root/projects && \
    ln -s /workspace /home/$USER_ME/projects && \
    chown -R $USER_UID:$USER_GID /home/$USER_ME && \
    \
    find /startup -name '*.sh' -exec chmod a+x {} + && \
    chown -R $USER_UID:$USER_GID /startup && \
    rm -f /startup.sh && \
    ln -s /startup/startup.sh /startup.sh
ENV \
    HOME=/home/$USER_ME USER=$USER_ME SHELL=/bin/bash

## jupyter tensorboard
EXPOSE 8888 6006

## user mode container
USER $USER_UID

CMD ["/startup.sh"]
