#/bin/sh
rsync -avh -e ssh --exclude=.git --exclude=.rsync-proj.sh ./ ${1:-sipadan}:projects/${PWD##*/}
