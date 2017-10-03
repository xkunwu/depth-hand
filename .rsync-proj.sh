#/bin/sh
#rsync -avh -e ssh --exclude=.git --exclude=.rsync-proj.sh --exclude-from='.gitignore' ./ ${1:-sipadan}:projects/${PWD##*/}
rsync -avh -e ssh --exclude=.git --exclude-from='.gitignore' ./ ${1:-sipadan}:projects/${PWD##*/}
