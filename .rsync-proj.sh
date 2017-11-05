#/bin/sh
#rsync -auvh -e ssh --exclude=.git --exclude=.rsync-proj.sh --exclude-from='.gitignore' ${PWD} ${1:-sipadan}:projects/${PWD##*/}
# rsync -auvh -e ssh --exclude=.git --exclude-from='.gitignore' ${PWD} ${1:-sipadan}:projects/${PWD##*/}
rsync -auvh -e ssh --exclude-from='.gitignore' ${PWD} ${1:-sipadan}:projects/
scp ${1:-sipadan}:projects/univue-hand-pose/output/hands17/predict/* ./output/hands17/predict_remote/
