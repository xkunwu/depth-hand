#/bin/sh
#rsync -auvh -e ssh --exclude=.git --exclude=.rsync-proj.sh --exclude-from='.gitignore' ${PWD} ${1:-sipadan}:projects/${PWD##*/}
# rsync -auvh -e ssh --exclude=.git --exclude-from='.gitignore' ${PWD} ${1:-sipadan}:projects/${PWD##*/}
rsync -auvh -e ssh --exclude-from='.gitignore' ${PWD} ${1:-sipadan}:projects/
mkdir -p ./output/hands17/predict_${1:-sipadan}
# scp ${1:-sipadan}:projects/univue-hand-pose/output/hands17/predict/* ./output/hands17/predict_${1:-sipadan}/
rsync -auvh -e ssh ${1:-sipadan}:projects/univue-hand-pose/output/hands17/predict/* ./output/hands17/predict_${1:-sipadan}/
