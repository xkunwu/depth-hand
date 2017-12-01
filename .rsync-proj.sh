#/bin/sh
#rsync -auvh -e ssh --exclude=.git --exclude=.rsync-proj.sh --exclude-from='.gitignore' ${PWD} ${1:-palau}:projects/${PWD##*/}
# rsync -auvh -e ssh --exclude=.git --exclude-from='.gitignore' ${PWD} ${1:-palau}:projects/${PWD##*/}
rsync -auvh -e ssh --exclude-from='.gitignore' ${PWD} ${1:-palau}:projects/
mkdir -p ./output/hands17/predict_${1:-palau}
# scp ${1:-palau}:projects/univue-hand-pose/output/hands17/predict/* ./output/hands17/predict_${1:-palau}/
rsync -auvh -e ssh ${1:-palau}:projects/univue-hand-pose/output/hands17/predict/* ./output/hands17/predict_${1:-palau}/
