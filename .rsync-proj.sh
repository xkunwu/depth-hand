#/bin/sh
OUT_DIR=projects/univue-hand-pose/output/hands17
SERVER=${1:-palau}
MODEL=${2:-base_regre}
## upload code
echo uploading \
    from: [${PWD}] \
    to: [${SERVER}:projects/]
rsync -auvh -e ssh \
    --exclude-from='.gitignore' \
    ${PWD} \
    ${SERVER}:projects/
## download predictions
mkdir -p ${HOME}/${OUT_DIR}/${SERVER}/predict
echo downloading \
    from: [${SERVER}:${OUT_DIR}/predict] \
    to: [${HOME}/${OUT_DIR}/${SERVER}/predict]
rsync -auvh -e ssh \
    ${SERVER}:${OUT_DIR}/predict/* \
    ${HOME}/${OUT_DIR}/${SERVER}/predict
## download model checkpoint
mkdir -p ${HOME}/${OUT_DIR}/${SERVER}/log/${MODEL}
echo downloading \
    from: [${SERVER}:${OUT_DIR}/log/blinks/${MODEL}] \
    to: [${HOME}/${OUT_DIR}/${SERVER}/log/${MODEL}]
rsync -auvh -e ssh \
    ${SERVER}:${OUT_DIR}/log/blinks/${MODEL}/* \
    ${HOME}/${OUT_DIR}/${SERVER}/log/${MODEL}
