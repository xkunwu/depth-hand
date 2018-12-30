#!/bin/bash
PROJ_NAME=${PWD##*/}
PROJ_DIR=projects
OUT_DIR=data/univue
SERVER=${1:-palau}
DATA_NAME=${2:-hands17}
MODEL=${3:-base_clean}

## upload code
SOURCE=${PWD}/
TARGET=${SERVER}:${PROJ_DIR}/${PROJ_NAME}/
printf "uploading ...\n"
printf "  from: [${SOURCE}]\n"
printf "  to: [${TARGET}]\n"
rsync -auvh -e ssh \
    --exclude-from='.gitignore' \
    ${SOURCE} \
    ${TARGET}
printf "rsync -auvh -e ssh --exclude-from='.gitignore' --delete ${SOURCE} ${TARGET}\n"

## run download script
printf "\n\n"
printf "downloading (dry-run) ...\n"
printf "  from: [${TARGET}] \n"
printf "  to: [${SOURCE}]\n"
rsync -auvhn -e ssh \
    --exclude-from='.gitignore' \
    --exclude-from='.git' \
    ${TARGET} \
    ${SOURCE}
#printf "rsync -auvh -e ssh --exclude-from='.gitignore' --exclude='.git' ${TARGET} ${SOURCE}\n"
printf "\n\n"
read -p "download? y/[n] >" -n 1 -r
printf "\n"
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rsync -auvh -e ssh \
        --exclude-from='.gitignore' \
        --exclude-from='.git' \
        ${TARGET} \
        ${SOURCE}
fi

## download predictions
SOURCE=${SERVER}:${OUT_DIR}/output/${DATA_NAME}/predict/
TARGET=${HOME}/${OUT_DIR}/${SERVER}/${DATA_NAME}/predict/
printf "\n\n"
printf "downloading ...\n"
printf "  from: [${SOURCE}]\n"
printf "  to: [${TARGET}]\n"
read -p "download predictions? y/[n] >" -n 1 -r
printf "\n"
if [[ $REPLY =~ ^[Yy]$ ]]; then
    mkdir -p ${TARGET}
    rsync -auvh -e ssh \
        ${SOURCE} \
        ${TARGET}
fi

## download full log (including checkpoint)
SOURCE=${SERVER}:${OUT_DIR}/output/${DATA_NAME}/log/blinks/${MODEL}/
TARGET=${HOME}/${OUT_DIR}/${SERVER}/${DATA_NAME}/log/${MODEL}
printf "\n\n"
printf "downloading ...\n"
printf "  from: [${SOURCE}]\n"
printf "  to: [${TARGET}]\n"
read -p "download full log? y/[n] >" -n 1 -r
printf "\n"
if [[ $REPLY =~ ^[Yy]$ ]]; then
    mkdir -p ${TARGET}
    rsync -auvhm -e ssh \
        --include='*.txt' --include='*.log' \
        --include='*.png' --include='*/' \
        --include='model.ckpt*' \
        --exclude '*' \
        ${SOURCE} \
        ${TARGET}
fi

# ## download model checkpoint
# SOURCE=${SERVER}:${OUT_DIR}/output/${DATA_NAME}/log/blinks/${MODEL}/model.ckpt*
# TARGET=${HOME}/${OUT_DIR}/${SERVER}/${DATA_NAME}/log/${MODEL}
# mkdir -p ${TARGET}
# echo downloading \
#     from: [${SOURCE}] \
#     to: [${TARGET}]
# rsync -auvh -e ssh \
#     ${SOURCE} \
#     ${TARGET}
