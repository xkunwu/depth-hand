#!/bin/bash

set -e

cd /tmp/$1/

rm -rf data/
rm -rf docker/
rm -rf scripts/
rm -rf eval/figures/
rm -rf eval/line-count.txt
rm -f .gitignore
rm -f HEADER
rm -f code/playground.ipynb
rm -f code/prereq
rm -rf code/scripts*
rm -rf code/test*
rm -f code/camera/anim_test.py
rm -rf code/data/nyu_hand/
mv code/train/train_abc.py ./train_abc.py
rm -f code/train/train_*
mv ./train_abc.py code/train/train_abc.py
rm -f code/model/affine_net.py
rm -f code/model/super_edt3.py
rm -f code/model/base_clean_hg.py
rm -f code/model/base_regre_hg.py
rm -f code/model/hourglass.py
rm -f code/model/base_inres.py
rm -f code/model/base_conv3_inres.py
rm -f code/model/localizer*

find . -regextype posix-extended -regex ".*\.py($|\..*)" -type f -exec sed -i '3i\    Cleaned and handover to Camera@UoB, 21 December 2018' {} \;

sed -i '/^The code in this repo.*/s//The code in this repo was written by Xiaokun when he was a postdoc at the CAMERA group of Uni Bath./' README.md

cd /tmp
tar -cf $HOME/Downloads/$1.tar $1/
