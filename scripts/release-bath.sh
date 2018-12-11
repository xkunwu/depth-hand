#!/bin/bash

set -e

cd /tmp/$1/

rm -rf data/
rm -rf docker/
rm -rf scripts/
rm -rf eval/figures/
rm -f .gitignore
rm -f HEADER
rm -f code/playground.ipynb
rm -f code/prereq
rm -rf code/scripts*
rm -rf code/test*
rm -f code/camera/anim_test.py
rm -rf code/data/nyu_hand/
mv ./code/train/{train_abc.py,train_super_edt2.py} ./
rm -f code/train/train_*
mv ./{train_abc.py,train_super_edt2.py} ./code/train/
rm -f code/model/affine_net.py
rm -f code/model/hourglass.py
rm -f code/model/base_hourglass.py
rm -f code/model/base_clean_hg.py
rm -f code/model/base_regre_hg.py
rm -f code/model/base_inres.py
rm -f code/model/base_conv3.py
rm -f code/model/base_conv3_inres.py
rm -f code/model/localizer*
rm -f code/model/voxel_*
rm -f code/model/trunc_dist.py
rm -f code/model/ortho3view.py
rm -f code/model/inresnet3d.py
rm -f code/model/direc_tsdf.py
rm -f code/model/dense_regre.py
rm -f code/model/super_edt3.py
rm -f code/model/super_vxhit.py
rm -f code/model/super_udir2.py
rm -f code/model/super_ov3edt2m.py
rm -f code/model/super_ov3edt2.py
rm -f code/model/super_ov3dist2.py
rm -f code/model/super_hmap2.py
rm -f code/model/super_dist3.py
rm -f code/model/super_dist2.py

find -type d -name "__pycache__" -exec rm -rf {} +

find . -regextype posix-extended -regex ".*\.py($|\..*)" -type f -exec sed -i "3i\    Cleaned and hand-over to Camera@UoB, `date`" {} \;

sed -i "/^The code in this repo.*/s//This code was written by Xiaokun when he was a postdoc at the CAMERA group of Uni Bath. - `date`/" README.md

cd /tmp
tar -czf $HOME/Downloads/$1.tgz $1/
