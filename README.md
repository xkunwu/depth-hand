docker build -t handpose .
nvidia-docker run -ti --rm \
    -v ${HOME}/projects/univue-hand-pose/code:/workspace/code:ro \
    -v ${HOME}/data:/data:ro \
    -v ${HOME}/data/univue/output:/output \
    -e "TERM=xterm-256color" \
    --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
    handpose

cd code
python -m train.evaluate \
    --data_root=/data \
    --out_root=/output \
    --max_epoch=1 --batch_size=5 --bn_decay=0.9 \
    --show_draw=True --model_name=base_clean

mv -t ../ predict_*
mv -t ./ ../predict_base_clean ../predict_base_clean_inres ../predict_super_dist2 ../predict_super_edt2 ../predict_super_edt2m ../predict_super_hmap2 ../predict_super_udir2
mv -t ./ ../predict_base_conv3 ../predict_super_dist3 ../predict_trunc_dist ../predict_voxel_regre
mv -t ./ ../predict_direc_tsdf ../predict_ortho3view ../predict_super_edt2m ../predict_voxel_detect
mv -t ./ ../predict_voxel_detect ../predict_ortho3view ../predict_super_ov3dist2 ../predict_super_ov3edt2 ../predict_super_ov3edt2m ../predict_super_vxhit
mv -t ./ ../predict_*
