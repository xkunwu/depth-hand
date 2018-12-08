## Prerequisites
Tested on Ubuntu (18.04/16.04), Python (2/3), CPU or NVIDIA GPU.
-   Install requirements (presuming [Miniconda](https://conda.io/miniconda.html)):
    ```
    export PYTHON_VERSION=3.6
    conda create -n hand python=$PYTHON_VERSION numpy pip
    source activate hand
    ```
-   [Tensorflow](https://www.tensorflow.org/install/): follow official instructions.

    Note: make sure the python version is correct

## Resources
-   Pretrained model:
-   Data preprocessing output:
-   Results used to plot figures in the ECCV2018 paper:
### File structure
Please make sure to the downloaded files are organized in the following structure, otherwise the code will automatically redo data preprocessing and training (taking quite a while).
```
%out_root%/output/hands17/
    - log/: log, pretrained models
        - univue.log: main log
        - log-%model_name%-%timestamp%/: training output (include pretrained model)
            - args.txt: arguments for this run
            - univue.log: main log for this run
            - train.log: training/validation/evaluation log
            - model.ckpt*: pretrained model
        - blinks/: soft links for each model, so referring to different timestamp is possible
    - predict/: predictions (used in paper), plots for sanity check and statistics
        - predict_%model_name%: predictions for test split
        - draw_%model_name%_%frame_id%.png: sanity check
        - detection_%model_name%_%frame_id%.png: detection example visualization
        - %model_name%_error_rate.png: error rate (see paper)
        - %model_name%_error_bar.png: error bar (see paper)
        - error_rate.png, error_bar.png: final summary for chosen models
    - prepared/: preprocessing output
    - capture/: output for real-time tracking
        - stream/: raw stream
        - detection_%timestamp%/: detection with timestamp
```

## Usage
### Installation
-   Clone this repo:
    ```
    git clone https://github.com/xkunwu/depth-hand.git
    cd depth-hand/code
    pip install -r requirements.txt
    ```
-   Test using pre-trained model:
    ```
    python -m train.evaluate \
        --data_root=$HOME/data \
        --out_root=$HOME/data/univue/output \
        --model_name=base_clean
    ```
-   Retrain the model:
    ```
    python -m train.evaluate \
        --data_root=$HOME/data \
        --out_root=$HOME/data/univue/output \
        --retrain \
        --max_epoch=1 \
        --model_name=base_clean
    ```
-   Print currently implemented model list:
    ```
    python -m train.evaluate --print_models
    ```
    Format: (model name) --> (file path): (annotation)

    Current output:
    ```
    Currently implemented model list:
    super_edt3  -->  model.super_edt3 :  EDT3
    super_ov3edt2m  -->  model.super_ov3edt2m :  MV-CR w/ surface distance (weighted)
    super_ov3dist2  -->  model.super_ov3dist2 :  MV-CR w/ Euclidean distance
    super_ov3edt2  -->  model.super_ov3edt2 :  MV-CR w/ surface distance
    super_edt2m  -->  model.super_edt2m :  2D CR w/ surface distance (weighted)
    super_edt2  -->  model.super_edt2 :  2D CR w/ surface distance
    super_dist3  -->  model.super_dist3 :  3D CR w/ Euclidean distance
    voxel_regre  -->  model.voxel_regre :  3D CR w/ offset
    voxel_offset  -->  model.voxel_offset :  3D offset regression
    super_vxhit  -->  model.super_vxhit :  3D CR w/ detection
    voxel_detect  -->  model.voxel_detect :  Moon et al. (CVPR'18)
    super_dist2  -->  model.super_dist2 :  2D CR w/ Euclidean distance
    super_udir2  -->  model.super_udir2 :  2D CR w/ offset
    super_hmap2  -->  model.super_hmap2 :  2D CR w/ heatmap
    dense_regre  -->  model.dense_regre :  2D offset regression
    direc_tsdf  -->  model.direc_tsdf :  Ge et al. (CVPR'17)
    trunc_dist  -->  model.trunc_dist :  3D truncated Euclidean distance
    base_conv3  -->  model.base_conv3 :  3D CR
    base_conv3_inres  -->  model.base_inres :  3D CR w/ inception-resnet
    ortho3view  -->  model.ortho3view :  Ge et al. (CVPR'16)
    base_clean  -->  model.base_clean :  2D CR
    base_regre  -->  model.base_regre :  2D CR-background
    base_clean_inres  -->  model.base_inres :  2D CR w/ inception-resnet
    base_regre_inres  -->  model.base_inres :  2D CR-background w/ inception-resnet
    base_clean_hg  -->  model.base_hourglass :  2D CR w/ hourglass
    base_regre_hg  -->  model.base_hourglass :  2D CR-background w/ hourglass
    localizer3  -->  model.localizer3 :  3D localizer
    localizer2  -->  model.localizer2 :  2D localizer
    ```

### File structure
```
%proj_root%/code/
    - args_holder.py: arguments parser (check all the default values here)
    - camera/: the tracking code
    - data/hands17/: data operations (only 'hands17' working)
        - holder.py: info about the data, preprocessing framework, storage dependencies, preprocessing function entry points
        - draw.py: visualization
        - eval.py: collect evaluation statistics
        - io.py: IO related
        - ops.py: data processing code
        - provider.py: multiprocessing storage provider, save preprocessed data to save training time
    - model/: all models
        - base_regre.py: base class, many template implementation
        - base_conv3.py: base class for 3D convolution
        - base_clean.py: baseline evaluation
        - incept_resnet.py: [Inception-ResNet](https://arxiv.org/abs/1602.07261) module (STAR while I started this project)
        - super_edt2m.py: best standalone model, used in the tracker part
        - super_ov3edt2m.py: best overall, as reported in the paper
    - train/: all training code (some tweaked to specific models)
    - utils/: utility code
```

## Misc topics
### Remote management
```
tensorboard --logdir log
jupyter-notebook --no-browser --port=8888
ssh ${1:-sipadan} -L localhost:${2:-1}6006:localhost:6006 -L localhost:${2:-1}8888:localhost:8888
```

### Get hardware info
```
cat /proc/meminfo
hwinfo --short
```
