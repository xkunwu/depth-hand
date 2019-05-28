# Hand pose estimation

> Single view depth image based pose estimation, given cropped out hand region.

Please check the [paper webpage](https://xkunwu.github.io/research/18HandPose/18HandPose/) for more details about the pipeline and algorithms.

## Prerequisites
Tested on Ubuntu (18.04/16.04) or Windows 10, Python (3.6/2.7), NVIDIA GPU (9.0/8.0) or CPU.
-   Install requirements (presuming [Miniconda](https://conda.io/miniconda.html)):
    ```
    export PYTHON_VERSION=3.6
    conda create -n hand python=$PYTHON_VERSION numpy pip
    source activate hand
    ```
-   [Tensorflow](https://www.tensorflow.org/install/): follow official instructions.

    Note: make sure python version is correct, also install correct version of NVIDIA CUDA/CUDNN if using GPU.

## Usage
#### Installation
-   Clone this repo:
    ```
    git clone https://github.com/xkunwu/depth-hand.git
    cd depth-hand/code
    pip install -r requirements.txt
    ```
#### Basic usage
-   Test using pretrained model (this prepare required data automatically):
    ```
    python -m train.evaluate \
        --data_root=$HOME/data \
        --out_root=$HOME/data/univue/output \
        --model_name=base_clean
    ```
-   Retrain the model (this happen automatically when pretrained model does not present):
    ```
    python -m train.evaluate \
        --data_root=$HOME/data \
        --out_root=$HOME/data/univue/output \
        --retrain \
        --max_epoch=1 \
        --model_name=base_clean
    ```
    Note: this implies that the soft link to pretrained model will be overwritten with your new training output.
    Please see the [Pretrained models section](#pretrained-models) below for the details, if you want to link it back.
#### Print useful information
-   Print currently implemented model list:
    <a name="print-model-list"></a>
    ```
    python -m train.evaluate --print_models
    ```
    Format: (model name) --> (file path): (annotation)

    Current output should like this:
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

#### Code structure
```
%proj_root%/code/
    - args_holder.py: arguments parser (check all the default values here)
    - camera/: the tracking code
    - data/hands17/: data operations (mainly tested on 'hands17' dataset)
        - holder.py: info about the data, preprocessing framework, storage dependencies, preprocessing function entry points
        - draw.py: visualization
        - eval.py: collect evaluation statistics
        - io.py: IO related
        - ops.py: data processing code
        - provider.py: multiprocessing storage provider, save preprocessed data to save training time
    - model/: all models
        - base_regre.py: base class, many template implementation
        - base_clean.py: default setting, baseline evaluation, base class for every other models except 'base_regre', good starting point for extensions
        - base_conv3.py: base class for 3D convolution
        - incept_resnet.py: [Inception-ResNet](https://arxiv.org/abs/1602.07261) module (STAR while I started this project)
        - super_edt2m.py: best standalone model, used in the tracker part
        - super_ov3edt2m.py: best overall, as reported in the paper
    - train/: all training code (some tweaked to specific models)
    - utils/: utility code
```

## Resources
<a name="resources"></a>
Resources can be downloaded from: \[[Baidu Cloud Storage](https://pan.baidu.com/s/1aPda9jG83d9nrAoPr-0MGw)\], \[[Google Drive](https://drive.google.com/open?id=16zZDVJKnQ8QYX94ecRoGl9LVHMPvF5zD)\].

-   [Hands17 dataset](http://icvl.ee.ic.ac.uk/hands17/):
    I am not authorized to redistribute. Please contact the organizers for downloading the data.
-   Pretrained models:
    <a name="pretrained-models"></a>
    ```
    %out_root%/output/hands17/log/
    ```
-   Data preprocessing output: if you do not have the raw data, using preprocessed data is just fine. This one-time preprocessing step is performed so that no additional data operations are necessary in the training stage, which help with reducing training time.

    Especially:
    ```
    %out_root%/output/hands17/prepared/: prepared data root
        - annotation: annotation - training and validation split
        - annotation_test: annotation - test split
        - clean_128: cropped frames for the hand region
        - anything else: preprocessed data tailored to each model
    ```
-   Results used to plot figures in the ECCV2018 paper:
    ```
    %out_root%/output/hands17/predict/
    ```

**Note**: Only pretrained models (i.e. 'log') is shared in the 'Google Drive' link, due to my limited online storage. These pretrained models are the only prerequisite of [Hand tracking part](/code/camera/README.md), so you can still run the demo.

#### File structure
Please make sure that the downloaded files are organized in the following structure, otherwise the code will automatically redo data preprocessing and training (will take quite a while).
```
%out_root%/hands17: if you have Hands17 dataset, put it here
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

#### Prepared data dependency structure
This is the key to understand the data preprocessing work flow.
In a nutshell, each model/algorithm requires different types preprocessed data, and sometimes one type may depends on several other types as input.
So the data types are modularized and abstracted into a hierarchical structure - in correspondence to the inheritance structure of model classes.
-   Print data dependency for a specific model class:
    ```
    python -m train.evaluate --print_model_prereq
    ```
    For example, the output for the best performing standalone model 'super_edt2m' looks like this:
    ```
    Prepared dependency for super_edt2m: {'annotation', 'edt2m_32', 'pose_c', 'clean_128', 'udir2_32', 'edt2_32'}
    ```
    But it requires huge storage: 'udir2_32' is 66G! - An example of time-storage trade-off.
-   Print data dependency hierarchy for a data set:
    ```
    python -m train.evaluate --print_data_deps
    ```
    Format: (data type) --> (dependencies)

    Currently, the output for the Hands17 dataset should look like this:
    ```
    Data dependency hierarchy for hands17:
    index  -->  []
    poses  -->  []
    resce  -->  []
    pose_c  -->  ['poses', 'resce']
    pose_hit  -->  ['poses', 'resce']
    pose_lab  -->  ['poses', 'resce']
    crop2  -->  ['index', 'resce']
    clean  -->  ['index', 'resce']
    ortho3  -->  ['index', 'resce']
    pcnt3  -->  ['index', 'resce']
    tsdf3  -->  ['pcnt3']
    vxhit  -->  ['index', 'resce']
    vxudir  -->  ['pcnt3', 'poses', 'resce']
    edt2  -->  ['clean', 'poses', 'resce']
    hmap2  -->  ['poses', 'resce']
    udir2  -->  ['clean', 'poses', 'resce']
    edt2m  -->  ['edt2', 'udir2']
    ov3dist2  -->  ['vxudir']
    ov3edt2  -->  ['ortho3', 'poses', 'resce']
    ov3edt2m  -->  ['ov3edt2', 'ov3dist2']
    ```
    Note: the output in the 'prepared' fold also have a number appended to these base names, which means the resolution of the data type (configurable for each model class).
    For example, 'clean_128' means the image should be 128x128 in dimensions.

#### Date structure
Each data type is sequentially stored in compressed [HDF5 format](https://www.hdfgroup.org/solutions/hdf5/).
E.g., pose are stored like [this](https://github.com/xkunwu/depth-hand/blob/656309b5d0cd907a4482f50880140c8a23dedacc/code/data/hands17/holder.py#L231).
Note that the data processing part is multi-threaded, so there are a few seemingly convolved code.

Image frames is preprocessed by each model class as required.
E.g., the base model 'base_regre' just crop a sub-region, which looks like [this](https://github.com/xkunwu/depth-hand/blob/656309b5d0cd907a4482f50880140c8a23dedacc/code/model/base_regre.py#L366).

## FAQ
##### Q: Only a few prepared data shared online?
A: The prepared data is huge, but my internet bandwidth is limited, my online storage limit is low.
I would suggest you just run the data preprocessing locally - it's multi-threaded, so actually quite fast if you have a good CPU with many cores.

##### Q: The prepared data is too HUGE!
A: Agreed. But HDF5 format is already doing good job on compression.
Given that the storage is much cheaper than our precious time waiting for training, I would bear with it right now.

##### Q: Where should I start to read the code?
A: Thanks for your interest.
Take your time and read the text above first.
If you are in a hurry, delete everything except those (abstract) base classes and utils - including data preprocessing work flow - then you will feel much relieved to see the essence.
Or take a look at '[scripts/release-bath.sh](/scripts/release-bath.sh)' to see how to clean the code.

## Misc topics
#### Remote management
```
tensorboard --logdir log
jupyter-notebook --no-browser --port=8888
ssh ${1:-sipadan} -L localhost:${2:-1}6006:localhost:6006 -L localhost:${2:-1}8888:localhost:8888
```

#### Get hardware info
```
cat /proc/meminfo
hwinfo --short
```

#### Windows
    ```
    python -m train.evaluate --data_root=C:\Users\xkunw\data --out_root=C:\Users\xkunw\data\univue\output --model_name=super_edt2m
    python -m camera.capture --data_root=C:\Users\xkunw\data --out_root=C:\Users\xkunw\data\univue\output --model_name=super_edt2m
    ```
