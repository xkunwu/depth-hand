# Hand in Depth

> Single view depth image based hand detection and pose estimation.

<span style="display:block;text-align:center">![Test sequence.](eval/test_seq.gif)</span>

## Introduction
This repo includes the code for  paper:

> HandMap: Robust hand pose estimation via intermediate dense guidance map supervision, in ECCV2018. \[[BibTex](eval/Wu18HandPose.txt)\]

If you use this code for your research, please cite the above mentioned paper.

This repo also includes following up work for real-time [tracking](#tracking).

The code in this repo is written and maintained by [Xiaokun Wu](https://xkunwu.github.io/).

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
    pip install -r code/requirements.txt
    ```
-   Test using pre-trained model:
    ```
    python -m train.evaluate \
        --data_root=$(pwd)/../data \
        --out_root=$(pwd)/../output \
        --model_name=base_clean
    ```
-   Print currently implemented model list:
    ```
    python -m train.evaluate --print_models
    ```
    Format: (model name) --> (file path): (annotation)

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
    - train/: all training code (some tweaked to specific models)
    - utils/: utility code
```

## Tracking
<a name="tracking"></a>
If you happen to have a Intel® RealSense™ depth camera, you can also take a look and test the [tracking part](code/camera/) of this project.
Code for real-time capture, detection, tracking is provided there, which can be use for pose estimation.
