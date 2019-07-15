# Hand in Depth

> Single view depth image based hand detection and pose estimation.

<span style="display:block;text-align:center">![Test sequence.](/eval/test_seq.gif)</span>

## Introduction
This repo contains the code for:

> HandMap: Robust hand pose estimation via intermediate dense guidance map supervision, in ECCV2018. \[[Webpage](https://xkunwu.github.io/research/18HandPose/18HandPose)\] \[[BibTex](/eval/Wu18HandPose.txt)\]

This repo also contains following up work for real-time tracking: \[[Webpage](https://xkunwu.github.io/projects/hand-track/hand-track)\].

If you use code in this repo for your research, please cite the above mentioned paper.

The code in this repo is written and maintained by [Xiaokun Wu](https://xkunwu.github.io).

## Pose estimation
<a name="pose-estimation"></a>
Please check [code/README.md](/code/README.md) for the pose estimation part of this project.

## Tracking
<a name="tracking"></a>
If you happen to have a Intel® RealSense™ depth camera, you can also take a look at [code/camera/README.md](/code/camera/README.md) for the tracking part of this project.

Code for real-time capture, detection, and tracking is provided there, which is an application of the pose estimation part and produces the teaser figure above.

## Quick start
<a name="quick-start"></a>
Tested on Windows10 (with Anaconda), July 15, 2019. Linux is similar and should be easier.

1.  Setting up:

```
// create a virtual env using conda
export PYTHON_VERSION=3.6
conda create -n hand python=$PYTHON_VERSION numpy pip
source activate hand

// install tensorflow, change to 'tensorflow' package if you do not want/have GPU
pip install --upgrade tensorflow-gpu

// clone this repo
git clone https://github.com/xkunwu/depth-hand.git
cd depth-hand/code
pip install -r requirements.txt
```

2.  Prepare data (using my favorite model 'super_edt2m' as example):

-   Download from \[[Baidu Cloud Storage](https://pan.baidu.com/s/1aPda9jG83d9nrAoPr-0MGw)\] into '$HOME/data/univue/output', so you should find
'$HOME/data/univue/output/hands17/prepared/annotation', and
'$HOME/data/univue/output/hands17/log/log-super_edt2m-180222-112534/model.ckpt.index', and others.
-   Create a softlink folder (shortcut in Windows) '$HOME/data/univue/output/hands17/log/blinks/super_edt2m/', which should point to '$HOME/data/univue/output/hands17/log/log-super_edt2m-180222-112534/'.

Note: you should change '$HOME' to a meaningful path in Windows.

3.  Test using pretrained model:

```
python -m train.evaluate --data_root=$HOME/data --out_root=$HOME/data/univue/output --model_name=super_edt2m
```

There should be no errors, and you are ready to go. Check the details in [README.md in the pose estimation part](/code/README.md) if you want to train your own model.

4.  Live detection using depth camera:

```
python -m camera.capture --data_root=$HOME/data --out_root=$HOME/data/univue/output --model_name=super_edt2m
```
