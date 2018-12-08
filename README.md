# Hand in Depth

> Single view depth image based hand detection and pose estimation.

<span style="display:block;text-align:center">![Test sequence.](/eval/test_seq.gif)</span>

## Introduction
This repo includes the code for:

> HandMap: Robust hand pose estimation via intermediate dense guidance map supervision, in ECCV2018. \[[BibTex](/eval/Wu18HandPose.txt)\]

This repo also includes following up work for real-time [tracking](#tracking).

If you use code in this repo for your research, please cite the above mentioned paper.

The code in this repo is written and maintained by [Xiaokun Wu](https://xkunwu.github.io/).

## Pose estimation
<a name="pose-estimation"></a>
Please check [code/README.md](/code/README.md) for the pose estimation part of this project.

## Tracking
<a name="tracking"></a>
If you happen to have a Intel® RealSense™ depth camera, you can also take a look at [code/camera/README.md](/code/camera/README.md) for the tracking part of this project.
Code for real-time capture, detection, tracking is provided there, which can be use for pose estimation.
