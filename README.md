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
