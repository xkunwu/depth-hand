import os
import sys
from importlib import import_module
import numpy as np
import matplotlib.pyplot as mpplot
import matplotlib.patches as mppatches
# from mpl_toolkits.mplot3d import Axes3D
import imageio
from colour import Color
import linecache
import csv
from . import ops as dataops
from . import io as dataio
from utils.image_ops import make_color_range
from utils.image_ops import fig2data
from utils.iso_boxes import iso_cube


def draw_pose2d(ax, thedata, pose2d, show_margin=True):
    """ Draw 2D pose on the image domain.
        Args:
            pose2d: nx2 array, image domain coordinates
    """
    pose2d = pose2d[:, ::-1]
    p2wrist = np.array([pose2d[0, :]])
    for fii, joints in enumerate(thedata.join_id[2:]):
        p2joints = pose2d[joints, :]
        color_v0 = Color(thedata.join_color[fii + 2])
        color_v0.luminance = 0.3
        color_range = [C.rgb for C in make_color_range(
            color_v0, thedata.join_color[fii + 2], len(p2joints) + 1)]
        for jj, joint in enumerate(p2joints):
            ax.plot(
                p2joints[jj, 0], p2joints[jj, 1],
                'o',
                color=color_range[jj + 1]
            )
        p2joints = np.vstack((p2wrist, p2joints))
        ax.plot(
            p2joints[:, 0], p2joints[:, 1],
            '-',
            linewidth=2.0,
            color=thedata.join_color[fii + 2].rgb
        )
    ax.plot(
        p2wrist[0, 0], p2wrist[0, 1],
        'o',
        color=thedata.join_color[0].rgb
    )
    p2joints = np.array(pose2d[thedata.join_id[1], :])
    ax.plot(
        p2joints[:, 0], p2joints[:, 1],
        'o',
        color=thedata.join_color[1].rgb
    )
    p2joints = np.vstack((p2wrist, p2joints))
    p2joints[[0, 1], :] = p2joints[[1, 0], :]
    ax.plot(
        p2joints[:, 0], p2joints[:, 1],
        '-',
        linewidth=2.0,
        color=thedata.join_color[1].rgb
    )


def draw_pose_raw(ax, thedata, img, pose_raw, show_margin=False):
    """ Draw 3D pose onto 2D image domain: using only (x, y).
        Args:
            pose_raw: nx3 array
    """
    pose2d = dataops.raw_to_2d(pose_raw, thedata)

    # draw bounding cube
    cube = iso_cube(
        (np.max(pose_raw, axis=0) + np.min(pose_raw, axis=0)) / 2,
        thedata.region_size
    )
    rect = dataops.get_rect2(cube, thedata)
    rect.draw(ax)

    img_posed = draw_pose2d(
        ax, thedata,
        pose2d,
        show_margin)
    return img_posed


def draw_raw3d_pose(ax, thedata, pose_raw, zdir='z'):
    p3wrist = np.array([pose_raw[0, :]])
    for fii, joints in enumerate(thedata.join_id[2:]):
        p3joints = pose_raw[joints, :]
        color_v0 = Color(thedata.join_color[fii + 2])
        color_v0.luminance = 0.3
        color_range = [C.rgb for C in make_color_range(
            color_v0, thedata.join_color[fii + 2], len(p3joints) + 1)]
        for jj, joint in enumerate(p3joints):
            ax.scatter(
                p3joints[jj, 0], p3joints[jj, 1], p3joints[jj, 2],
                color=color_range[jj + 1],
                zdir=zdir
            )
        p3joints_w = np.vstack((p3wrist, p3joints))
        ax.plot(
            p3joints_w[:, 0], p3joints_w[:, 1], p3joints_w[:, 2],
            '-',
            linewidth=2.0,
            color=thedata.join_color[fii + 2].rgb,
            zdir=zdir
        )
    ax.scatter(
        p3wrist[0, 0], p3wrist[0, 1], p3wrist[0, 2],
        color=thedata.join_color[0].rgb,
        zdir=zdir
    )
    p3joints = np.array(pose_raw[thedata.join_id[1], :])
    ax.plot(
        p3joints[:, 0], p3joints[:, 1], p3joints[:, 2],
        'o',
        color=thedata.join_color[1].rgb
    )
    p3joints = np.vstack((p3wrist, p3joints))
    p3joints[[0, 1], :] = p3joints[[1, 0], :]
    ax.plot(
        p3joints[:, 0], p3joints[:, 1], p3joints[:, 2],
        '-',
        linewidth=2.0,
        color=thedata.join_color[1].rgb
    )
