import os
import sys
from importlib import import_module
import numpy as np
import matplotlib.pyplot as mpplot
import matplotlib.patches as mppatches
import imageio
from colour import Color
from random import randint
import linecache
import csv
import ops as dataops
import io as dataio

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
make_color_range = getattr(
    import_module('image_ops'),
    'make_color_range'
)
fig2data = getattr(
    import_module('image_ops'),
    'fig2data'
)


def draw_pose2d(thedata, img, pose2d, show_margin=False):
    """ Draw 2D pose on the image domain.
        Args:
            pose2d: nx2 array
    """
    p2wrist = np.array([pose2d[0, :]])
    for fii, joints in enumerate(thedata.join_id):
        p2joints = pose2d[joints, :]
        # color_list = make_color_range(
        #     Color('black'), thedata.join_color[fii + 1], 4)
        # color_range = [C.rgb for C in make_color_range(
        #     color_list[-2], thedata.join_color[fii + 1], len(p2joints) + 1)]
        color_v0 = Color(thedata.join_color[fii + 1])
        color_v0.luminance = 0.3
        color_range = [C.rgb for C in make_color_range(
            color_v0, thedata.join_color[fii + 1], len(p2joints) + 1)]
        for jj, joint in enumerate(p2joints):
            mpplot.plot(
                p2joints[jj, 0], p2joints[jj, 1],
                'o',
                color=color_range[jj + 1]
            )
        p2joints = np.vstack((p2wrist, p2joints))
        mpplot.plot(
            p2joints[:, 0], p2joints[:, 1],
            '-',
            linewidth=2.0,
            color=thedata.join_color[fii + 1].rgb
        )
        # path = mpath.Path(p2joints)
        # verts = path.interpolated(steps=step).vertices
        # x, y = verts[:, 0], verts[:, 1]
        # z = np.linspace(0, 1, step)
        # colorline(x, y, z, cmap=mpplot.get_cmap('jet'))
    # mpplot.gcf().gca().add_artist(
    #     mpplot.Circle(
    #         p2wrist[0, :],
    #         20,
    #         color=[i / 255 for i in thedata.join_color[0]]
    #     )
    # )
    mpplot.plot(
        p2wrist[0, 0], p2wrist[0, 1],
        'o',
        color=thedata.join_color[0].rgb
    )
    # for fii, bone in enumerate(thedata.bone_id):
    #     for jj in range(4):
    #         p0 = pose2d[bone[jj][0], :]
    #         p2 = pose2d[bone[jj][1], :]
    #         mpplot.plot(
    #             (int(p0[0]), int(p0[1])), (int(p2[0]), int(p2[1])),
    #             color=[i / 255 for i in thedata.join_color[fii + 1]],
    #             linewidth=2.0
    #         )
    #         # cv2.line(img,
    #         #          (int(p0[0]), int(p0[1])),
    #         #          (int(p2[0]), int(p2[1])),
    #         #          thedata.join_color[fii + 1], 1)

    return fig2data(mpplot.gcf(), show_margin)


def draw_pose3(thedata, img, pose_mat, show_margin=False):
    """ Draw 3D pose onto 2D image domain: using only (x, y).
        Args:
            pose_mat: nx3 array
    """
    pose2d = dataops.get2d(thedata, pose_mat)

    # draw bounding box
    # bm = dataops.getbm(pose_mat[3, 2])
    bm = 0.25
    rect = dataops.get_rect(thedata, pose2d, bm)
    mpplot.gca().add_patch(mppatches.Rectangle(
        rect[0, :], rect[1, 0], rect[1, 1],
        linewidth=1, facecolor='none',
        edgecolor=thedata.bbox_color.rgb)
    )

    img_posed = draw_pose2d(thedata, img, pose2d, show_margin)
    return img_posed


def draw_pred_random(thedata, image_dir, annot_echt, annot_pred):
    img_id = randint(1, sum(1 for _ in open(annot_pred, 'r')))
    line_echt = linecache.getline(annot_echt, img_id)
    line_pred = linecache.getline(annot_pred, img_id)
    img_name, pose_echt, rescen_echt = dataio.parse_line_pose(line_echt)
    _, pose_pred, rescen_pred = dataio.parse_line_pose(line_pred)
    img_path = os.path.join(image_dir, img_name)
    print('drawing pose #{:d}: {}'.format(img_id, img_path))
    img = dataio.read_image(img_path)

    fig, ax = mpplot.subplots(nrows=1, ncols=2)
    mpplot.subplot(1, 2, 1)
    mpplot.imshow(img, cmap='bone')
    if rescen_echt is None:
        draw_pose3(thedata, img, pose_echt, show_margin=True)
    else:
        draw_pose3(
            thedata,
            img,
            dataops.get3d(thedata, pose_echt, rescen_echt),
            show_margin=True
        )
    mpplot.gcf().gca().set_title('Ground truth')
    mpplot.subplot(1, 2, 2)
    mpplot.imshow(img, cmap='bone')
    if rescen_pred is None:
        draw_pose3(thedata, img, pose_pred, show_margin=True)
    else:
        draw_pose3(
            thedata,
            img,
            dataops.get3d(thedata, pose_pred, rescen_pred),
            show_margin=True
        )
    mpplot.gcf().gca().set_title('Prediction')
    mpplot.show()


def draw_pose_random(thedata, image_dir, annot_txt, img_id=-1):
    """ Draw 3D pose of a randomly picked image.
    """
    if 0 > img_id:
        # img_id = randint(1, thedata.num_training)
        img_id = randint(1, thedata.num_training)
    # Notice that linecache counts from 1
    annot_line = linecache.getline(annot_txt, img_id)
    # annot_line = linecache.getline(annot_txt, 652)  # palm
    # annot_line = linecache.getline(annot_txt, 465)  # the finger
    # print(annot_line)

    img_name, pose_mat, rescen = dataio.parse_line_pose(annot_line)
    img_path = os.path.join(image_dir, img_name)
    print('drawing pose #{:d}: {}'.format(img_id, img_path))
    img = dataio.read_image(img_path)

    mpplot.imshow(img, cmap='bone')
    if rescen is None:
        draw_pose3(thedata, img, pose_mat)
    else:
        draw_pose2d(thedata, img, pose_mat[:, 0:2])
    mpplot.show()
    return img_id


def draw_pose_stream(thedata, gif_file, max_draw=100):
    """ Draw 3D poses and streaming output as GIF file.
    """
    with imageio.get_writer(gif_file, mode='I', duration=0.2) as gif_writer:
        with open(thedata.training_annot_cleaned, 'r') as fa:
            csv_reader = csv.reader(fa, delimiter='\t')
            for lii, annot_line in enumerate(csv_reader):
                if lii >= max_draw:
                    return
                    # raise coder.break_with.Break
                img_name, pose_mat, rescen = dataio.parse_line_pose(annot_line)
                img = dataio.read_image(os.path.join(thedata.training_images, img_name))
                mpplot.imshow(img, cmap='bone')
                img_posed = draw_pose3(img, pose_mat)
                # mpplot.show()
                gif_writer.append_data(img_posed)
                mpplot.gcf().clear()


def draw_bbox_random(thedata):
    """ Draw 3D pose of a randomly picked image.
    """
    # img_id = randint(1, thedata.num_training)
    img_id = randint(1, thedata.num_training)
    # Notice that linecache counts from 1
    annot_line = linecache.getline(thedata.frame_bbox, img_id)
    # annot_line = linecache.getline(thedata.frame_bbox, 652)

    img_name, bbox = dataio.parse_line_bbox(annot_line)
    img_path = os.path.join(thedata.frame_images, img_name)
    print('drawing BoundingBox #{:d}: {}'.format(img_id, img_path))
    img = dataio.read_image(img_path)
    mpplot.imshow(img, cmap='bone')
    # rect = bbox.astype(int)
    rect = bbox
    mpplot.gca().add_patch(mppatches.Rectangle(
        rect[0, :], rect[1, 0], rect[1, 1],
        linewidth=1, facecolor='none',
        edgecolor=thedata.bbox_color.rgb)
    )
    mpplot.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    mpplot.gcf().gca().axis('off')
    mpplot.show()


def draw_hist_random(thedata, image_dir, img_id=-1):
    if 0 > img_id:
        img_id = randint(1, thedata.num_training)
    img_name = 'image_D{:08d}.png'.format(img_id)
    img_path = os.path.join(image_dir, img_name)
    print('drawing hist: {}'.format(img_path))
    img = dataio.read_image(img_path)

    fig, ax = mpplot.subplots(nrows=2, ncols=2)
    mpplot.subplot(2, 2, 1)
    mpplot.imshow(img, cmap='bone')
    mpplot.subplot(2, 2, 2)
    img_val = img.flatten()
    # img_val = [v for v in img.flatten() if (10 > v)]
    mpplot.hist(img_val)
    mpplot.subplot(2, 2, 3)
    img_matt = img
    img_matt[2 > img_matt] = 9999
    mpplot.imshow(img_matt, cmap='bone')
    mpplot.subplot(2, 2, 4)
    img_val = [v for v in img_matt.flatten() if (10 > v)]
    mpplot.hist(img_val)
    mpplot.show()
    return img_id
