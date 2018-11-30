import os
import numpy as np
import matplotlib.pyplot as mpplot
import matplotlib.patches as mppatches
# from mpl_toolkits.mplot3d import Axes3D
import imageio
from colour import Color
import linecache
import csv
from .ops import ops as dataops
from .io import io as dataio
from utils.image_ops import make_color_range
from utils.image_ops import fig2data
from utils.iso_boxes import iso_cube


class draw(object):
    @classmethod
    def draw_pose2d(
        cls, ax, thedata, pose2d,
            with_color=True, show_margin=True):
        """ Draw 2D pose on the image domain.
            Args:
                pose2d: nx2 array, image domain coordinates
        """
        plot_list = []
        p2wrist = np.array([pose2d[0, :]])
        no_color = Color('black').rgb
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
                if with_color:
                    pp = ax.plot(
                        p2joints[jj, 1], p2joints[jj, 0],
                        # p2joints[jj, 0], p2joints[jj, 1],
                        'o',
                        color=color_range[jj + 1]
                    )
                    plot_list.append(pp)
                else:
                    pp = ax.plot(
                        p2joints[jj, 1], p2joints[jj, 0],
                        # p2joints[jj, 0], p2joints[jj, 1],
                        'o',
                        color=no_color
                    )
                    plot_list.append(pp)
            p2joints = np.vstack((p2wrist, p2joints))
            if with_color:
                pp = ax.plot(
                    p2joints[:, 1], p2joints[:, 0],
                    # p2joints[:, 0], p2joints[:, 1],
                    '-',
                    linewidth=2.0,
                    color=thedata.join_color[fii + 1].rgb
                )
                plot_list.append(pp)
            else:
                pp = ax.plot(
                    p2joints[:, 1], p2joints[:, 0],
                    # p2joints[:, 0], p2joints[:, 1],
                    '-',
                    linewidth=2.0,
                    color=no_color
                )
                plot_list.append(pp)
            # path = mpath.Path(p2joints)
            # verts = path.interpolated(steps=step).vertices
            # x, y = verts[:, 0], verts[:, 1]
            # z = np.linspace(0, 1, step)
            # colorline(x, y, z, cmap=mpplot.get_cmap('jet'))
        # ax.add_artist(
        #     ax.Circle(
        #         p2wrist[0, :],
        #         20,
        #         color=[i / 255 for i in thedata.join_color[0]]
        #     )
        # )
        if with_color:
            pp = ax.plot(
                p2wrist[0, 1], p2wrist[0, 0],
                # p2wrist[0, 0], p2wrist[0, 1],
                'o',
                color=thedata.join_color[0].rgb
            )
            plot_list.append(pp)
        else:
            pp = ax.plot(
                p2wrist[0, 1], p2wrist[0, 0],
                # p2wrist[0, 0], p2wrist[0, 1],
                'o',
                color=no_color
            )
            plot_list.append(pp)
        # for fii, bone in enumerate(thedata.bone_id):
        #     for jj in range(4):
        #         p0 = pose2d[bone[jj][0], :]
        #         p2 = pose2d[bone[jj][1], :]
        #         ax.plot(
        #             (int(p0[0]), int(p0[1])), (int(p2[0]), int(p2[1])),
        #             color=[i / 255 for i in thedata.join_color[fii + 1]],
        #             linewidth=2.0
        #         )
        #         # cv2.line(img,
        #         #          (int(p0[0]), int(p0[1])),
        #         #          (int(p2[0]), int(p2[1])),
        #         #          thedata.join_color[fii + 1], 1)

        # return fig2data(mpplot.gcf(), show_margin)
        return plot_list

    @classmethod
    def draw_pose2d_compare(cls, ax, thedata, poses_pred, poses_echt):
        cls.draw_pose2d(ax, thedata, poses_echt, with_color=False)
        cls.draw_pose2d(ax, thedata, poses_pred, with_color=True)

    @classmethod
    def draw_pose_raw(cls, ax, thedata, img, pose_raw, show_margin=False):
        """ Draw 3D pose onto 2D image domain: using only (x, y).
            Args:
                pose_raw: nx3 array
        """
        pose2d = dataops.raw_to_2d(pose_raw, thedata)
        # indz = np.hstack((
        #     pose2d,
        #     pose_raw[:, 2].reshape(-1, 1))
        # )
        # points3 = dataops.d2z_to_raw(indz, thedata)
        # print(pose_raw - points3)

        # draw bounding cube
        cube = iso_cube(
            (np.max(pose_raw, axis=0) + np.min(pose_raw, axis=0)) / 2,
            thedata.region_size
        )
        rect = dataops.get_rect2(cube, thedata)
        rect.draw(ax)

        img_posed = cls.draw_pose2d(
            ax, thedata,
            pose2d,
            show_margin)
        return img_posed

    @classmethod
    def draw_pose_raw_random(cls, thedata, image_dir, annot_txt, img_id=-1):
        """ Draw pose in the frontal view of a randomly picked image.
        """
        if 1 > img_id:
            # img_id = randint(1, thedata.num_training)
            img_id = np.random.randint(1, high=thedata.num_training)
        # Notice that linecache counts from 1
        annot_line = linecache.getline(annot_txt, img_id)
        # annot_line = linecache.getline(annot_txt, 219)  # palm
        # annot_line = linecache.getline(annot_txt, 465)  # the finger
        # print(annot_line)

        img_name_id = dataio.index2imagename(img_id)
        img_name, pose_raw = dataio.parse_line_annot(annot_line)
        if img_name_id != img_name:
            annot_line = dataio.get_line(annot_txt, img_id)
            img_name, pose_raw = dataio.parse_line_annot(annot_line)
        img_path = os.path.join(image_dir, img_name)
        print('drawing image #{:d}: {}'.format(img_id, img_path))
        img = dataio.read_image(img_path)

        mpplot.imshow(img, cmap=mpplot.cm.bone_r)
        ax = mpplot.gca()
        # if resce is None:
        cls.draw_pose_raw(ax, thedata, img, pose_raw)
        # else:
        #     cls.draw_pose2d(ax, thedata, pose_raw[:, 0:2])
        # mpplot.show()
        return img_id

    @classmethod
    def draw_pose_stream(cls, thedata, gif_file, max_draw=100):
        """ Draw 3D poses and streaming output as GIF file.
        """
        with imageio.get_writer(gif_file, mode='I', duration=0.2) as gif_writer:
            with open(thedata.annotation_train, 'r') as fa:
                csv_reader = csv.reader(fa, delimiter='\t')
                for lii, annot_line in enumerate(csv_reader):
                    if lii >= max_draw:
                        return
                        # raise coder.break_with.Break
                    img_name, pose_raw, resce = dataio.parse_line_annot(annot_line)
                    img = dataio.read_image(os.path.join(thedata.training_images, img_name))
                    mpplot.imshow(img, cmap=mpplot.cm.bone_r)
                    ax = mpplot.gca()
                    img_posed = cls.draw_pose_raw(ax, img, pose_raw)
                    # mpplot.show()
                    gif_writer.append_data(img_posed)
                    mpplot.gcf().clear()

    @classmethod
    def draw_raw3d_pose(cls, ax, thedata, pose_raw, zdir='z'):
        p3wrist = np.array([pose_raw[0, :]])
        for fii, joints in enumerate(thedata.join_id):
            p3joints = pose_raw[joints, :]
            color_v0 = Color(thedata.join_color[fii + 1])
            color_v0.luminance = 0.3
            color_range = [C.rgb for C in make_color_range(
                color_v0, thedata.join_color[fii + 1], len(p3joints) + 1)]
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
                color=thedata.join_color[fii + 1].rgb,
                zdir=zdir
            )
        ax.scatter(
            p3wrist[0, 0], p3wrist[0, 1], p3wrist[0, 2],
            color=thedata.join_color[0].rgb,
            zdir=zdir
        )

    @classmethod
    def draw_raw3d(cls, thedata, img, pose_raw):
        cube = iso_cube()
        cube.build(pose_raw)
        # draw full image
        fig_size = (2 * 6, 6)
        fig = mpplot.figure(figsize=fig_size)
        points3 = dataops.img_to_raw(img, thedata)
        numpts = points3.shape[0]
        if 1000 < numpts:
            samid = np.random.choice(numpts, 1000, replace=False)
            points3_sam = points3[samid, :]
        else:
            points3_sam = points3
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.scatter(
            points3_sam[:, 0], points3_sam[:, 1], points3_sam[:, 2],
            color=Color('lightsteelblue').rgb)
        ax.view_init(azim=-90, elev=-75)
        ax.set_zlabel('depth (mm)', labelpad=15)
        cls.draw_raw3d_pose(ax, thedata, pose_raw)
        corners = cube.get_corners()
        corners = cube.transform_add_center(corners)
        cube.draw_wire(corners)
        # draw cropped region
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        points3_trans = cube.pick(points3)
        points3_trans = cube.transform_to_center(points3_trans)
        numpts = points3_trans.shape[0]
        if 1000 < numpts:
            points3_sam = points3_trans[np.random.choice(numpts, 1000, replace=False), :]
        else:
            points3_sam = points3_trans
        points3_sam = cube.transform_to_center(points3_sam)
        pose_trans = cube.transform_to_center(pose_raw)
        ax.scatter(
            points3_sam[:, 0], points3_sam[:, 1], points3_sam[:, 2],
            color=Color('lightsteelblue').rgb)
        cls.draw_raw3d_pose(ax, thedata, pose_trans)
        corners = cube.get_corners()
        cube.draw_wire(corners)
        ax.view_init(azim=-120, elev=-150)
        mpplot.tight_layout()
        mpplot.show()
        # draw projected image
        fig_size = (3 * 5, 5)
        mpplot.subplots(nrows=1, ncols=3, figsize=fig_size)
        for spi in range(3):
            ax = mpplot.subplot(1, 3, spi + 1)
            coord, depth = cube.project_ortho(points3_trans, roll=spi)
            img = cube.print_image(coord, depth, thedata.crop_size)
            pose2d, _ = cube.project_ortho(pose_trans, roll=spi, sort=False)
            cls.draw_pose2d(ax, thedata, pose2d)
            ax.imshow(img, cmap=mpplot.cm.bone_r)
            ax.axis('off')
        mpplot.tight_layout()
        mpplot.show()

    @classmethod
    def draw_raw3d_random(cls, thedata, image_dir, annot_txt, img_id=-1):
        """ Draw 3D pose of a randomly picked image.
        """
        if 0 > img_id:
            # img_id = np.random.randint(1, high=thedata.num_training)
            img_id = np.random.np.random.randint(1, high=thedata.num_training)
        # Notice that linecache counts from 1
        annot_line = linecache.getline(annot_txt, img_id)
        # annot_line = linecache.getline(annot_txt, 219)  # palm
        # annot_line = linecache.getline(annot_txt, 465)  # the finger
        # print(annot_line)

        img_name, pose_raw = dataio.parse_line_annot(annot_line)
        img_path = os.path.join(image_dir, img_name)
        print('drawing image #{:d}: {}'.format(img_id, img_path))
        img = dataio.read_image(img_path)

        cls.draw_raw3d(thedata, img, pose_raw)
        fig_size = (3 * 5, 5)
        mpplot.subplots(nrows=1, ncols=2, figsize=fig_size)
        ax = mpplot.subplot(1, 3, 1)
        ax.imshow(img, cmap=mpplot.cm.bone_r)
        cls.draw_pose_raw(thedata, img, pose_raw)
        ax = mpplot.subplot(1, 3, 2)
        img_crop_resize, resce = dataops.crop_resize(
            img, pose_raw, thedata)
        ax.imshow(img_crop_resize, cmap=mpplot.cm.bone_r)
        cls.draw_pose2d(
            ax, thedata,
            dataops.raw_to_2d(pose_raw, thedata, resce))
        ax.set_title('Cropped')
        ax = mpplot.subplot(1, 3, 3)
        img_crop_resize, resce = dataops.crop_resize_pca(
            img, pose_raw, thedata)
        ax.imshow(img_crop_resize, cmap=mpplot.cm.bone_r)
        cls.draw_pose2d(
            ax, thedata,
            dataops.raw_to_2d(pose_raw, thedata, resce))
        ax.set_title('Cleaned')
        ax.axis('off')
        mpplot.tight_layout()
        mpplot.show()
        return img_id

    @classmethod
    def draw_bbox_random(cls, thedata):
        """ Draw 3D pose of a randomly picked image.
        """
        # img_id = np.random.randint(1, high=thedata.num_training)
        img_id = np.random.randint(1, high=thedata.num_training)
        # Notice that linecache counts from 1
        annot_line = linecache.getline(thedata.frame_bbox, img_id)
        # annot_line = linecache.getline(thedata.frame_bbox, 652)

        img_name, bbox = dataio.parse_line_bbox(annot_line)
        img_path = os.path.join(thedata.frame_images, img_name)
        print('drawing BoundingBox #{:d}: {}'.format(img_id, img_path))
        img = dataio.read_image(img_path)
        mpplot.imshow(img, cmap=mpplot.cm.bone_r)
        # rect = bbox.astype(int)
        rect = bbox
        mpplot.gca().add_patch(mppatches.Rectangle(
            rect[0, :], rect[1, 0], rect[1, 1],
            linewidth=1, facecolor='none',
            edgecolor=thedata.bbox_color.rgb)
        )
        mpplot.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        mpplot.gca().axis('off')
        mpplot.show()

    @classmethod
    def draw_hist_random(cls, thedata, image_dir, img_id=-1):
        if 0 > img_id:
            img_id = np.random.randint(1, high=thedata.num_training)
        img_name = dataio.index2imagename(img_id)
        img_path = os.path.join(image_dir, img_name)
        print('drawing hist: {}'.format(img_path))
        img = dataio.read_image(img_path)

        mpplot.subplots(nrows=2, ncols=2)
        mpplot.subplot(2, 2, 1)
        mpplot.imshow(img, cmap=mpplot.cm.bone_r)
        mpplot.subplot(2, 2, 2)
        img_val = img.flatten()
        # img_val = [v for v in img.flatten() if (10 > v)]
        mpplot.hist(img_val)
        mpplot.subplot(2, 2, 3)
        img_matt = img
        img_matt[2 > img_matt] = thedata.z_max
        mpplot.imshow(img_matt, cmap=mpplot.cm.bone_r)
        mpplot.subplot(2, 2, 4)
        img_val = [v for v in img_matt.flatten() if (10 > v)]
        mpplot.hist(img_val)
        mpplot.show()
        return img_id
