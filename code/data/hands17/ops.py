# import os
# import sys
from importlib import import_module
import numpy as np
import matplotlib.pyplot as mpplot
import skfmm
import scipy.ndimage as ndimage
from cv2 import resize as cv2resize
from utils.iso_boxes import iso_rect
# from utils.iso_boxes import iso_aabb
from utils.iso_boxes import iso_cube
from utils.regu_grid import regu_grid
# from utils.regu_grid import grid_cell
from utils.regu_grid import latice_image


def raw_to_pca(points3, resce=np.array([1, 0, 0, 0])):
    cube = iso_cube()
    cube.load(resce)
    return cube.transform_to_center(points3)


def pca_to_raw(points3, resce=np.array([1, 0, 0, 0])):
    cube = iso_cube()
    cube.load(resce)
    return cube.transform_add_center(points3)


def raw_to_local(points3, resce=np.array([1, 0, 0, 0])):
    cube = iso_cube()
    cube.load(resce)
    return cube.transform_to_center(points3)
    # aabb = iso_aabb()
    # aabb.load(resce)
    # return aabb.transform_to_center(points3)


def local_to_raw(points3, resce=np.array([1, 0, 0, 0])):
    cube = iso_cube()
    cube.load(resce)
    return cube.transform_add_center(points3)
    # aabb = iso_aabb()
    # aabb.load(resce)
    # return aabb.transform_add_center(points3)


def d2z_to_raw(p2z, caminfo, resce=np.array([1, 0, 0])):
    """ reproject 2d poses to 3d.
        p2z: nx3 array, 2d position and real z
    """
    p2z = p2z.astype(float)
    pose2d = p2z[:, 0:2] / resce[0] + resce[1:3]
    pose_z = np.array(p2z[:, 2]).reshape(-1, 1)
    pose2d = pose2d[:, ::-1]
    pose3d = (pose2d - caminfo.centre) / caminfo.focal * pose_z
    return np.hstack((pose3d, pose_z))


def raw_to_2dz(points3, caminfo, resce=np.array([1, 0, 0])):
    """ project 3D point onto image plane using camera info
        Args:
            points3: nx3 array, raw input in real world coordinates
    """
    points3 = points3.astype(float)
    pose_z = points3[:, 2]
    pose2d = points3[:, 0:2] / pose_z.reshape(-1, 1) * caminfo.focal + caminfo.centre
    pose2d = pose2d[:, ::-1]
    return (pose2d - resce[1:3]) * resce[0], pose_z


def raw_to_2d(points3, caminfo, resce=np.array([1, 0, 0])):
    pose2d, _ = raw_to_2dz(points3, caminfo, resce)
    return pose2d


def raw_to_heatmap2(pose_raw, cube, hmap_size, caminfo):
    """ 2d heatmap for each joint """
    points3_trans = cube.transform_center_shrink(pose_raw)
    coord, depth = cube.project_pca(points3_trans, sort=False)
    img_l = []
    for c, d in zip(coord, depth):
        img = cube.print_image(
            c.reshape(1, -1), np.array([d]), hmap_size)
        img = ndimage.gaussian_filter(img, sigma=0.8)
        img /= np.max(img)
        # mpplot = import_module('matplotlib.pyplot')
        # mpplot.imshow(img, cmap='bone')
        # mpplot.show()
        img_l.append(img)
    img_arr = np.stack(img_l, axis=2)
    return img_arr


def raw_to_offset(img, pose_raw, cube, hmap_size, caminfo):
    """ offset map from depth to each joint """
    from numpy import linalg
    depth_raw = cube.pick(img_to_raw(img, caminfo))
    # depth_normed = cube.transform_center_shrink(depth_raw)
    # coord, _ = cube.project_pca(depth_normed)
    # coord, _ = cube.raw_to_unit(depth_raw)
    coord, depth = cube.raw_to_unit(depth_raw)
    depth_id = np.argsort(depth)
    coord = coord[depth_id, :]
    depth_raw = depth_raw[depth_id, :]
    omap_l = []
    hmap_l = []
    umap_l = []
    theta = caminfo.region_size * 2  # maximal - cube size
    for joint in pose_raw:
        offset = joint - depth_raw  # offset in raw 3d
        offset[:, 1] *= -1  # NOTE: projective - reversed y
        # offset[:, 0], offset[:, 1] = offset[:, 1], offset[:, 0].copy()
        dist = linalg.norm(offset, axis=1)  # offset norm
        if np.min(dist) > theta:
            # due to occlution, we cannot use small radius
            print(np.min(dist))
            # from colour import Color
            # ax = mpplot.subplot(projection='3d')
            # points3 = img_to_raw(img, caminfo)
            # # points3_trans = cube.pick(points3)
            # # points3_trans = cube.transform_to_center(points3_trans)
            # points3_trans = points3
            # numpts = points3_trans.shape[0]
            # if 1000 < numpts:
            #     points3_trans = points3_trans[
            #         np.random.choice(numpts, 1000, replace=False), :]
            # ax.scatter(
            #     points3_trans[:, 0], points3_trans[:, 1], points3_trans[:, 2],
            #     color=Color('lightsteelblue').rgb)
            # import data.hands17.draw as data_draw
            # data_draw.draw_raw3d_pose(caminfo, pose_raw)
            # corners = cube.transform_to_center(cube.get_corners())
            # cube.draw_cube_wire(corners)
            # ax.view_init(azim=-120, elev=-150)
            # mpplot.show()

        valid_id = np.where(np.logical_and(
            1e-1 < dist,  # remove sigular point
            theta > dist  # limit support within theta
        ))
        offset = offset[valid_id]
        dist = dist[valid_id]
        unit_off = offset / np.tile(dist, [3, 1]).T  # unit offset
        dist = theta - dist  # inverse propotional
        coord_valid = coord[valid_id]
        for dim in range(3):
            om = cube.print_image(coord_valid, offset[:, dim], hmap_size)
            omap_l.append(om)
            um = cube.print_image(coord_valid, unit_off[:, dim], hmap_size)
            umap_l.append(um)
            # mpplot.subplot(3, 3, 4 + dim)
            # mpplot.imshow(om, cmap='bone')
            # mpplot.subplot(3, 3, 7 + dim)
            # mpplot.imshow(um, cmap='bone')
        hm = cube.print_image(coord_valid, dist, hmap_size)
        hmap_l.append(hm)
        # print(np.histogram(dist, range=(1e-4, np.max(dist))))
        # mpplot.subplot(3, 3, 1)
        # mpplot.imshow(hm, cmap='bone')
        # mpplot.subplot(3, 3, 3)
        # mpplot.imshow(img, cmap=mpplot.cm.jet)
        # mpplot.show()
    offset_map = np.stack(omap_l, axis=2)
    hmap3 = np.stack(hmap_l, axis=2)
    uomap = np.stack(umap_l, axis=2)
    return offset_map, hmap3, uomap


def hmap3_to_raw(hmap2, hmap3, uomap, depth, cube, hmap_size, caminfo, nn=5):
    """ recover 3d from weight avarage """
    num_joint = hmap3.shape[2]
    theta = caminfo.region_size * 2
    pose_out = np.empty([num_joint, 3])
    for joint in range(num_joint):
        # restore from 3d
        hm3 = hmap3[..., joint]
        hm3[np.where(1e-2 > depth)] = 0  # mask out void
        top_id = hm3.argpartition(-nn, axis=None)[-nn:]  # top elements
        x3, y3 = np.unravel_index(top_id, hm3.shape)
        dist = hm3[x3, y3]
        dist = theta - dist  # inverse propotional
        unit_off = uomap[x3, y3, 3 * joint:3 * (joint + 1)]
        offset = unit_off * np.tile(dist, [3, 1]).T
        offset[:, 1] *= -1  # NOTE: projective - reversed y
        p0 = cube.unit_to_raw(
            np.vstack([x3, y3]).astype(float).T / hmap_size,
            depth[x3, y3])
        pred3 = p0 + offset
        # restore from 2d
        hm2 = hmap2[..., joint]
        coord, _ = cube.raw_to_unit(pred3)
        coord = coord.clip(0., 1.)  # pointing out is not good
        coord *= hmap_size
        coord *= 0.999999
        coord = np.floor(coord).astype(int)
        x2, y2 = coord[:, 0], coord[:, 1]
        conf = hm2[x2, y2]
        pred32 = np.sum(pred3 * np.tile(conf, [3, 1]).T, axis=0) / np.sum(conf)
        pose_out[joint, :] = pred32
    return pose_out


def estimate_z(l3, l2, focal):
    """ depth can be estimated due to:
        - same projective mapping
        - fixed region size
    """
    # p3 = np.array([[-12, -54, 456], [22, 63, 456]])
    # # p3 = np.array([[0, 0, 456], [12, 34, 456]])
    # # p3 = np.array([[456, -456, 456], [456, 456, 456]])
    # p2, z = ARGS.data_ops.raw_to_2dz(p3, ARGS.data_inst)
    # print(p2, z)
    # print(ARGS.data_ops.estimate_z(
    #     np.sqrt(np.sum((p3[0] - p3[1]) ** 2)),
    #     np.sqrt(np.sum((p2[0] - p2[1]) ** 2)),
    #     ARGS.data_inst.focal[0]))
    return float(l3) * focal / l2  # assume same focal


def proj_cube_to_rect(cube, region_size, caminfo):
    """ central z-plane of 3D cube --> image plane """
    c3a = np.array([
        np.append(cube.cen[:2] - region_size, cube.cen[2]),
        np.append(cube.cen[:2] + region_size, cube.cen[2])
    ])  # central z-plane
    c2a = raw_to_2d(c3a, caminfo)
    cll = c2a[0, :]
    ctr = c2a[1, :]
    return iso_rect(cll, np.max(ctr - cll))


def recover_from_rect(rect, region_size, caminfo):
    z_cen = estimate_z(region_size, rect.sidelen / 2, caminfo.focal[0])
    centre = d2z_to_raw(
        np.append(rect.cll + rect.sidelen / 2, z_cen).reshape(1, -1),
        caminfo
    )
    return iso_cube(centre, region_size)


def img_to_raw(img, caminfo, crop_lim=None):
    conds = np.logical_and(
        caminfo.z_range[1] > img,
        caminfo.z_range[0] < img
    )
    indx = np.where(conds)
    zval = np.array(img[conds])
    indz = np.hstack((
        np.asarray(indx).astype(float).T,
        zval.reshape(-1, 1))
    )
    points3 = d2z_to_raw(indz, caminfo)
    if crop_lim is not None:
        conds = np.logical_and.reduce([
            -crop_lim < points3[:, 0], crop_lim > points3[:, 0],
            -crop_lim < points3[:, 1], crop_lim > points3[:, 1],
        ])
        return points3[conds, :]
    else:
        return points3


def normalize_depth(img, caminfo):
    """ normalization is based on empirical depth range """
    return np.clip(
        (img.astype(float) - caminfo.z_range[0]) /
        (caminfo.z_range[1] - caminfo.z_range[0]),
        0., 1.
    )


def resize_localizer(img, caminfo):
    """ rescale to fixed cropping size """
    img_rescale = cv2resize(
        img, (caminfo.crop_size, caminfo.crop_size))
    img_rescale = normalize_depth(img_rescale, caminfo)
    return img_rescale


def frame_size_localizer(img, caminfo):
    img_rescale = img * (caminfo.z_range[1] - caminfo.z_range[0]) + \
        caminfo.z_range[0]
    img_rescale = cv2resize(
        img_rescale, (caminfo.image_size[1], caminfo.image_size[0]))
    return img_rescale


def getbm(base_z, caminfo, base_margin=20):
    """ return margin (x, y) accroding to projective-z of MMCP.
        Args:
            base_z: base z-value in mm
            base_margin: base margin in mm
    """
    marg = np.tile(base_margin, (2, 1)) * caminfo.focal / base_z
    m = max(marg)
    return m


def clip_image_border(rect, caminfo):
    """ clip to image border """
    ctl = rect.cll
    cbr = rect.cll + rect.sidelen
    cen = rect.cll + rect.sidelen / 2
    obm = np.min([ctl, caminfo.image_size - cbr])
    if 0 > obm:
        # print(ctl, caminfo.image_size - cbr, obm, rect.sidelen)
        rect.sidelen += obm * 2
        rect.cll = cen - rect.sidelen / 2
    return rect


def voxelize_depth(img, pose_raw, step, anchor_num, caminfo):
    halflen = caminfo.crop_range
    points3 = img_to_raw(img, caminfo, halflen)
    grid = regu_grid(
        np.array([-halflen, -halflen, caminfo.z_range[0]]),
        step, halflen * 2 / step)
    pcnt = grid.fill(points3)
    cube = iso_cube(
        (np.max(pose_raw, axis=0) + np.min(pose_raw, axis=0)) / 2,
        caminfo.region_size
    )
    grid.step = anchor_num
    grid.cellen = halflen * 2 / anchor_num
    anchors = grid.prow_anchor_single(cube.cen, caminfo.region_size)
    cubecen = grid.fill([cube.cen])
    cubecen_anchors = np.append(
        cubecen.flatten(),
        anchors)
    resce = cube.dump()
    # mpplot = import_module('matplotlib.pyplot')
    # print(np.histogram(pcnt))
    # grid.show_dims()
    # cube.show_dims()
    # index = np.array(np.unravel_index(np.argmax(cubecen), cubecen.shape))
    # print(index)
    # print(grid.yank_anchor_single(
    #     index,
    #     anchors
    # ))
    # ax = mpplot.subplot(projection='3d')
    # numpts = points3.shape[0]
    # if 1000 < numpts:
    #     samid = np.random.choice(numpts, 1000, replace=False)
    #     points3_sam = points3[samid, :]
    # else:
    #     points3_sam = points3
    # ax.scatter(
    #     points3_sam[:, 0], points3_sam[:, 1], points3_sam[:, 2])
    # ax.view_init(azim=-90, elev=-60)
    # ax.set_zlabel('depth (mm)', labelpad=15)
    # corners = cube.get_corners()
    # iso_cube.draw_cube_wire(corners)
    # from mayavi import mlab
    # mlab.figure(size=(800, 800))
    # mlab.pipeline.volume(mlab.pipeline.scalar_field(pcnt))
    # mlab.pipeline.image_plane_widget(
    #     mlab.pipeline.scalar_field(pcnt),
    #     plane_orientation='z_axes',
    #     slice_index=halflen)
    # np.set_printoptions(precision=4)
    # mlab.outline()
    # mpplot.show()
    return pcnt, cubecen_anchors, resce


def generate_anchors_2d(img, pose_raw, anchor_num, caminfo):
    """ two sections concatenated:
        - positive probability,
        - parameters
    """
    lattice = latice_image(
        np.array(img.shape).astype(float), anchor_num)
    cube = iso_cube(
        (np.max(pose_raw, axis=0) + np.min(pose_raw, axis=0)) / 2,
        caminfo.region_size
    )
    cen2d = raw_to_2d(cube.cen.reshape(1, -1), caminfo)
    rect = proj_cube_to_rect(cube, caminfo.region_size, caminfo)
    pcnt = lattice.fill(cen2d)  # only one-shot here
    anchors = lattice.prow_anchor_single(cen2d, rect.sidelen / 2)
    # import matplotlib.pyplot as mpplot
    # print(cen2d, rect.sidelen / 2)
    # index = np.array(np.unravel_index(np.argmax(pcnt), pcnt.shape))
    # print(lattice.yank_anchor_single(
    #     index,
    #     anchors
    # ))
    # mpplot.imshow(img, cmap='bone')
    # rect.show_dims()
    # rect.draw()
    # mpplot.show()
    resce = cube.dump()
    return np.append(pcnt.flatten(), anchors), resce


def direc_belief(pcnt):
    size = pcnt.shape[0]
    phi = np.ones_like(pcnt)
    z0front = np.ones((size, size)) * size
    for index in np.transpose(np.where(0 < pcnt)):
        if z0front[index[0], index[1]] > index[2]:
            z0front[index[0], index[1]] = index[2]
    # print(z0front)
    zrange = np.ones((size, size, size))
    zrange[:, :, np.arange(size)] *= np.arange(size)
    # print(zrange[..., 2])
    for z in range(size):
        phi[zrange[..., z] == z0front, z] = 0
        phi[zrange[..., z] > z0front, z] = -1
    # print(phi[..., 0])
    # print(phi[..., 1])
    # print(phi[..., 2])
    # print(phi[..., 3])
    # print(phi[..., 4])
    # print(phi[..., 5])
    bef = skfmm.distance(phi, dx=1e-1, narrow=0.3)
    return bef


def trunc_belief(pcnt):
    pcnt_r = np.copy(pcnt)
    befs = []
    for spi in range(3):
        pcnt_r = np.rollaxis(pcnt_r, -1)
        befs.append(direc_belief(pcnt_r))
    return np.stack(befs, axis=3)


def prop_dist(pcnt):
    phi = np.ones_like(pcnt)
    phi[1e-4 < pcnt] = 0
    tdf = skfmm.distance(phi, dx=1e-1, narrow=0.2)
    return tdf


def fill_grid(img, pose_raw, step, caminfo):
    cube = iso_cube(
        (np.max(pose_raw, axis=0) + np.min(pose_raw, axis=0)) / 2,
        caminfo.region_size
    )
    points3_pick = cube.pick(img_to_raw(img, caminfo))
    points3_trans = cube.transform_center_shrink(points3_pick)
    grid = regu_grid()
    grid.from_cube(cube, step)
    pcnt = grid.fill(points3_trans)
    resce = cube.dump()
    return pcnt, resce


def proj_ortho3(img, pose_raw, caminfo):
    cube = iso_cube(
        (np.max(pose_raw, axis=0) + np.min(pose_raw, axis=0)) / 2,
        caminfo.region_size
    )
    points3_pick = cube.pick(img_to_raw(img, caminfo))
    points3_trans = cube.transform_center_shrink(points3_pick)
    img_l = []
    for spi in range(3):
        coord, depth = cube.project_pca(points3_trans, roll=spi)
        img_crop = cube.print_image(coord, depth, caminfo.crop_size)
        # img_l.append(
        #     cv2resize(img_crop, (caminfo.crop_size, caminfo.crop_size))
        # )
        img_l.append(
            img_crop
        )
        # pose2d, _ = cube.project_pca(pose_trans, roll=spi, sort=False)
    # resce = np.concatenate((
    #     np.array([float(caminfo.crop_size) / cube.get_sidelen()]),
    #     np.ones(2) * cube.get_sidelen(),
    #     cube.dump()
    # ))
    # resce = np.append(
    #     float(caminfo.crop_size) / cube.get_sidelen(),
    #     cube.cen
    # )
    # resce = np.concatenate((
    #     resce,
    #     cube.evecs.flatten()))
    img_crop_resize = np.stack(img_l, axis=2)
    resce = cube.dump()
    return img_crop_resize, resce


def get_rect3(cube, caminfo):
    """ return a rectangle with margin that 3d points
        NOTE: there is still a perspective problem
    """
    cen = raw_to_2d(cube.cen.reshape(1, -1), caminfo).flatten()
    rect = iso_rect(
        cen - cube.sidelen,
        cube.get_sidelen()
    )
    rect = clip_image_border(rect, caminfo)
    return rect


def collect_heatmap(pose_raw, resce):
    cube = iso_cube()
    cube.load(resce)
    points3_trans = cube.transform_center_shrink(pose_raw)
    coord, depth = cube.project_pca(points3_trans, sort=False)
    img_crop_resize = cube.print_image(
        coord, depth, caminfo.crop_size)


def crop_resize_pca(img, pose_raw, caminfo):
    # cube = iso_cube()
    # cube.build(pose_raw, 0.6)
    cube = iso_cube(
        (np.max(pose_raw, axis=0) + np.min(pose_raw, axis=0)) / 2,
        caminfo.region_size
    )
    points3_pick = cube.pick(img_to_raw(img, caminfo))
    points3_trans = cube.transform_center_shrink(points3_pick)
    coord, depth = cube.project_pca(points3_trans, sort=False)
    # x = np.arange(-1, 1, 0.5)
    # y = np.stack([x, x, -x]).T
    # print(y)
    # coord, depth = cube.project_pca(y)
    # print(coord)
    # print(depth)
    img_crop_resize = cube.print_image(
        coord, depth, caminfo.crop_size)
    # mpplot = import_module('matplotlib.pyplot')
    # mpplot.imshow(img_crop_resize, cmap='bone')
    # mpplot.show()
    resce = cube.dump()
    return img_crop_resize, resce


def get_rect2(cube, caminfo):
    rect = proj_cube_to_rect(cube, caminfo.region_size, caminfo)
    rect = clip_image_border(rect, caminfo)
    return rect


# def get_rect(pose2d, caminfo, bm=0.6):
#     """ return a rectangle with margin that contains 2d point set
#     """
#     rect = iso_rect()
#     rect.build(pose2d, bm)
#     rect = clip_image_border(rect, caminfo)
#     return rect


def crop_resize(img, pose_raw, caminfo):
    # cube.build(pose_raw)
    cube = iso_cube(
        (np.max(pose_raw, axis=0) + np.min(pose_raw, axis=0)) / 2,
        caminfo.region_size
    )
    rect = get_rect2(cube, caminfo)
    # import matplotlib.pyplot as mpplot
    # mpplot.imshow(img, cmap='bone')
    # rect.show_dims()
    # rect.draw()
    # rect = proj_cube_to_rect(cube, caminfo.region_size, caminfo)
    # rect.show_dims()
    # cube.show_dims()
    # recover_from_rect(rect, caminfo.region_size, caminfo).show_dims()
    # mpplot.show()
    cll_i = np.floor(rect.cll).astype(int)
    sizel = np.floor(rect.sidelen).astype(int)
    img_crop = img[
        cll_i[0]:cll_i[0] + sizel,
        cll_i[1]:cll_i[1] + sizel,
    ]
    img_crop_resize = resize_localizer(img_crop, caminfo)
    resce = np.concatenate((
        cube.dump(),
        rect.dump(),
    ))
    return img_crop_resize, resce
