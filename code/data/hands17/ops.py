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


def softmax(x):
    '''
    # producing 2d array
    # each row should be a probability
    >>> res = softmax(np.array([0, 200, 10]))
    >>> np.sum(res)
    1.0
    >>> np.all(np.abs(res - np.array([0, 1, 0])) < 0.0001)
    True
    >>> res = softmax(np.array([[0, 200, 10], [0, 10, 200], [200, 0, 10]]))
    >>> np.sum(res, axis=1)
    array([ 1.,  1.,  1.])
    >>> res = softmax(np.array([[0, 200, 10], [0, 10, 200]]))
    >>> np.sum(res, axis=1)
    array([ 1.,  1.])
    '''
    if x.ndim == 1:
        x = x.reshape((1, -1))
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))


def raw_to_pca(points3, resce=np.array([1, 0, 0, 0])):
    cube = iso_cube()
    cube.load(resce)
    # return cube.transform_center_shrink(points3)
    return cube.transform_to_center(points3)


def pca_to_raw(points3, resce=np.array([1, 0, 0, 0])):
    cube = iso_cube()
    cube.load(resce)
    # return cube.transform_expand_move(points3)
    return cube.transform_add_center(points3)


def raw_to_local(points3, resce=np.array([1, 0, 0, 0])):
    cube = iso_cube()
    cube.load(resce)
    # return cube.transform_center_shrink(points3)
    return cube.transform_to_center(points3)


def local_to_raw(points3, resce=np.array([1, 0, 0, 0])):
    cube = iso_cube()
    cube.load(resce)
    # return cube.transform_expand_move(points3)
    return cube.transform_add_center(points3)


def d2z_to_raw(p2z, caminfo, resce=np.array([1, 0, 0])):
    """ reproject 2d poses to 3d.
        p2z: nx3 array, 2d position and real z
    """
    p2z = p2z.astype(float)
    pose2d = p2z[:, 0:2] / resce[0] + resce[1:3]
    pose_z = np.array(p2z[:, 2]).reshape(-1, 1)
    pose2d = pose2d[:, ::-1]  # image coordinates: reverse x, y
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
    pose2d = pose2d[:, ::-1]  # image coordinates: reverse x, y
    return (pose2d - resce[1:3]) * resce[0], pose_z


def raw_to_2d(points3, caminfo, resce=np.array([1, 0, 0])):
    pose2d, _ = raw_to_2dz(points3, caminfo, resce)
    # print(points3)
    # print(pose2d)
    return pose2d


def raw_to_heatmap2(pose_raw, cube, caminfo):
    """ 2d heatmap for each joint """
    hmap_size = caminfo.hmap_size
    # return np.zeros((hmap_size, hmap_size, 21))
    coord, depth = cube.raw_to_unit(pose_raw)
    img_l = []
    for c, d in zip(coord, depth):
        img = cube.print_image(
            c.reshape(1, -1), np.array([d]), hmap_size)
        img = ndimage.gaussian_filter(  # still a probability
            img, sigma=0.8)
        # img /= np.max(img)
        # mpplot = import_module('matplotlib.pyplot')
        # mpplot.imshow(img, cmap=mpplot.cm.bone_r)
        # mpplot.show()
        img_l.append(img)
    return np.stack(img_l, axis=2)


def raw_to_udir2(image_crop, pose_raw, cube, caminfo):
    from numpy import linalg
    step = caminfo.hmap_size
    scale = int(image_crop.shape[0] / step)
    if 1 == scale:
        image_hmap = image_crop
    else:
        image_hmap = image_crop[::scale, ::scale]  # downsampling
    coord, depth = cube.image_to_unit(image_hmap)
    depth_raw = cube.unit_to_raw(coord, depth)
    offset = pose_raw[:, None] - depth_raw  # JxPx3
    offlen = linalg.norm(offset, axis=-1)  # offset norm
    theta = caminfo.region_size * 2  # maximal - cube size
    offlen = np.clip(offlen, 1e-2, theta)  # 0.01mm minimum
    unit_off = offset / offlen[:, :, None]
    offlen = (theta - offlen) / theta  # inverse propotional
    num_j = caminfo.join_num
    olmap = np.zeros((step, step, num_j))
    uomap = np.zeros((step, step, num_j, 3))
    xx = np.floor(coord[:, 0] * step).astype(int)
    yy = np.floor(coord[:, 1] * step).astype(int)
    for jj in range(num_j):
        olmap[xx, yy, jj] = offlen[jj, :]
        for dd in range(3):
            uomap[xx, yy, jj, dd] = unit_off[jj, :, dd]
    return np.concatenate(
        (olmap, uomap.reshape((step, step, 3 * num_j))),
        axis=-1)


def raw_to_offset(image_crop, pose_raw, cube, caminfo):
    """ offset map from depth to each joint
        Args:
            img: should be size of 128
    """
    hmap_size = caminfo.hmap_size
    scale = int(image_crop.shape[0] / hmap_size)
    if 1 == scale:
        image_hmap = image_crop
    else:
        image_hmap = image_crop[::scale, ::scale]  # downsampling
    coord, depth = cube.image_to_unit(image_hmap)
    depth_raw = cube.unit_to_raw(coord, depth)
    # points3_pick = cube.pick(img_to_raw(image_crop, caminfo))
    # depth_raw = cube.transform_center_shrink(points3_pick)
    from numpy import linalg
    omap_l = []
    hmap_l = []
    umap_l = []
    theta = caminfo.region_size * 2  # maximal - cube size
    # theta = 90  # used for illustration
    for joint in pose_raw:
        offset = joint - depth_raw  # offset in raw 3d
        dist = linalg.norm(offset, axis=1)  # offset norm
        if np.min(dist) > theta:
            # due to occlution, we cannot use small radius
            print(np.min(dist))
        #     import data.hands17.draw as data_draw
        #     from colour import Color
        #     mpplot.subplots(nrows=1, ncols=2)
        #     ax = mpplot.subplot(1, 2, 1)
        #     ax.imshow(image_crop, cmap=mpplot.cm.bone_r)
        #     data_draw.draw_pose2d(
        #         ax, caminfo,
        #         raw_to_2d(pose_raw, caminfo))
        #     ax.axis('off')
        #     ax = mpplot.subplot(1, 2, 2, projection='3d')
        #     points3_trans = points3_pick
        #     numpts = points3_trans.shape[0]
        #     if 1000 < numpts:
        #         points3_trans = points3_trans[
        #             np.random.choice(numpts, 1000, replace=False), :]
        #     ax.scatter(
        #         points3_trans[:, 0], points3_trans[:, 1], points3_trans[:, 2],
        #         color=Color('lightsteelblue').rgb)
        #     data_draw.draw_raw3d_pose(ax, caminfo, pose_raw)
        #     ax.view_init(azim=-120, elev=-150)
        #     mpplot.show()
        # else:
        #     continue

        valid_id = np.where(np.logical_and(
            1e-1 < dist,  # remove sigular point
            theta > dist  # limit support within theta
        ))
        offset = offset[valid_id]
        dist = dist[valid_id]
        unit_off = offset / np.tile(dist, [3, 1]).T  # unit offset
        dist = (theta - dist) / theta  # inverse propotional
        coord_valid = coord[valid_id]
        for dim in range(3):
            om = cube.print_image(coord_valid, offset[:, dim], hmap_size)
            omap_l.append(om)
            um = cube.print_image(coord_valid, unit_off[:, dim], hmap_size)
            umap_l.append(um)
            # mpplot.subplot(3, 3, 4 + dim)
            # mpplot.imshow(om, cmap=mpplot.cm.bone_r)
            # mpplot.subplot(3, 3, 7 + dim)
            # mpplot.imshow(um, cmap=mpplot.cm.bone_r)
        hm = cube.print_image(coord_valid, dist, hmap_size)
        hmap_l.append(hm)
        # print(np.histogram(dist, range=(1e-4, np.max(dist))))
        # mpplot.subplot(3, 3, 1)
        # mpplot.imshow(hm, cmap=mpplot.cm.bone_r)
        # mpplot.subplot(3, 3, 3)
        # mpplot.imshow(img, cmap=mpplot.cm.jet)
        # mpplot.show()
    offset_map = np.stack(omap_l, axis=2)
    olmap = np.stack(hmap_l, axis=2)
    uomap = np.stack(umap_l, axis=2)
    return offset_map, olmap, uomap


def udir2_to_raw(
    olmap, uomap, image_crop,
        cube, caminfo, nn=5):
    from sklearn.preprocessing import normalize
    hmap_size = caminfo.hmap_size
    num_joint = olmap.shape[2]
    theta = caminfo.region_size * 2
    pose_out = np.empty([num_joint, 3])
    scale = int(image_crop.shape[0] / hmap_size)
    if 1 == scale:
        image_hmap = image_crop
    else:
        image_hmap = image_crop[::scale, ::scale]  # downsampling
    for joint in range(num_joint):
        # restore from 3d
        om = olmap[..., joint]
        # hm[np.where(1e-2 > image_hmap)] = 0  # mask out void is wrong - joint not on the surface
        top_id = om.argpartition(-nn, axis=None)[-nn:]  # top elements
        x3, y3 = np.unravel_index(top_id, om.shape)
        conf = om[x3, y3]
        dist = theta - om[x3, y3] * theta  # inverse propotional
        uom = uomap[..., 3 * joint:3 * (joint + 1)]
        unit_off = uom[x3, y3, :]
        unit_off = normalize(unit_off, norm='l2')
        offset = unit_off * np.tile(dist, [3, 1]).T
        p0 = cube.unit_to_raw(
            np.vstack([x3, y3]).astype(float).T / hmap_size,
            image_hmap[x3, y3])
        pred3 = p0 + offset
        pred32 = np.sum(
            pred3 * np.tile(conf, [3, 1]).T, axis=0
        ) / np.sum(conf)
        pose_out[joint, :] = pred32
    return pose_out


def offset_to_raw(
    hmap2, olmap, uomap, image_crop,
        cube, caminfo, nn=5):
    """ recover 3d from weight avarage """
    from sklearn.preprocessing import normalize
    hmap_size = caminfo.hmap_size
    num_joint = olmap.shape[2]
    theta = caminfo.region_size * 2
    pose_out = np.empty([num_joint, 3])
    scale = int(image_crop.shape[0] / hmap_size)
    if 1 == scale:
        image_hmap = image_crop
    else:
        image_hmap = image_crop[::scale, ::scale]  # downsampling
    for joint in range(num_joint):
        # restore from 3d
        hm = hmap2[..., joint]
        om = olmap[..., joint]
        # hm[np.where(1e-2 > image_hmap)] = 0  # mask out void is wrong - joint not on the surface
        hm[np.where(1e-2 > om)] = 0  # consistent 2d-3d prediction
        top_id = hm.argpartition(-nn, axis=None)[-nn:]  # top elements
        x3, y3 = np.unravel_index(top_id, hm.shape)
        conf = hm[x3, y3]
        dist = theta - om[x3, y3] * theta  # inverse propotional
        uom = uomap[..., 3 * joint:3 * (joint + 1)]
        unit_off = uom[x3, y3, :]
        unit_off = normalize(unit_off, norm='l2')
        offset = unit_off * np.tile(dist, [3, 1]).T
        p0 = cube.unit_to_raw(
            np.vstack([x3, y3]).astype(float).T / hmap_size,
            image_hmap[x3, y3])
        pred3 = p0 + offset
        pred32 = np.sum(
            pred3 * np.tile(conf, [3, 1]).T, axis=0
        ) / np.sum(conf)
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
    # indz = np.hstack((
    #     np.asarray(indx).astype(float).T,
    #     zval.reshape(-1, 1))
    # )
    indz = np.vstack((
        np.asarray(indx).astype(float),
        zval
    )).T
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


def resize_normalize(img, caminfo):
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
    # ax.view_init(azim=-90, elev=-75)
    # ax.set_zlabel('depth (mm)', labelpad=15)
    # corners = cube.get_corners()
    # iso_cube.draw_cube_wire(ax, corners)
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
    rect = cube.proj_to_rect(caminfo.region_size, raw_to_2d, caminfo)
    pcnt = lattice.fill(cen2d)  # only one-shot here
    anchors = lattice.prow_anchor_single(cen2d, rect.sidelen / 2)
    # import matplotlib.pyplot as mpplot
    # print(cen2d, rect.sidelen / 2)
    # index = np.array(np.unravel_index(np.argmax(pcnt), pcnt.shape))
    # print(lattice.yank_anchor_single(
    #     index,
    #     anchors
    # ))
    # mpplot.imshow(img, cmap=mpplot.cm.bone_r)
    # rect.show_dims()
    # rect.draw(ax)
    # mpplot.show()
    resce = cube.dump()
    return np.append(pcnt.flatten(), anchors), resce


def direc_belief(pcnt):
    """ not optimized! """
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


def prop_edt2(image_crop, pose_raw, cube, caminfo):
    from scipy.ndimage.morphology import distance_transform_edt
    hmap_size = caminfo.hmap_size
    scale = int(image_crop.shape[0] / hmap_size)
    if 1 == scale:
        image_hmap = image_crop
    else:
        image_hmap = cv2resize(image_crop, (hmap_size, hmap_size))
    mask = (1e-4 > image_hmap)
    masked_edt = np.ma.masked_array(
        distance_transform_edt(image_hmap),
        mask)
    pose3d = cube.transform_center_shrink(pose_raw)
    pose2d, _ = cube.project_ortho(pose3d, roll=0, sort=False)
    pose2d = np.floor(pose2d * hmap_size).astype(int)
    vol_l = []
    for pose in pose2d:
        phi = masked_edt.copy()
        phi[pose[0], pose[1]] = 0.
        df = skfmm.distance(phi, dx=1e-1)
        df_max = np.max(df)
        df = (df_max - df) / df_max
        df[mask] = 0.
        df[1. < df] = 0.  # outside isolated region have value > 1
        vol_l.append(df)
    return np.stack(vol_l, axis=2)


def prop_edt3(vxhit, pose_raw, cube, caminfo):
    from scipy.ndimage.morphology import distance_transform_edt
    step = caminfo.hmap_size
    scale = int(vxhit.shape[0] / step)
    vol_shape = (step, step, step)
    if 1 == scale:
        vxhit_hmap = vxhit
    else:
        vxhit_hmap = np.zeros(vol_shape)
        for i0 in np.mgrid[0:scale, 0:scale, 0:scale].reshape(3, -1).T:
            vxhit_hmap += vxhit[i0[0]::scale, i0[1]::scale, i0[2]::scale]
        # print(np.sum(vxhit_hmap) - np.sum(vxhit))
    mask = (1e-4 > vxhit_hmap)
    masked_edt = np.ma.masked_array(
        distance_transform_edt(vxhit_hmap),
        mask)
    # masked_edt = np.ma.masked_array(
    #     vxhit_hmap,
    #     mask)
    grid = regu_grid()
    grid.from_cube(cube, step)
    indices = grid.putit(pose_raw)
    vol_l = []
    for index in indices:
        # phi = np.zeros_like(masked_edt)
        # phi[index[0], index[1], index[2]] = 1.
        phi = masked_edt.copy()
        phi[index[0], index[1], index[2]] = 0.
        df = skfmm.distance(phi, dx=1e-1)
        df_max = np.max(df)
        df = (df_max - df) / df_max
        df[mask] = 0.
        vol_l.append(df)
    return np.stack(vol_l, axis=3)


def prop_dist(pcnt):
    phi = np.ones_like(pcnt)
    phi[1e-4 < pcnt] = 0
    tdf = skfmm.distance(phi, dx=1e-1, narrow=0.2)
    return tdf


def to_pcnt3(img, cube, caminfo):
    step = caminfo.crop_size
    points3_pick = cube.pick(img_to_raw(img, caminfo))
    grid = regu_grid()
    grid.from_cube(cube, step)
    pcnt3 = grid.fill(points3_pick)
    return pcnt3


def fill_grid(img, pose_raw, caminfo):
    cube = iso_cube(
        (np.max(pose_raw, axis=0) + np.min(pose_raw, axis=0)) / 2,
        caminfo.region_size
    )
    pcnt3 = to_pcnt3(img, cube, caminfo)
    resce = cube.dump()
    return pcnt3, resce


def to_vxhit(img, cube, caminfo):
    step = caminfo.crop_size
    points3_pick = cube.pick(img_to_raw(img, caminfo))
    grid = regu_grid()
    grid.from_cube(cube, step)
    pcnt3 = grid.hit(points3_pick)
    return pcnt3


def voxel_hit(img, pose_raw, step, caminfo):
    cube = iso_cube(
        (np.max(pose_raw, axis=0) + np.min(pose_raw, axis=0)) / 2,
        caminfo.region_size
    )
    pcnt3 = to_vxhit(img, cube, caminfo)
    resce = cube.dump()
    return pcnt3, resce


def raw_to_vxoff_flat(vxcnt, pose_raw, cube, caminfo):
    """ offset map from voxel center to each joint
    """
    step = caminfo.hmap_size
    grid = regu_grid()
    grid.from_cube(cube, step)
    # pose_raw = grid.voxen(np.array([[0, 0, 0], [1, 1, 1]]))
    # print(pose_raw)
    voxcens = grid.voxen(
        np.mgrid[0:step, 0:step, 0:step].reshape(3, -1).T)
    # print(voxcens)
    offset = pose_raw - voxcens[:, None]  # (S*S*S)xJx3
    return offset
    # return offset.reshape((step ** 3, -1))


def _void_id(vxcnt, step):
    scale = int(vxcnt.shape[0] / step)
    vol_shape = (step, step, step)
    if 1 == scale:
        vxcnt_hmap = vxcnt
    else:
        # # vxcnt_hmap = vxcnt[::scale, ::scale, ::scale]
        # from itertools import product
        # r = [0, 1]
        # vxcnt_hmap = np.zeros(vol_shape)
        # for i0 in np.array(list(product(r, r, r))):
        #     vxcnt_hmap += vxcnt[i0[0]::2, i0[1]::2, i0[2]::2]
        vxcnt_hmap = np.zeros(vol_shape)
        for i0 in np.mgrid[0:scale, 0:scale, 0:scale].reshape(3, -1).T:
            vxcnt_hmap += vxcnt[i0[0]::scale, i0[1]::scale, i0[2]::scale]
        # print(np.sum(vxcnt_hmap) - np.sum(vxcnt))
    void_id = (1e-2 > vxcnt_hmap.ravel())
    return void_id


def raw_to_vxoff(vxcnt, pose_raw, cube, caminfo):
    """ offset map from voxel center to each joint
    """
    step = caminfo.hmap_size
    offset = raw_to_vxoff_flat(vxcnt, pose_raw, cube, caminfo)
    void_id = _void_id(vxcnt, step)
    offset[void_id] = 0
    return offset.reshape(step, step, step, -1)


def raw_to_vxudir(vxcnt, pose_raw, cube, caminfo):
    """ offset map from voxel center to each joint
    """
    from numpy import linalg
    step = caminfo.hmap_size
    offset = raw_to_vxoff_flat(vxcnt, pose_raw, cube, caminfo)
    # print(offset)
    offlen = linalg.norm(offset, axis=-1)  # offset norm
    # print(offlen)
    theta = caminfo.region_size * 2  # maximal - cube size
    offlen = np.clip(offlen, 1e-2, theta)  # 0.01mm minimum
    # invalid_id = np.logical_or(
    #     1e-2 > offlen,  # remove sigular point
    #     theta < offlen,  # limit support within theta
    # )
    # print(invalid_id)
    # offset[invalid_id, ...] = 0
    # print(offset)
    # offlen[invalid_id] = theta
    # print(offlen)
    unit_off = offset / offlen[:, :, None]
    # print(unit_off)
    void_id = _void_id(vxcnt, step)
    unit_off[void_id, ...] = 0
    offlen[void_id] = theta
    offlen = (theta - offlen) / theta  # inverse propotional
    return np.concatenate(
        (offlen.reshape(step, step, step, -1),
            unit_off.reshape(step, step, step, -1)),
        axis=-1)


def vxudir_to_raw(vxhit, vxudir, cube, caminfo, nn=5):
    """ recover 3d from weight avarage """
    from sklearn.preprocessing import normalize
    step = caminfo.hmap_size
    theta = caminfo.region_size * 2
    grid = regu_grid()
    grid.from_cube(cube, step)
    num_joint = caminfo.join_num
    offlen = vxudir[..., :num_joint].reshape(-1, num_joint).T
    top_id = np.argpartition(offlen, -nn, axis=1)[:, -nn:]  # top elements
    conf3 = np.take(offlen, top_id)
    voxcens = grid.voxen(
        np.mgrid[0:step, 0:step, 0:step].reshape(3, -1).T)
    pose_out = np.empty((num_joint, 3))
    for jj in range(num_joint):
        unit_off = vxudir[..., num_joint + 3 * jj:num_joint + 3 * (jj + 1)].reshape(-1, 3)
        unit_off = normalize(unit_off[top_id[jj], :])
        d = theta - offlen[jj, top_id[jj]] * theta
        p0 = voxcens[top_id[jj], :]
        pred3 = p0 + unit_off * d[:, None]
        c = conf3[jj]
        pred32 = np.sum(
            pred3 * c[:, None], axis=0
        ) / np.sum(c)
        pose_out[jj, :] = pred32

    # print(conf3.shape)
    # unit = vxudir[..., num_joint:].reshape(num_joint, -1, 3)
    # print(np.take(unit, top_id).shape)
    # ux = np.take(unit[..., 0], top_id)
    # uy = np.take(unit[..., 1], top_id)
    # uz = np.take(unit[..., 2], top_id)
    # unit = normalize(
    #     np.stack((ux, uy, uz), axis=2),
    #     axis=2)
    # print(unit.shape)
    # ux = np.take(voxcens[..., 0], top_id)
    # uy = np.take(voxcens[..., 1], top_id)
    # uz = np.take(voxcens[..., 2], top_id)
    # pose_pred = voxcens + offset
    # vol_shape = (step, step, step)
    # pose_out = np.empty([num_joint, 3])
    # for joint in range(num_joint):
    #     # restore from 3d
    #     hm = vxhit[..., joint]
    #     # hm = softmax(vxhit[..., joint].flatten()).flatten()
    #     # hm[np.where(1e-2 > vxcnt_hmap)] = 0  # mask out void is wrong - joint not on the surface
    #     hm[np.where(0 > hm)] = 0  # not training goal
    #     top_id = hm.argpartition(-nn, axis=None)[-nn:]  # top elements
    #     x3, y3, z3 = np.unravel_index(top_id, vol_shape)
    #     conf3 = hm[x3, y3, z3]
    #     # conf3 = hm[top_id]
    #     print(conf3)
    #     dist = olmap[x3, y3, z3, joint]
    #     dist = theta - dist * theta  # inverse propotional
    #     uom = uomap[..., 3 * joint:3 * (joint + 1)]
    #     unit_off = uom[x3, y3, z3, :]
    #     unit_off = normalize(unit_off, norm='l2')
    #     offset = unit_off * np.tile(dist, [3, 1]).T
    #     p0 = grid.voxen(np.vstack([x3, y3, z3]).astype(float).T)
    #     pred3 = p0 + offset
    #     pred32 = np.sum(
    #         pred3 * np.tile(conf3, [3, 1]).T, axis=0
    #     ) / np.sum(conf3)
        pose_out[jj, :] = pred32
    return pose_out


# def vxoff_to_raw(
#     vxhit, olmap, uomap, vxcnt,
#         cube, step, caminfo, nn=5):
#     """ recover 3d from weight avarage """
#     from sklearn.preprocessing import normalize
#     grid = regu_grid()
#     grid.from_cube(cube, step)
#     vol_shape = (step, step, step)
#     num_joint = olmap.shape[-1]
#     theta = caminfo.region_size * 2
#     pose_out = np.empty([num_joint, 3])
#     for joint in range(num_joint):
#         # restore from 3d
#         hm = vxhit[..., joint]
#         # hm = softmax(vxhit[..., joint].flatten()).flatten()
#         # hm[np.where(1e-2 > vxcnt_hmap)] = 0  # mask out void is wrong - joint not on the surface
#         hm[np.where(0 > hm)] = 0  # not training goal
#         top_id = hm.argpartition(-nn, axis=None)[-nn:]  # top elements
#         x3, y3, z3 = np.unravel_index(top_id, vol_shape)
#         conf3 = hm[x3, y3, z3]
#         # conf3 = hm[top_id]
#         print(conf3)
#         dist = olmap[x3, y3, z3, joint]
#         dist = theta - dist * theta  # inverse propotional
#         uom = uomap[..., 3 * joint:3 * (joint + 1)]
#         unit_off = uom[x3, y3, z3, :]
#         unit_off = normalize(unit_off, norm='l2')
#         offset = unit_off * np.tile(dist, [3, 1]).T
#         p0 = grid.voxen(np.vstack([x3, y3, z3]).astype(float).T)
#         pred3 = p0 + offset
#         pred32 = np.sum(
#             pred3 * np.tile(conf3, [3, 1]).T, axis=0
#         ) / np.sum(conf3)
#         pose_out[joint, :] = pred32
#     return pose_out


def raw_to_vxhit(pose_raw, cube, caminfo):
    """ 01-voxel heatmap """
    step = caminfo.hmap_size
    grid = regu_grid()
    grid.from_cube(cube, step)
    indices = grid.putit(pose_raw)
    vol_l = []
    for index in indices:
        vol = np.zeros((step, step, step))
        vol[index[0], index[1], index[2]] = 1.
        vol_l.append(vol)
    return np.stack(vol_l, axis=3)


def raw_to_vxlab(pose_raw, cube, caminfo):
    """ 01-voxel heatmap converted to labels """
    step = caminfo.hmap_size
    grid = regu_grid()
    grid.from_cube(cube, step)
    indices = grid.putit(pose_raw)
    return np.ravel_multi_index(
        indices.T, (step, step, step))


def vxlab_to_raw(vxlab, cube, caminfo):
    """ vxlab: sequential number """
    step = caminfo.hmap_size
    grid = regu_grid()
    grid.from_cube(cube, step)
    num_joint = vxlab.shape[-1]
    pose_out = np.empty([num_joint, 3])
    vol_shape = (step, step, step)
    for joint in range(num_joint):
        vh = vxlab[..., joint]
        # index = np.array(np.unravel_index(
        #     np.argmax(vh), vh.shape))
        index = np.array(np.unravel_index(
            int(vh), vol_shape))
        pose_out[joint, :] = grid.voxen(index)
    return pose_out


def to_ortho3(img, cube, caminfo, sort=False):
    points3_pick = cube.pick(img_to_raw(img, caminfo))
    points3_norm = cube.transform_center_shrink(points3_pick)
    img_l = []
    for spi in range(3):
        coord, depth = cube.project_ortho(points3_norm, roll=spi)
        img_crop = cube.print_image(coord, depth, caminfo.crop_size)
        # img_l.append(
        #     cv2resize(img_crop, (caminfo.crop_size, caminfo.crop_size))
        # )
        img_l.append(
            img_crop
        )
        # pose2d, _ = cube.project_ortho(pose_trans, roll=spi, sort=False)
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
    img_ortho3 = np.stack(img_l, axis=2)
    return img_ortho3


def proj_ortho3(img, pose_raw, caminfo, sort=False):
    cube = iso_cube(
        (np.max(pose_raw, axis=0) + np.min(pose_raw, axis=0)) / 2,
        caminfo.region_size
    )
    img_ortho3 = to_ortho3(img, cube, caminfo, sort=sort)
    resce = cube.dump()
    return img_ortho3, resce


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


def to_clean(img, cube, caminfo, sort=False):
    points3_pick = cube.pick(img_to_raw(img, caminfo))
    coord, depth = cube.raw_to_unit(points3_pick, sort=sort)
    img_clean = cube.print_image(
        coord, depth, caminfo.crop_size)
    # I0 = np.zeros((8, 8))
    # I0[0, 1] = 10.
    # I0[3, 7] = 20.
    # I0[6, 2] = 30.
    # I0[7, 3] = 40.
    # print(I0)
    # c0, d0 = cube.image_to_unit(I0)
    # I1 = cube.print_image(
    #     c0, d0, 8)
    # print(I1)
    # c1, d1 = cube.image_to_unit(I1)
    # from numpy import linalg
    # print(linalg.norm(c0 - c1), linalg.norm(d0 - d1))
    # mpplot = import_module('matplotlib.pyplot')
    # mpplot.imshow(img_clean, cmap=mpplot.cm.bone_r)
    # mpplot.show()
    return img_clean


def crop_resize_pca(img, pose_raw, caminfo, sort=False):
    # return np.zeros((caminfo.crop_size, caminfo.crop_size)), np.ones(4)
    cube = iso_cube(
        (np.max(pose_raw, axis=0) + np.min(pose_raw, axis=0)) / 2,
        caminfo.region_size
    )
    img_clean = to_clean(img, cube, caminfo, sort=sort)
    resce = cube.dump()
    return img_clean, resce


def get_rect2(cube, caminfo):
    rect = cube.proj_to_rect(caminfo.region_size, raw_to_2d, caminfo)
    rect = clip_image_border(rect, caminfo)
    return rect


# def get_rect(pose2d, caminfo, bm=0.6):
#     """ return a rectangle with margin that contains 2d point set
#     """
#     rect = iso_rect()
#     rect.build(pose2d, bm)
#     rect = clip_image_border(rect, caminfo)
#     return rect


def to_crop2(img, cube, caminfo):
    rect = get_rect2(cube, caminfo)
    # import matplotlib.pyplot as mpplot
    # mpplot.imshow(img, cmap=mpplot.cm.bone_r)
    # rect.show_dims()
    # rect.draw(ax)
    # rect = cube.proj_to_rect(caminfo.region_size, raw_to_2d, caminfo)
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
    img_crop2 = resize_normalize(img_crop, caminfo)
    return img_crop2


def crop_resize(img, pose_raw, caminfo):
    # cube.build(pose_raw)
    cube = iso_cube(
        (np.max(pose_raw, axis=0) + np.min(pose_raw, axis=0)) / 2,
        caminfo.region_size
    )
    img_crop = to_crop2(img, cube, caminfo)
    resce = cube.dump()
    return img_crop, resce
