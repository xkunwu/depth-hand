import numpy as np
import matplotlib.pyplot as mpplot
import skfmm
import scipy.ndimage as ndimage
from cv2 import resize as cv2resize
from utils.iso_boxes import iso_rect
from utils.iso_boxes import iso_cube
from utils.regu_grid import regu_grid
from utils.regu_grid import latice_image


def d2z_to_raw(p2z, caminfo):
    """ reproject 2d poses to 3d.
        p2z: nx3 array, 2d position and real z
    """
    p2z = p2z.astype(float)
    pose2d = p2z[:, 0:2]
    pose_z = np.array(p2z[:, 2]).reshape(-1, 1)
    pose2d = pose2d[:, ::-1]  # image coordinates: reverse x, y
    pose3d = (pose2d - caminfo.centre) / caminfo.focal * pose_z
    return np.hstack((pose3d, pose_z))


def raw_to_2dz(points3, caminfo):
    """ project 3D point onto image plane using camera info
        Args:
            points3: nx3 array, raw input in real world coordinates
    """
    points3 = points3.astype(float)
    pose_z = points3[:, 2]
    pose2d = points3[:, 0:2] / pose_z.reshape(-1, 1) * caminfo.focal + caminfo.centre
    pose2d = pose2d[:, ::-1]  # image coordinates: reverse x, y
    return pose2d, pose_z


def raw_to_2d(points3, caminfo):
    pose2d, _ = raw_to_2dz(points3, caminfo)
    return pose2d


def img_to_raw(img, caminfo, crop_lim=None):
    conds = np.logical_and(
        caminfo.z_range[1] > img,
        caminfo.z_range[0] < img
    )
    indx = np.where(conds)
    zval = np.array(img[conds])
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


def prop_edt2(image_crop, pose_raw, cube, caminfo, roll=0):
    from scipy.ndimage.morphology import distance_transform_edt
    hmap_size = caminfo.hmap_size
    istep = 2. / hmap_size
    scale = int(image_crop.shape[0] / hmap_size)
    # image_hmap = image_crop
    if 1 == scale:
        image_hmap = image_crop
    else:
        image_hmap = image_crop[::scale, ::scale]  # downsampling
    mask = (1e-4 > image_hmap)
    masked_edt = np.ma.masked_array(image_hmap, mask)
    edt_out = distance_transform_edt(mask) * istep
    edt = image_hmap - edt_out
    pose3d = cube.transform_center_shrink(pose_raw)
    pose2d, _ = cube.project_ortho(pose3d, roll=roll, sort=False)
    pose2d = np.floor(pose2d * hmap_size).astype(int)
    edt_l = []
    for pose in pose2d:
        val = edt[pose[0], pose[1]]
        if 0 > val:
            ring = np.ones_like(image_hmap)
            ring[pose[0], pose[1]] = 0
            ring = - distance_transform_edt(ring)
            ring = np.ma.masked_array(
                ring, np.logical_and((val > ring), mask))
            ring = np.max(ring) - ring
            ring[~mask] = 0
            phi = image_hmap + ring + 1
            # phi = np.ma.masked_array(
            #     # distance_transform_edt(0 < edt - val) + istep,
            #     edt - val + istep,
            #     (val > edt))
        else:
            phi = masked_edt.copy()
        phi[pose[0], pose[1]] = 0.
        df = skfmm.distance(phi, dx=1e-1)
        df_max = np.max(df)
        if 1e-2 < df_max:
            df = (df_max - df) / df_max
        df[mask] = 0.
        df[1. < df] = 0.  # outside isolated region have value > 1
        # if 1 != scale:
        #     df = df[::scale, ::scale]  # downsampling
        edt_l.append(df)
    return np.stack(edt_l, axis=2)


def prop_ov3edt2(ortho3_crop, pose_raw, cube, caminfo, roll=0):
    crop_size = caminfo.crop_size
    images = ortho3_crop.reshape((crop_size, crop_size, 3))
    edt_l = []
    for dd in range(3):
        edt_l.append(prop_edt2(
            images[..., dd], pose_raw, cube, caminfo, roll=dd
        ))
    return np.concatenate(edt_l, axis=2)


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


def to_vxhit(img, cube, caminfo):
    step = caminfo.crop_size
    points3_pick = cube.pick(img_to_raw(img, caminfo))
    grid = regu_grid()
    grid.from_cube(cube, step)
    pcnt3 = grid.hit(points3_pick)
    return pcnt3


def to_pcnt3(img, cube, caminfo):
    step = caminfo.crop_size
    points3_pick = cube.pick(img_to_raw(img, caminfo))
    grid = regu_grid()
    grid.from_cube(cube, step)
    pcnt3 = grid.fill(points3_pick)
    return pcnt3


def to_ortho3(img, cube, caminfo, sort=False):
    points3_pick = cube.pick(img_to_raw(img, caminfo))
    points3_norm = cube.transform_center_shrink(points3_pick)
    img_l = []
    for spi in range(3):
        coord, depth = cube.project_ortho(points3_norm, roll=spi)
        img_crop = cube.print_image(coord, depth, caminfo.crop_size)
        img_l.append(
            img_crop
        )
    img_ortho3 = np.stack(img_l, axis=2)
    return img_ortho3


def to_clean(img, cube, caminfo, sort=False):
    # mpplot.imshow(img, cmap=mpplot.cm.bone_r)
    # mpplot.show()
    points3_pick = cube.pick(img_to_raw(img, caminfo))
    coord, depth = cube.raw_to_unit(points3_pick, sort=sort)
    img_clean = cube.print_image(
        coord, depth, caminfo.crop_size)
    # mpplot.imshow(img_clean, cmap=mpplot.cm.bone_r)
    # mpplot.show()
    return img_clean


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


def get_rect2(cube, caminfo):
    rect = cube.proj_to_rect(caminfo.region_size, raw_to_2d, caminfo)
    rect = clip_image_border(rect, caminfo)
    return rect


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


def to_crop2(img, cube, caminfo):
    rect = get_rect2(cube, caminfo)
    cll_i = np.floor(rect.cll).astype(int)
    sizel = np.floor(rect.sidelen).astype(int)
    img_crop = img[
        cll_i[0]:cll_i[0] + sizel,
        cll_i[1]:cll_i[1] + sizel,
    ]
    img_crop2 = resize_normalize(img_crop, caminfo)
    return img_crop2
