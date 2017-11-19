import os
import sys
from importlib import import_module
import numpy as np
import skfmm
from cv2 import resize as cv2resize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
iso_rect = getattr(
    import_module('iso_boxes'),
    'iso_rect'
)
iso_aabb = getattr(
    import_module('iso_boxes'),
    'iso_aabb'
)
iso_cube = getattr(
    import_module('iso_boxes'),
    'iso_cube'
)
regu_grid = getattr(
    import_module('regu_grid'),
    'regu_grid'
)
grid_cell = getattr(
    import_module('regu_grid'),
    'grid_cell'
)


def raw_to_pca(points3, resce=np.array([1, 0, 0, 0, 1, 0, 0, 0])):
    cube = iso_cube()
    cube.load(resce)
    return cube.transform(points3) / resce[0]


def pca_to_raw(points3, resce=np.array([1, 0, 0, 0, 1, 0, 0, 0])):
    cube = iso_cube()
    cube.load(resce)
    return cube.transform_inv(points3 * resce[0])


def raw_to_local(points3, resce=np.array([1, 0, 0, 0])):
    aabb = iso_aabb()
    aabb.load(resce)
    return aabb.transform(points3) / resce[0]


def local_to_raw(points3, resce=np.array([1, 0, 0, 0])):
    aabb = iso_aabb()
    aabb.load(resce)
    return aabb.transform_inv(points3 * resce[0])


def d2z_to_raw(p2z, caminfo, resce=np.array([1, 0, 0])):
    """ reproject 2d poses to 3d.
        p2z: nx3 array, 2d position and real z
    """
    pose2d = p2z[:, 0:2] / resce[0] + resce[1:3]
    pose_z = np.array(p2z[:, 2]).reshape(-1, 1)
    pose3d = (pose2d - caminfo.centre) / caminfo.focal * pose_z
    # pose3d = (pose2d - caminfo.centre) / (-caminfo.focal[0], caminfo.focal[1]) * pose_z
    return np.hstack((pose3d, pose_z))


def img_to_raw(img, caminfo):
    conds = np.logical_and(
        caminfo.z_far > img,
        caminfo.z_near < img
    )
    indx = np.where(conds)[::-1]  # need to reverse image plane coordinates
    zval = np.array(img[conds])
    indz = np.hstack((
        np.asarray(indx).astype(float).T,
        zval.reshape(-1, 1))
    )
    points3 = d2z_to_raw(indz, caminfo)
    return points3


def raw_to_2dz(points3, caminfo, resce=np.array([1, 0, 0])):
    """ project 3D point onto image plane using camera info
        Args:
            points3: nx3 array, raw input in real world coordinates
    """
    pose_z = points3[:, 2]
    pose2d = points3[:, 0:2] / pose_z.reshape(-1, 1) * caminfo.focal + caminfo.centre
    return (pose2d - resce[1:3]) * resce[0], pose_z


def raw_to_2d(points3, caminfo, resce=np.array([1, 0, 0])):
    pose2d, _ = raw_to_2dz(points3, caminfo, resce)
    return pose2d


def normalize_depth(img, caminfo):
    return np.clip(
        (img.astype(float) - caminfo.z_near) / (caminfo.z_far - caminfo.z_near),
        0., 1.
    )


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
    # clip to image border
    ctl = rect.cll
    cbr = rect.cll + rect.len
    cen = rect.cll + rect.len / 2
    obm = np.min([ctl, caminfo.image_size - cbr])
    if 0 > obm:
        # print(ctl, caminfo.image_size - cbr, obm, rect.len)
        rect.len += obm * 2
        rect.cll = cen - rect.len / 2
    return rect


def fill_grid(img, pose_raw, step, caminfo):
    cube = iso_cube()
    cube.build(pose_raw)
    _, points3_trans = cube.pick(img_to_raw(img, caminfo))
    grid = regu_grid()
    grid.from_cube(cube, step)
    grid.fill(points3_trans)
    resce = np.append(
        cube.dump(),
        step
    )
    return grid.pcnt, resce


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


def proj_ortho3(img, pose_raw, caminfo):
    cube = iso_cube()
    cube.build(pose_raw)
    _, points3_trans = cube.pick(img_to_raw(img, caminfo))
    img_l = []
    for spi in range(3):
        coord, depth = cube.project_pca(points3_trans, roll=spi)
        img_crop = cube.print_image(coord, depth)
        img_l.append(
            cv2resize(img_crop, (caminfo.crop_size, caminfo.crop_size))
        )
        # pose2d, _ = cube.project_pca(pose_trans, roll=spi, sort=False)
    resce = np.concatenate((
        np.array([float(caminfo.crop_size) / cube.get_sidelen()]),
        np.ones(2) * cube.get_sidelen(),
        cube.dump()
    ))
    # resce = np.append(
    #     float(caminfo.crop_size) / cube.get_sidelen(),
    #     cube.cen
    # )
    # resce = np.concatenate((
    #     resce,
    #     cube.evecs.flatten()))
    img_crop_resize = np.stack(img_l, axis=2)
    return img_crop_resize, resce


def crop_resize_pca(img, pose_raw, caminfo):
    cube = iso_cube()
    cube.build(pose_raw)
    points3, _ = cube.pick(img_to_raw(img, caminfo))
    points2, pose_z = raw_to_2dz(points3, caminfo)
    rect = iso_rect()
    rect.build(points2, 0.5)
    rect = clip_image_border(rect, caminfo)
    conds = rect.pick(points2)
    img_crop = rect.print_image(points2[conds, :], pose_z[conds])
    img_crop_resize = cv2resize(img_crop, (caminfo.crop_size, caminfo.crop_size))
    img_crop_resize = (img_crop_resize.astype(float) - np.min(img_crop_resize)) /\
        (np.max(img_crop_resize) - np.min(img_crop_resize))
    resce = np.concatenate((
        np.array([float(caminfo.crop_size) / rect.len]),
        rect.cll,
        cube.dump()
    ))
    return img_crop_resize, resce


def get_rect2(aabb, caminfo, m=0.2):
    cen = aabb.cll + aabb.len / 2
    c3a = np.array([
        np.append(cen[:2] - aabb.len / 2, aabb.cll[2]),
        np.append(cen[:2] + aabb.len / 2, aabb.cll[2]),
        # aabb.cll,
        # np.append(aabb.cll[:2] + aabb.len, aabb.cll[2]),
        # aabb.cll + aabb.len / 2
    ])
    c2a = raw_to_2d(c3a, caminfo)
    cll = c2a[0, :]
    ctr = c2a[1, :]
    rect = iso_rect(cll, np.max(ctr - cll), m)
    rect = clip_image_border(rect, caminfo)
    return rect


def crop_resize(img, pose_raw, caminfo):
    aabb = iso_aabb()
    aabb.build(pose_raw)
    rect = get_rect2(aabb, caminfo)
    cll_i = np.floor(rect.cll).astype(int)
    bl_i = np.floor(rect.len).astype(int)
    img_crop = img[
        cll_i[1]:cll_i[1] + bl_i,
        cll_i[0]:cll_i[0] + bl_i,
    ]

    # points2 = raw_to_2d(pose_raw, caminfo)
    # rect = get_rect(points2, caminfo)
    # cll = np.floor(rect.cll).astype(int)
    # bl = np.floor(rect.len).astype(int)
    # img_crop = img[
    #     cll[0]:cll[0] + bl,
    #     cll[1]:cll[1] + bl,
    # ]

    # rect = get_rect3(
    #     pose_raw, caminfo
    # )
    # points3 = img_to_raw(img, caminfo)
    # points2, pose_z = raw_to_2dz(points3, caminfo)
    # conds = rect.pick(points2)
    # img_crop = rect.print_image(points2[conds, :], pose_z[conds])

    img_crop_resize = cv2resize(img_crop, (caminfo.crop_size, caminfo.crop_size))
    img_crop_resize = normalize_depth(img_crop_resize, caminfo)
    # cen = (np.min(pose_raw, axis=0) + np.max(pose_raw, axis=0)) / 2
    # cen2, cenz = raw_to_2dz(np.expand_dims(cen, axis=0), caminfo)
    # cen2 = (cll + ctr) / 2
    resce = np.concatenate((
        np.array([float(caminfo.crop_size) / rect.len]),
        rect.cll,
        aabb.dump()
    ))
    return img_crop_resize, resce


def get_rect3(points3, caminfo, m=0.6):
    """ return a rectangle with margin that 3d points
        NOTE: there is still a perspective problem
    """
    cube = iso_cube()
    cube.build(points3, m)
    cen = raw_to_2d(cube.cen.reshape(1, -1), caminfo).flatten()
    rect = iso_rect(
        cen - cube.len,
        cube.get_sidelen()
    )
    rect = clip_image_border(rect, caminfo)
    return rect


def get_rect(pose2d, caminfo, bm=0.6):
    """ return a rectangle with margin that contains 2d point set
    """
    rect = iso_rect()
    rect.build(pose2d, bm)
    rect = clip_image_border(rect, caminfo)
    return rect
