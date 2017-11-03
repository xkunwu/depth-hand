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
iso_cube = getattr(
    import_module('iso_boxes'),
    'iso_cube'
)
iso_rect = getattr(
    import_module('iso_boxes'),
    'iso_rect'
)
regu_grid = getattr(
    import_module('regu_grid'),
    'regu_grid'
)
grid_cell = getattr(
    import_module('regu_grid'),
    'grid_cell'
)


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
        rect.len += obm
        rect.cll = cen - rect.len / 2
    return rect


def fill_grid(img, pose_raw, step, caminfo):
    box = iso_cube()
    box.build(pose_raw)
    points3 = box.pick(img_to_raw(img, caminfo))
    points3_trans = box.transform(points3) + box.len  # 0-based
    cell = grid_cell()
    cell.build(points3_trans)
    grid = regu_grid()
    grid.build(cell, step)
    grid.fill(points3_trans)
    resce = np.append(
        grid.cll.flatten(),
        grid.len
    )
    return grid.pcnt, resce


def trunc_belief(pcnt):
    size = pcnt.shape[0]
    # print(pcnt.shape, np.max(pcnt))
    phi = np.ones_like(pcnt)
    z0front = np.ones((size, size)) * (size - 1)
    # print(z0front)
    # print(pcnt[..., 15])
    # print(np.where(0 < pcnt))
    for index in zip(*np.where(0 < pcnt)):
        if z0front[index[0:2]] > index[2]:
            z0front[index[0:2]] = index[2]
    zrange = np.repeat(np.arange(size), size * size).reshape(size, size, size)
    phi[(z0front == zrange)] = 0
    phi[(z0front < zrange)] = -1
    bef = skfmm.distance(phi, dx=1e-2, narrow=0.3)
    # plt.title('Distance calculation limited to narrow band')
    # plt.contour(X, Y, phi, [0], linewidths=(3), colors='black')
    # plt.contour(X, Y, d, 15)
    # plt.colorbar()
    # plt.show()
    return bef


def proj_ortho3(img, pose_raw, caminfo):
    box = iso_cube()
    box.build(pose_raw)
    points3 = box.pick(img_to_raw(img, caminfo))
    points3_trans = box.transform(points3)
    img_crop_resize = []
    for spi in range(3):
        coord, depth = box.project_pca(points3_trans, roll=spi)
        img_crop = box.print_image(coord, depth)
        img_crop_resize.append(
            cv2resize(img_crop, (caminfo.crop_size, caminfo.crop_size))
        )
        # pose2d, _ = box.project_pca(pose_trans, roll=spi, sort=False)
    resce = np.append(
        float(caminfo.crop_size) / box.get_sidelen(),
        box.cen
    )
    resce = np.concatenate((
        resce,
        box.evecs.flatten()))
    return np.stack(img_crop_resize, axis=2), resce


def crop_resize_pca(img, pose_raw, caminfo):
    box = iso_cube()
    box.build(pose_raw)
    points3 = box.pick(img_to_raw(img, caminfo))
    points2, pose_z = raw_to_2dz(points3, caminfo)
    rect = iso_rect()
    rect.build(points2, 0.5)
    rect = clip_image_border(rect, caminfo)
    conds = rect.pick(points2)
    img_crop = rect.print_image(points2[conds, :], pose_z[conds])
    img_crop_resize = cv2resize(img_crop, (caminfo.crop_size, caminfo.crop_size))
    resce = np.append(
        float(caminfo.crop_size) / rect.len,
        rect.cll)
    return img_crop_resize, resce


def crop_resize(img, pose_raw, caminfo):
    rect = get_rect3(
        pose_raw, caminfo
    )
    points3 = img_to_raw(img, caminfo)
    points2, pose_z = raw_to_2dz(points3, caminfo)
    conds = rect.pick(points2)
    img_crop = rect.print_image(points2[conds, :], pose_z[conds])
    img_crop_resize = cv2resize(img_crop, (caminfo.crop_size, caminfo.crop_size))
    resce = np.append(float(caminfo.crop_size) / rect.len, rect.cll)
    return img_crop_resize, resce


def get_rect3(points3, caminfo):
    """ return a rectangle with margin that 3d points
        NOTE: there is still a perspective problem
    """
    box = iso_cube()
    box.build(points3)
    cen = raw_to_2d(box.cen.reshape(1, -1), caminfo).flatten()
    rect = iso_rect(
        cen - box.len,
        box.get_sidelen()
    )
    rect = clip_image_border(rect, caminfo)
    return rect


def get_rect(pose2d, caminfo, bm):
    """ return a rectangle with margin that contains 2d point set
    """
    ptp = np.ptp(pose2d, axis=0)
    ctl = np.min(pose2d, axis=0)
    cen = ctl + ptp / 2
    ptp_m = max(ptp)
    if 1 > bm:
        bm = ptp_m * bm
    ptp_m = ptp_m + 2 * bm
    # clip to image border
    ctl = cen - ptp_m / 2
    cbr = ctl + ptp_m
    obm = np.min([ctl, caminfo.image_size - cbr])
    if 0 > obm:
        # print(ctl, caminfo.image_size - cbr, obm, ptp_m)
        ptp_m = ptp_m + 2 * obm
    ctl = cen - ptp_m / 2
    return np.vstack((ctl, np.array([ptp_m, ptp_m])))
