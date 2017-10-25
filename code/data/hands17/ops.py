import os
import sys
from importlib import import_module
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
iso_box = getattr(
    import_module('regu_grid'),
    'iso_box'
)


def d2z_to_raw(p2z, centre, focal, rescen=None):
    """ reproject 2d poses to 3d.
        p2z: nx3 array, 2d position and real z
    """
    if rescen is None:
        rescen = np.array([1, 0, 0])
    pose2d = p2z[:, 0:2] / rescen[0] + rescen[1:3]
    pose_z = np.array(p2z[:, 2]).reshape(-1, 1)
    pose3d = (pose2d - centre) / focal * pose_z
    # pose3d = (pose2d - centre) / (-focal[0], focal[1]) * pose_z
    return np.hstack((pose3d, pose_z))


def img_to_raw(img, centre, focal, z_near=0, z_far=np.inf):
    conds = np.logical_and(
        z_far > img,
        z_near < img
    )
    indx = np.where(conds)[::-1]  # need to reverse image plane coordinates
    zval = np.array(img[conds])
    indz = np.hstack((
        np.asarray(indx).T,
        zval.reshape(-1, 1))
    )
    points3 = d2z_to_raw(indz, centre, focal)
    return points3


def raw_to_2d(points3, centre, focal):
    """ project 3D point onto image plane using camera info
        Args:
            points3: nx3 array, raw input in real world coordinates
    """
    pose_z = np.array(points3[:, 2]).reshape(-1, 1)
    return points3[:, 0:2] / pose_z * focal + centre


def getbm(base_z, focal, base_margin=20):
    """ return margin (x, y) accroding to projective-z of MMCP.
        Args:
            base_z: base z-value in mm
            base_margin: base margin in mm
    """
    marg = np.tile(base_margin, (2, 1)) * focal / base_z
    m = max(marg)
    return m


def get_cube_pca(pose3d, image_size, bm):
    """ this cube contains all 3d points, and align with their PCA axes
    """


def get_rect3(points3, centre, focal, image_size, m):
    """ return a rectangle with margin that 3d points
        NOTE: there is still a perspective problem
    """
    box = iso_box()
    box.build(points3, m)
    # clip to image border
    ctl = raw_to_2d((box.cen - box.len).reshape(1, -1), centre, focal)
    cbr = raw_to_2d((box.cen + box.len).reshape(1, -1), centre, focal)
    obm = np.min([ctl, image_size - cbr])
    if 0 > obm:
        # print(ctl, image_size - cbr, obm, box.len)
        box.len = box.len + obm
    ctl = raw_to_2d((box.cen - box.len).reshape(1, -1), centre, focal)
    cbl = box.cen - box.len
    cbl[0] += 2 * box.len
    print(ctl)
    cbl = raw_to_2d(cbl.reshape(1, -1), centre, focal)
    print(cbl)
    sidelen = cbl[0, 0] - ctl[0, 0]
    print(sidelen)
    return np.vstack((ctl, np.array([sidelen, sidelen])))


def get_rect(pose2d, image_size, bm):
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
    obm = np.min([ctl, image_size - cbr])
    if 0 > obm:
        # print(ctl, image_size - cbr, obm, ptp_m)
        ptp_m = ptp_m + 2 * obm
    ctl = cen - ptp_m / 2
    return np.vstack((ctl, np.array([ptp_m, ptp_m])))
