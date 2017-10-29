import os
import sys
from importlib import import_module
import numpy as np
from cv2 import resize as cv2resize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
iso_box = getattr(
    import_module('regu_grid'),
    'iso_box'
)
iso_rect = getattr(
    import_module('regu_grid'),
    'iso_rect'
)


def d2z_to_raw(p2z, caminfo, rescen=np.array([1, 0, 0])):
    """ reproject 2d poses to 3d.
        p2z: nx3 array, 2d position and real z
    """
    pose2d = p2z[:, 0:2] / rescen[0] + rescen[1:3]
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
        np.asarray(indx).T,
        zval.reshape(-1, 1))
    )
    points3 = d2z_to_raw(indz, caminfo)
    return points3


def raw_to_2dz(points3, caminfo, rescen=np.array([1, 0, 0])):
    """ project 3D point onto image plane using camera info
        Args:
            points3: nx3 array, raw input in real world coordinates
    """
    pose_z = points3[:, 2]
    pose2d = points3[:, 0:2] / pose_z.reshape(-1, 1) * caminfo.focal + caminfo.centre
    return (pose2d - rescen[1:3]) * rescen[0], pose_z


def raw_to_2d(points3, caminfo, rescen=np.array([1, 0, 0])):
    pose2d, _ = raw_to_2dz(points3, caminfo, rescen)
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


def crop_resize_pca(img, pose_raw, caminfo):
    box = iso_box()
    box.build(pose_raw)
    points3 = box.pick(
        img_to_raw(img, caminfo)
    )
    points2, pose_z = raw_to_2dz(points3, caminfo)
    rect = iso_rect()
    rect.build(points2, -0.3)  # shrink size, as PCA produced much larger box
    img_crop = rect.print_image(points2, pose_z)
    img_crop_resize = cv2resize(img_crop, (caminfo.crop_size, caminfo.crop_size))
    rescen = np.append(float(caminfo.crop_size) / rect.get_sidelen(), rect.cen - rect.len)
    return img_crop_resize, rescen


def clip_image_border(rect, caminfo):
    # clip to image border
    ctl = rect.cen - rect.len
    cbr = rect.cen + rect.len
    obm = np.min([ctl, caminfo.image_size - cbr])
    if 0 > obm:
        # print(ctl, caminfo.image_size - cbr, obm, rect.len)
        rect.len += obm
    return rect


def crop_resize(img, pose_raw, caminfo):
    rect = get_rect3(
        pose_raw, caminfo
    )
    rect = clip_image_border(rect, caminfo)
    points3 = img_to_raw(img, caminfo)
    points2, pose_z = raw_to_2dz(points3, caminfo)
    conds = rect.pick(points2)
    img_crop = rect.print_image(points2[conds, :], pose_z[conds])
    img_crop_resize = cv2resize(img_crop, (caminfo.crop_size, caminfo.crop_size))
    rescen = np.append(float(caminfo.crop_size) / rect.get_sidelen(), rect.cen - rect.len)
    return img_crop_resize, rescen


def get_rect3(points3, caminfo):
    """ return a rectangle with margin that 3d points
        NOTE: there is still a perspective problem
    """
    box = iso_box()
    box.build(points3)
    rect = iso_rect(
        raw_to_2d(box.cen.reshape(1, -1), caminfo),
        box.len
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
