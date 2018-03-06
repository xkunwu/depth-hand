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
