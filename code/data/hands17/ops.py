import numpy as np


def get3d(pose_mat, centre, focal, rescen=None):
    """ reproject 2d poses to 3d.
        pose_mat: nx3 array
    """
    if rescen is None:
        rescen = np.array([1, 0, 0])
    pose2d = pose_mat[:, 0:2] / rescen[0] + rescen[1:3]
    pose_z = np.array(pose_mat[:, 2]).reshape(-1, 1)
    pose3d = (pose2d - centre) / focal * pose_z
    return np.hstack((pose3d, pose_z))


def get2d(points3, centre, focal):
    """ project 3D point onto image plane using camera info
        Args:
            points3: nx3 array
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


def get_rect(pose2d, image_size, bm):
    """ return a rectangle with margin that contains 2d point set
    """
    # bm = 300
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
