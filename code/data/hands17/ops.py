import os
import sys
from importlib import import_module
import numpy as np
import skfmm
from cv2 import resize as cv2resize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir, os.pardir))
sys.path.append(BASE_DIR)
iso_rect = getattr(
    import_module('utils.iso_boxes'),
    'iso_rect'
)
iso_aabb = getattr(
    import_module('utils.iso_boxes'),
    'iso_aabb'
)
iso_cube = getattr(
    import_module('utils.iso_boxes'),
    'iso_cube'
)
regu_grid = getattr(
    import_module('utils.regu_grid'),
    'regu_grid'
)
grid_cell = getattr(
    import_module('utils.regu_grid'),
    'grid_cell'
)


def raw_to_pca(points3, resce=np.array([1, 0, 0, 0])):
    cube = iso_cube()
    cube.load(resce)
    return cube.transform(points3)


def pca_to_raw(points3, resce=np.array([1, 0, 0, 0])):
    cube = iso_cube()
    cube.load(resce)
    return cube.transform_inv(points3)


def raw_to_local(points3, resce=np.array([1, 0, 0, 0])):
    aabb = iso_aabb()
    aabb.load(resce)
    return aabb.transform(points3)


def local_to_raw(points3, resce=np.array([1, 0, 0, 0])):
    aabb = iso_aabb()
    aabb.load(resce)
    return aabb.transform_inv(points3)


def d2z_to_raw(p2z, caminfo, resce=np.array([1, 0, 0])):
    """ reproject 2d poses to 3d.
        p2z: nx3 array, 2d position and real z
    """
    pose2d = p2z[:, 0:2] / resce[0] + resce[1:3]
    pose_z = np.array(p2z[:, 2]).reshape(-1, 1)
    pose3d = (pose2d - caminfo.centre) / caminfo.focal * pose_z
    # pose3d = (pose2d - caminfo.centre) / (-caminfo.focal[0], caminfo.focal[1]) * pose_z
    return np.hstack((pose3d, pose_z))


def img_to_raw(img, caminfo, crop_lim=None):
    # crop_lim = np.array([[-0.8, -0.8, 0], [0.8, 0.8, 1.6]])
    if crop_lim is None:
        conds = np.logical_and(
            caminfo.z_far > img,
            caminfo.z_near < img
        )
    else:
        conds = np.logical_and(
            crop_lim[0, 2] < img,
            crop_lim[1, 2] > img
        )
    indx = np.where(conds)[::-1]  # need to reverse image plane coordinates
    zval = np.array(img[conds])
    indz = np.hstack((
        np.asarray(indx).astype(float).T,
        zval.reshape(-1, 1))
    )
    points3 = d2z_to_raw(indz, caminfo)
    if crop_lim is not None:
        conds = np.logical_and.reduce([
            crop_lim[0, 0] < points3[:, 0],
            crop_lim[1, 0] > points3[:, 0],
            crop_lim[0, 1] < points3[:, 1],
            crop_lim[1, 1] > points3[:, 1],
        ])
        return points3[conds, :]
    else:
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
    cbr = rect.cll + rect.sidelen
    cen = rect.cll + rect.sidelen / 2
    obm = np.min([ctl, caminfo.image_size - cbr])
    if 0 > obm:
        # print(ctl, caminfo.image_size - cbr, obm, rect.sidelen)
        rect.sidelen += obm * 2
        rect.cll = cen - rect.sidelen / 2
    return rect


def fill_grid(img, pose_raw, step, caminfo):
    cube = iso_cube()
    cube.build(pose_raw)
    _, points3_trans = cube.pick(img_to_raw(img, caminfo))
    grid = regu_grid()
    grid.from_cube(cube, step)
    grid.fill(points3_trans)
    # resce = np.append(
    #     cube.dump(),
    #     step
    # )
    resce = cube.dump()
    return grid.pcnt, resce


def voxelize_depth(img, step, caminfo):
    sidelen = caminfo.crop_range
    halflen = sidelen / 2
    crop_lim = np.array(
        [[-halflen, -halflen, 0], [halflen, halflen, sidelen]])
    points3 = img_to_raw(img, caminfo, crop_lim)
    grid = regu_grid(crop_lim[0], step, sidelen / step)
    grid.fill(points3)
    # grid.show_dims()
    # print(np.histogram(grid.pcnt))
    # cube = iso_cube()
    # cube.build(pose_raw)
    # resce = np.concatenate((
    #     cube.dump(),
    #     np.array([sidelen, step])
    # ))
    # resce = cube.dump()
    # mpplot = import_module('matplotlib.pyplot')
    # ax = mpplot.subplot(projection='3d')
    # numpts = points3.shape[0]
    # if 1000 < numpts:
    #     samid = np.random.choice(numpts, 1000, replace=False)
    #     points3_sam = points3[samid, :]
    # else:
    #     points3_sam = points3
    # ax.scatter(
    #     points3_sam[:, 0], points3_sam[:, 1], points3_sam[:, 2])
    # corners = cube.get_corners()
    # iso_cube.draw_cube_wire(corners)
    # print(corners)
    # cube.show_dims()
    # mpplot.show()
    return grid.pcnt


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


def crop_resize_pca(img, pose_raw, caminfo):
    cube = iso_cube()
    cube.build(pose_raw)
    _, points3_trans = cube.pick(img_to_raw(img, caminfo))
    coord, depth = cube.project_pca(points3_trans)
    # x = np.arange(-1, 1, 0.5)
    # y = np.stack([x, x, -x]).T
    # print(y)
    # coord, depth = cube.project_pca(y)
    # print(coord)
    # print(depth)
    img_crop_resize = cube.print_image(coord, depth, caminfo.crop_size)
    # mpplot = import_module('matplotlib.pyplot')
    # mpplot.imshow(img_crop_resize, cmap='bone')
    # mpplot.show()
    # rect = get_rect3(cube, caminfo)
    # points2, pose_z = raw_to_2dz(points3, caminfo)
    # rect = iso_rect()
    # rect.build(points2, 0.5)
    # rect = clip_image_border(rect, caminfo)
    # conds = rect.pick(points2)
    # img_crop = rect.print_image(points2[conds, :], pose_z[conds])
    # img_crop_resize = cv2resize(img_crop, (caminfo.crop_size, caminfo.crop_size))
    # img_crop_resize = (img_crop_resize.astype(float) - np.min(img_crop_resize)) /\
    #     (np.max(img_crop_resize) - np.min(img_crop_resize))
    # img_crop_resize = img_crop
    # resce = np.concatenate((
    #     np.array([float(caminfo.crop_size) / rect.sidelen]),
    #     rect.cll,
    #     cube.dump()
    # ))
    resce = cube.dump()
    return img_crop_resize, resce


def get_rect2(aabb, caminfo, m=0.2):
    from colour import Color
    # mpplot = import_module('matplotlib.pyplot')
    c3a = np.array([
        aabb.cll,
        np.append(aabb.cll[:2] + aabb.sidelen, aabb.cll[2])
    ])  # central z-plane
    c2a = raw_to_2d(c3a, caminfo)
    cll = c2a[0, :]
    ctr = c2a[1, :]
    rect = iso_rect(cll, np.max(ctr - cll), m)
    rect.draw(Color('red').rgb)
    c3a = np.array([
        np.append(aabb.cll[:2], aabb.cll[2] + aabb.sidelen),
        np.append(aabb.cll[:2] + aabb.sidelen, aabb.cll[2] + aabb.sidelen)
    ])  # central z-plane
    c2a = raw_to_2d(c3a, caminfo)
    cll = c2a[0, :]
    ctr = c2a[1, :]
    rect = iso_rect(cll, np.max(ctr - cll), m)
    rect.draw(Color('blue').rgb)

    cen = aabb.cll + aabb.sidelen / 2
    c3a = np.array([
        np.append(cen[:2] - aabb.sidelen / 2, cen[2]),
        np.append(cen[:2] + aabb.sidelen / 2, cen[2]),
        # aabb.cll,
        # np.append(aabb.cll[:2] + aabb.sidelen, aabb.cll[2]),
        # aabb.cll + aabb.sidelen / 2
    ])  # central z-plane
    c2a = raw_to_2d(c3a, caminfo)
    cll = c2a[0, :]
    ctr = c2a[1, :]
    rect = iso_rect(cll, np.max(ctr - cll), m)
    rect = clip_image_border(rect, caminfo)
    return rect


def get_rect(pose2d, caminfo, bm=0.6):
    """ return a rectangle with margin that contains 2d point set
    """
    rect = iso_rect()
    rect.build(pose2d, bm)
    rect = clip_image_border(rect, caminfo)
    return rect


def crop_resize(img, pose_raw, caminfo):
    aabb = iso_aabb()
    aabb.build(pose_raw)
    # rect = get_rect2(aabb, caminfo)
    # cube = iso_cube()
    # cube.build(pose_raw)
    # rect = get_rect3(cube, caminfo)
    points2 = raw_to_2d(pose_raw, caminfo)
    rect = get_rect(points2, caminfo)
    # mpplot.imshow(img, cmap='bone')
    # rect.draw()
    # mpplot.show()
    cll_i = np.floor(rect.cll).astype(int)
    sizel = np.floor(rect.sidelen).astype(int)
    img_crop = img[
        cll_i[1]:cll_i[1] + sizel,
        cll_i[0]:cll_i[0] + sizel,
    ]

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
        # np.array([float(caminfo.crop_size) / sizel]),
        # rect.cll,
        rect.dump(),
        aabb.dump()
    ))
    return img_crop_resize, resce
