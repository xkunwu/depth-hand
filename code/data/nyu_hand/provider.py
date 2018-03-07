import os
import numpy as np
from . import ops as dataops
from . import io as dataio
from utils.iso_boxes import iso_cube
from utils.regu_grid import latice_image


def prow_ov3edt2m(args, thedata, mode, batch_data):
    bi, ov3edt2, ov3dist2 = \
        args[0], args[1], args[2]
    ov3edt2m = np.multiply(ov3edt2, ov3dist2)
    batch_data[bi, ...] = ov3edt2m


def prow_ov3dist2(args, thedata, mode, batch_data):
    bi, vxudir = \
        args[0], args[1]
    olmap = vxudir[..., :thedata.join_num]
    hmap_size = thedata.hmap_size
    num_j = thedata.join_num
    ov3dist2 = np.empty((hmap_size, hmap_size, num_j * 3))
    ov3dist2[..., :num_j] = np.swapaxes(np.max(olmap, axis=2), 0, 1)
    ov3dist2[..., num_j:(2 * num_j)] = np.max(olmap, axis=1)
    ov3dist2[..., (2 * num_j):] = np.swapaxes(np.max(olmap, axis=0), 0, 1)
    batch_data[bi, ...] = ov3dist2


def prow_ov3edt2(args, thedata, mode, batch_data):
    bi, ortho3, poses, resce = \
        args[0], args[1], args[2], args[3]
    cube = iso_cube()
    cube.load(resce)
    ov3edt2 = dataops.prop_ov3edt2(
        ortho3, poses.reshape(-1, 3), cube, thedata)
    batch_data[bi, ...] = ov3edt2


def prow_edt2m(args, thedata, mode, batch_data):
    bi, edt2, udir2 = \
        args[0], args[1], args[2]
    edt2m = np.multiply(edt2, udir2[..., :thedata.join_num])
    batch_data[bi, ...] = edt2m


def prow_edt2(args, thedata, mode, batch_data):
    bi, clean, poses, resce = \
        args[0], args[1], args[2], args[3]
    cube = iso_cube()
    cube.load(resce)
    edt2 = dataops.prop_edt2(
        clean, poses.reshape(-1, 3), cube, thedata)
    batch_data[bi, ...] = edt2


def prow_udir2(args, thedata, mode, batch_data):
    bi, clean, poses, resce, = \
        args[0], args[1], args[2], args[3]
    cube = iso_cube()
    cube.load(resce)
    udir2 = dataops.raw_to_udir2(
        clean, poses.reshape(-1, 3), cube, thedata)
    batch_data[bi, ...] = udir2


def prow_hmap2(args, thedata, mode, batch_data):
    bi, poses, resce, = \
        args[0], args[1], args[2]
    cube = iso_cube()
    cube.load(resce)
    hmap2 = dataops.raw_to_heatmap2(
        poses.reshape(-1, 3), cube, thedata)
    batch_data[bi, ...] = hmap2


def prow_vxudir(args, thedata, mode, batch_data):
    bi, pcnt3, poses, resce, = \
        args[0], args[1], args[2], args[3]
    cube = iso_cube()
    cube.load(resce)
    vxudir = dataops.raw_to_vxudir(
        pcnt3, poses.reshape(-1, 3), cube, thedata)
    batch_data[bi, ...] = vxudir


def prow_pose_lab(args, thedata, mode, batch_data):
    bi, poses, resce, = \
        args[0], args[1], args[2]
    cube = iso_cube()
    cube.load(resce)
    pose_lab = dataops.raw_to_vxlab(
        poses.reshape(-1, 3), cube, thedata)
    batch_data[bi, ...] = pose_lab


def prow_pose_hit(args, thedata, mode, batch_data):
    bi, poses, resce, = \
        args[0], args[1], args[2]
    cube = iso_cube()
    cube.load(resce)
    pose_hit = dataops.raw_to_vxhit(
        poses.reshape(-1, 3), cube, thedata)
    batch_data[bi, ...] = pose_hit


def prow_vxhit(args, thedata, mode, batch_data):
    bi, index, resce = \
        args[0], args[1], args[2]
    img_name = dataio.index2imagename(index)
    img = dataio.read_image(thedata.images_join(img_name, mode))
    cube = iso_cube()
    cube.load(resce)
    vxhit = dataops.to_vxhit(img, cube, thedata)
    batch_data[bi, ...] = vxhit


def prow_pcnt3(args, thedata, mode, batch_data):
    bi, index, resce = \
        args[0], args[1], args[2]
    img_name = dataio.index2imagename(index)
    img = dataio.read_image(thedata.images_join(img_name, mode))
    cube = iso_cube()
    cube.load(resce)
    pcnt3 = dataops.to_pcnt3(img, cube, thedata)
    batch_data[bi, ...] = pcnt3


def prow_ortho3(args, thedata, mode, batch_data):
    bi, index, resce = \
        args[0], args[1], args[2]
    img_name = dataio.index2imagename(index)
    img = dataio.read_image(thedata.images_join(img_name, mode))
    cube = iso_cube()
    cube.load(resce)
    img_ortho3 = dataops.to_ortho3(img, cube, thedata)
    batch_data[bi, ...] = img_ortho3


def prow_pose_c(args, thedata, mode, batch_data):
    bi, poses, resce, = \
        args[0], args[1], args[2]
    cube = iso_cube()
    cube.load(resce)
    pose_c = cube.transform_to_center(
        poses.reshape(-1, 3))
    batch_data[bi, ...] = pose_c.flatten()


def prow_crop2(args, thedata, mode, batch_data):
    bi, index, resce = \
        args[0], args[1], args[2]
    img_name = dataio.index2imagename(index)
    img = dataio.read_image(thedata.images_join(img_name, mode))
    cube = iso_cube()
    cube.load(resce)
    img_crop2 = dataops.to_crop2(img, cube, thedata)
    batch_data[bi, ...] = img_crop2


def prow_clean(args, thedata, mode, batch_data):
    bi, index, resce = \
        args[0], args[1], args[2]
    img_name = dataio.index2imagename(index)
    img = dataio.read_image(thedata.images_join(img_name, mode))
    cube = iso_cube()
    cube.load(resce)
    img_clean = dataops.to_clean(img, cube, thedata)
    batch_data[bi, ...] = img_clean


def prow_index(args, thedata, mode, batch_data):
    bi, index, poses = \
        args[0], args[1], args[2]
    pose_raw = poses.reshape(-1, 3)
    # pose_raw[:, [0, 1]] = pose_raw[:, [1, 0]]
    pose2d = dataops.raw_to_2d(pose_raw, thedata)
    if (0 > np.min(pose2d)) or (0 > np.min(thedata.image_size - pose2d)):
        return
    cube = iso_cube(
        (np.max(pose_raw, axis=0) + np.min(pose_raw, axis=0)) / 2,
        thedata.region_size
    )
    batch_data['valid'][bi] = True
    batch_data['index'][bi, ...] = index
    batch_data['poses'][bi, ...] = pose_raw
    batch_data['resce'][bi, ...] = cube.dump()


def test_puttensor(
        args, put_worker, thedata, mode, batch_data):
    from itertools import izip
    import copy
    test_copy = copy.deepcopy(batch_data)
    for args in izip(*args):
        put_worker(
            args, thedata, mode, batch_data)
    print('this is TEST only!!! DO NOT forget to write using mp version')
    return test_copy


def puttensor_mt(args, put_worker, thedata, mode, batch_data):
    # from timeit import default_timer as timer
    # from datetime import timedelta
    # time_s = timer()
    # test_copy = test_puttensor(
    #     args, put_worker, thedata, mode, batch_data)
    # time_e = str(timedelta(seconds=timer() - time_s))
    # print('single tread time: {}'.format(time_e))
    # return

    from functools import partial
    from multiprocessing.dummy import Pool as ThreadPool
    # time_s = timer()
    thread_pool = ThreadPool()
    thread_pool.map(
        partial(put_worker, thedata=thedata, mode=mode, batch_data=batch_data),
        zip(*args))
    thread_pool.close()  # that's it for this batch
    thread_pool.join()  # serilization point
    # time_e = str(timedelta(seconds=timer() - time_s))
    # print('multiprocessing time: {:.4f}'.format(time_e))

    # import numpy as np
    # print(np.linalg.norm(batch_data.batch_index - test_copy.batch_index))
    # print(np.linalg.norm(batch_data.batch_frame - test_copy.batch_frame))
    # print(np.linalg.norm(batch_data.batch_poses - test_copy.batch_poses))
    # print(np.linalg.norm(batch_data.batch_resce - test_copy.batch_resce))
