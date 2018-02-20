import os
import numpy as np
from . import ops as dataops
from . import io as dataio
from utils.iso_boxes import iso_cube
from utils.regu_grid import latice_image


def prow_vxudir(args, thedata, batch_data):
    bi, pcnt3, poses, resce, = \
        args[0], args[1], args[2], args[3]
    cube = iso_cube()
    cube.load(resce)
    vxudir = dataops.raw_to_vxudir(
        pcnt3, poses.reshape(-1, 3), cube, thedata)
    batch_data[bi, ...] = vxudir


def prow_vxoff(args, thedata, batch_data):
    bi, pcnt3, poses, resce, = \
        args[0], args[1], args[2], args[3]
    cube = iso_cube()
    cube.load(resce)
    vxoff = dataops.raw_to_vxoff(
        pcnt3, poses.reshape(-1, 3), cube, thedata)
    batch_data[bi, ...] = vxoff


def prow_pose_lab(args, thedata, batch_data):
    bi, poses, resce, = \
        args[0], args[1], args[2]
    cube = iso_cube()
    cube.load(resce)
    pose_lab = dataops.raw_to_vxlab(
        poses.reshape(-1, 3), cube, thedata)
    batch_data[bi, ...] = pose_lab


def prow_pose_hit(args, thedata, batch_data):
    bi, poses, resce, = \
        args[0], args[1], args[2]
    cube = iso_cube()
    cube.load(resce)
    pose_hit = dataops.raw_to_vxhit(
        poses.reshape(-1, 3), cube, thedata)
    batch_data[bi, ...] = pose_hit


def prow_vxhit(args, thedata, batch_data):
    bi, index, resce = \
        args[0], args[1], args[2]
    img_name = dataio.index2imagename(index)
    img = dataio.read_image(os.path.join(
        thedata.training_images, img_name))
    cube = iso_cube()
    cube.load(resce)
    vxhit = dataops.to_vxhit(img, cube, thedata)
    batch_data[bi, ...] = vxhit


def prow_tsdf3(args, thedata, batch_data):
    bi, pcnt3 = \
        args[0], args[1]
    tsdf3 = dataops.trunc_belief(pcnt3)
    batch_data[bi, ...] = tsdf3


def prow_truncd(args, thedata, batch_data):
    bi, pcnt3 = \
        args[0], args[1]
    truncd = dataops.prop_dist(pcnt3)
    batch_data[bi, ...] = truncd


def prow_pcnt3(args, thedata, batch_data):
    bi, index, resce = \
        args[0], args[1], args[2]
    img_name = dataio.index2imagename(index)
    img = dataio.read_image(os.path.join(
        thedata.training_images, img_name))
    cube = iso_cube()
    cube.load(resce)
    pcnt3 = dataops.to_pcnt3(img, cube, thedata)
    batch_data[bi, ...] = pcnt3


def prow_ortho3(args, thedata, batch_data):
    bi, index, resce = \
        args[0], args[1], args[2]
    img_name = dataio.index2imagename(index)
    img = dataio.read_image(os.path.join(
        thedata.training_images, img_name))
    cube = iso_cube()
    cube.load(resce)
    img_ortho3 = dataops.to_ortho3(img, cube, thedata)
    batch_data[bi, ...] = img_ortho3


def prow_pose_c1(args, thedata, batch_data):
    bi, poses, resce, = \
        args[0], args[1], args[2]
    cube = iso_cube()
    cube.load(resce)
    pose_c1 = cube.transform_center_shrink(
        poses.reshape(-1, 3))
    batch_data[bi, ...] = pose_c1.flatten()


def prow_pose_c(args, thedata, batch_data):
    bi, poses, resce, = \
        args[0], args[1], args[2]
    cube = iso_cube()
    cube.load(resce)
    pose_c = cube.transform_to_center(
        poses.reshape(-1, 3))
    batch_data[bi, ...] = pose_c.flatten()


def prow_crop2(args, thedata, batch_data):
    bi, index, resce = \
        args[0], args[1], args[2]
    img_name = dataio.index2imagename(index)
    img = dataio.read_image(os.path.join(
        thedata.training_images, img_name))
    cube = iso_cube()
    cube.load(resce)
    img_crop2 = dataops.to_crop2(img, cube, thedata)
    batch_data[bi, ...] = img_crop2


def prow_clean(args, thedata, batch_data):
    bi, index, resce = \
        args[0], args[1], args[2]
    img_name = dataio.index2imagename(index)
    img = dataio.read_image(os.path.join(
        thedata.training_images, img_name))
    cube = iso_cube()
    cube.load(resce)
    img_clean = dataops.to_clean(img, cube, thedata)
    batch_data[bi, ...] = img_clean


def prow_index(args, thedata, batch_data):
    bi, line = \
        args[0], args[1]
    img_name, pose_raw = dataio.parse_line_annot(line)
    pose2d = dataops.raw_to_2d(pose_raw, thedata)
    if (0 > np.min(pose2d)) or (0 > np.min(thedata.image_size - pose2d)):
        return
    index = dataio.imagename2index(img_name)
    cube = iso_cube(
        (np.max(pose_raw, axis=0) + np.min(pose_raw, axis=0)) / 2,
        thedata.region_size
    )
    batch_data['valid'][bi] = True
    batch_data['index'][bi, ...] = index
    batch_data['poses'][bi, ...] = pose_raw
    batch_data['resce'][bi, ...] = cube.dump()


def test_puttensor(
        args, put_worker, thedata, batch_data):
    from itertools import izip
    import copy
    test_copy = copy.deepcopy(batch_data)
    for args in izip(*args):
        put_worker(
            args, thedata, batch_data)
    print('this is TEST only!!! DO NOT forget to write using mp version')
    return test_copy


def puttensor_mt(args, put_worker, thedata, batch_data):
    # from timeit import default_timer as timer
    # from datetime import timedelta
    # time_s = timer()
    # test_copy = test_puttensor(
    #     args, put_worker, thedata, batch_data)
    # time_e = str(timedelta(seconds=timer() - time_s))
    # print('single tread time: {}'.format(time_e))
    # return

    from functools import partial
    from multiprocessing.dummy import Pool as ThreadPool
    # time_s = timer()
    thread_pool = ThreadPool()
    thread_pool.map(
        partial(put_worker, thedata=thedata, batch_data=batch_data),
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


# def write_region2(fanno, yanker, caminfo, batch_index, batch_resce, batch_poses):
#     for ii in range(batch_index.shape[0]):
#         img_name = dataio.index2imagename(batch_index[ii, 0])
#         pose_local = batch_poses[ii, :]
#         # resce = batch_resce[ii, :]
#         # centre, index, confidence = yanker(pose_local, resce, caminfo)
#         anchor_num = caminfo.anchor_num ** 2
#         pcnt = pose_local[:anchor_num]
#         label = np.argmax(pcnt)
#         index = np.array(np.unravel_index(
#             label,
#             (caminfo.anchor_num, caminfo.anchor_num)))
#         confidence = 1 / (1 + np.exp(-pcnt))
#         anchors = pose_local[anchor_num:]
#         _, _, centre = yank_localizer2_rect(
#             index, anchors, caminfo)
#         fanno.write(
#             img_name +
#             '\t' + '\t'.join("%.4f" % x for x in centre.flatten()) +
#             '\t' + '\t'.join("%.4f" % x for x in confidence.flatten()) +
#             '\n')
#
#
# def write_region(fanno, yanker, caminfo, batch_index, batch_resce, batch_poses):
#     for ii in range(batch_index.shape[0]):
#         img_name = dataio.index2imagename(batch_index[ii, 0])
#         pose_local = batch_poses[ii, :]
#         resce = batch_resce[ii, :]
#         centre = yanker(pose_local, resce, caminfo)
#         fanno.write(
#             img_name +
#             '\t' + '\t'.join("%.4f" % x for x in centre.flatten()) +
#             '\n')
#
#
# def write2d(fanno, yanker, caminfo, batch_index, batch_resce, batch_poses):
#     for ii in range(batch_index.shape[0]):
#         img_name = dataio.index2imagename(batch_index[ii, 0])
#         pose_local = batch_poses[ii, :].reshape(-1, 3)
#         resce = batch_resce[ii, :]
#         pose_raw = yanker(pose_local, resce, caminfo)
#         fanno.write(
#             img_name +
#             '\t' + '\t'.join("%.4f" % x for x in pose_raw.flatten()) +
#             '\n')
