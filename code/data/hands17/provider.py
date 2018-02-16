import os
import numpy as np
from itertools import islice
from . import ops as dataops
from . import io as dataio
from utils.iso_boxes import iso_cube
from utils.regu_grid import latice_image


def prow_pose_c(args, thedata, batchallot):
    bi, poses, resce, = \
        args[0], args[2], args[3]
    cube = iso_cube()
    cube.load(resce)
    pose_c = cube.transform_to_center(
        poses.reshape(-1, 3))
    batchallot.entry['pose_c'][bi, ...] = pose_c.flatten()


def prow_crop2(args, thedata, batchallot):
    bi, index, resce = \
        args[0], args[1], args[3]
    img_name = dataio.index2imagename(index)
    img = dataio.read_image(os.path.join(
        thedata.training_images, img_name))
    cube = iso_cube()
    cube.load(resce)
    img_crop2 = dataops.to_crop2(img, cube, thedata)
    batchallot.entry['crop2'][bi, ...] = img_crop2


def prow_clean(args, thedata, batchallot):
    bi, index, resce = \
        args[0], args[1], args[3]
    img_name = dataio.index2imagename(index)
    img = dataio.read_image(os.path.join(
        thedata.training_images, img_name))
    cube = iso_cube()
    cube.load(resce)
    img_clean = dataops.to_clean(img, cube, thedata)
    batchallot.entry['clean'][bi, ...] = img_clean


def test_puttensor(
        num_line, index, poses, resce, put_worker, thedata, batchallot):
    import copy
    test_copy = copy.deepcopy(batchallot)
    for bi, ii, pp, rr in zip(range(num_line), index, poses, resce):
        put_worker(
            (bi, ii, pp, rr), thedata, batchallot)
    print('this is TEST only!!! DO NOT forget to write using mp version')
    return test_copy


def puttensor_mt(
    fanno, store_beg, store_end,
        put_worker, thedata, batchallot):
    num_line = store_end - store_beg
    if 0 >= num_line:
        return 0
    index = fanno['index'][store_beg:store_end, 0]
    poses = fanno['poses'][store_beg:store_end, :]
    resce = fanno['resce'][store_beg:store_end, :]

    # from timeit import default_timer as timer
    # from datetime import timedelta
    # time_s = timer()
    # test_copy = test_puttensor(
    #     num_line, index, poses, resce, put_worker, thedata, batchallot)
    # time_e = str(timedelta(seconds=timer() - time_s))
    # print('single tread time: {}'.format(time_e))
    # return num_line

    from functools import partial
    from multiprocessing.dummy import Pool as ThreadPool
    # time_s = timer()
    thread_pool = ThreadPool()
    thread_pool.map(
        partial(put_worker, thedata=thedata, batchallot=batchallot),
        zip(range(num_line), index, poses, resce))
    thread_pool.close()
    thread_pool.join()
    # time_e = str(timedelta(seconds=timer() - time_s))
    # print('multiprocessing time: {:.4f}'.format(time_e))

    # import numpy as np
    # print(np.linalg.norm(batchallot.batch_index - test_copy.batch_index))
    # print(np.linalg.norm(batchallot.batch_frame - test_copy.batch_frame))
    # print(np.linalg.norm(batchallot.batch_poses - test_copy.batch_poses))
    # print(np.linalg.norm(batchallot.batch_resce - test_copy.batch_resce))

    return num_line


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
