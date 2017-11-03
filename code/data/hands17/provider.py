import os
import numpy as np
from itertools import islice
import ops as dataops
import io as dataio


def put3df_worker(args, thedata, image_dir, batchallot):
    bi = args[0]
    line = args[1]
    img_name, pose_raw = dataio.parse_line_annot(line)
    img = dataio.read_image(os.path.join(image_dir, img_name))
    pcnt, resce = dataops.fill_grid(img, pose_raw, batchallot.image_size, thedata)
    bef = dataops.trunc_belief(pcnt)
    batchallot.batch_index[bi, :] = dataio.imagename2index(img_name)
    batchallot.batch_frame[bi, ...] = np.expand_dims(bef, axis=3)
    batchallot.batch_poses[bi, :] = pose_raw.flatten().T
    batchallot.batch_resce[bi, :] = resce


def put3v_worker(args, thedata, image_dir, batchallot):
    bi = args[0]
    line = args[1]
    img_name, pose_raw = dataio.parse_line_annot(line)
    img = dataio.read_image(os.path.join(image_dir, img_name))
    pcnt, resce = dataops.fill_grid(img, pose_raw, batchallot.image_size, thedata)
    batchallot.batch_index[bi, :] = dataio.imagename2index(img_name)
    batchallot.batch_frame[bi, ...] = np.expand_dims(pcnt, axis=3)
    batchallot.batch_poses[bi, :] = pose_raw.flatten().T
    batchallot.batch_resce[bi, :] = resce


def put3o_worker(args, thedata, image_dir, batchallot):
    bi = args[0]
    line = args[1]
    img_name, pose_raw = dataio.parse_line_annot(line)
    img = dataio.read_image(os.path.join(image_dir, img_name))
    img_crop_resize, resce = dataops.proj_ortho3(
        img, pose_raw, thedata)
    # pose2d = dataops.raw_to_2d(pose_raw, thedata, resce)
    batchallot.batch_index[bi, :] = dataio.imagename2index(img_name)
    batchallot.batch_frame[bi, ...] = img_crop_resize
    batchallot.batch_poses[bi, :] = pose_raw.flatten().T
    batchallot.batch_resce[bi, :] = resce


def put2dc_worker(args, thedata, image_dir, batchallot):
    bi = args[0]
    line = args[1]
    img_name, pose_raw = dataio.parse_line_annot(line)
    img = dataio.read_image(os.path.join(image_dir, img_name))
    img_crop_resize, resce = dataops.crop_resize_pca(
        img, pose_raw, thedata)
    # pose2d = dataops.raw_to_2d(pose_raw, thedata, resce)
    batchallot.batch_index[bi, :] = dataio.imagename2index(img_name)
    batchallot.batch_frame[bi, ...] = np.expand_dims(img_crop_resize, axis=2)
    batchallot.batch_poses[bi, :] = pose_raw.flatten().T
    batchallot.batch_resce[bi, :] = resce


def put2d_worker(args, thedata, image_dir, batchallot):
    bi = args[0]
    line = args[1]
    img_name, pose_raw = dataio.parse_line_annot(line)
    img = dataio.read_image(os.path.join(image_dir, img_name))
    img_crop_resize, resce = dataops.crop_resize(
        img, pose_raw, thedata)
    # pose2d = dataops.raw_to_2d(pose_raw, thedata, resce)
    batchallot.batch_index[bi, :] = dataio.imagename2index(img_name)
    batchallot.batch_frame[bi, ...] = np.expand_dims(img_crop_resize, axis=2)
    batchallot.batch_poses[bi, :] = pose_raw.flatten().T
    batchallot.batch_resce[bi, :] = resce


def put2d_mt(fanno, worker, thedata, image_dir, batchallot):
    next_n_lines = list(islice(fanno, batchallot.store_size))
    if not next_n_lines:
        return -1
    num_line = len(next_n_lines)

    # from timeit import default_timer as timer
    from functools import partial
    from multiprocessing.dummy import Pool as ThreadPool
    # time_s = timer()
    thread_pool = ThreadPool()
    thread_pool.map(
        partial(worker, thedata=thedata, image_dir=image_dir, batchallot=batchallot),
        zip(range(num_line), next_n_lines))
    thread_pool.close()
    thread_pool.join()
    # print('multiprocessing time: {:.4f}'.format(timer() - time_s))

    # import copy
    # # time_s = timer()
    # test_copy = copy.deepcopy(batchallot)
    # for bi, line in enumerate(next_n_lines):
    #     worker((bi, line), thedata, image_dir, test_copy)
    # # print('single tread time: {:.4f}'.format(timer() - time_s))
    # print('this is TEST only!!! DO NOT forget to write using mp version')

    # import numpy as np
    # print(np.linalg.norm(batchallot.batch_index - test_copy.batch_index))
    # print(np.linalg.norm(batchallot.batch_frame - test_copy.batch_frame))
    # print(np.linalg.norm(batchallot.batch_poses - test_copy.batch_poses))
    # print(np.linalg.norm(batchallot.batch_resce - test_copy.batch_resce))
    return num_line


def write2d(fanno, batch_index, batch_poses):
    for ii in range(batch_index.shape[0]):
        img_name = dataio.index2imagename(batch_index[ii, 0])
        pose_raw = batch_poses[ii, :].flatten()
        crimg_line = ''.join("%12.4f" % x for x in pose_raw)
        fanno.write(img_name + crimg_line + '\n')
