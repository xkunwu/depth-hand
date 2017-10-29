import os
from itertools import islice
from functools import partial
from multiprocessing.dummy import Pool as ThreadPool
import ops as dataops
import io as dataio


def put2d_worker(args, thedata, image_dir, batchallot):
    bi = args[0]
    line = args[1]
    img_name, pose_raw = dataio.parse_line_annot(line)
    img = dataio.read_image(os.path.join(image_dir, img_name))
    img_crop_resize, rescen = dataops.crop_resize(
        img, pose_raw, thedata)
    # pose2d = dataops.raw_to_2d(pose_raw, thedata, rescen)
    batchallot.batch_index[bi, :] = dataio.imagename2index(img_name)
    batchallot.batch_frame[bi, :, :] = img_crop_resize
    batchallot.batch_poses[bi, :] = pose_raw.flatten().T
    batchallot.batch_resce[bi, :] = rescen


def put2d_mt(
        fanno, thedata, image_dir, batchallot
):
    next_n_lines = list(islice(fanno, batchallot.batch_size))
    if not next_n_lines:
        return -1
    num_line = len(next_n_lines)
    thread_pool = ThreadPool()
    thread_pool.map(
        partial(put2d_worker, thedata=thedata, image_dir=image_dir, batchallot=batchallot),
        zip(range(num_line), next_n_lines))
    # import numpy as np
    # import copy
    # test_copy = copy.deepcopy(batchallot)
    # for bi, line in enumerate(next_n_lines):
    #     put2d_worker((bi, line), thedata, image_dir, test_copy)
    # print(np.linalg.norm(batchallot.batch_index - test_copy.batch_index))
    # print(np.linalg.norm(batchallot.batch_frame - test_copy.batch_frame))
    # print(np.linalg.norm(batchallot.batch_poses - test_copy.batch_poses))
    # print(np.linalg.norm(batchallot.batch_resce - test_copy.batch_resce))
    return num_line
