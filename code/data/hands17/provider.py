import os
import numpy as np
from itertools import islice
from . import ops as dataops
from . import io as dataio


def prow_dirtsdf(line, image_dir, caminfo):
    img_name, pose_raw = dataio.parse_line_annot(line)
    img = dataio.read_image(os.path.join(image_dir, img_name))
    pcnt, resce = dataops.fill_grid(img, pose_raw, caminfo.crop_size, caminfo)
    befs = dataops.trunc_belief(pcnt)
    resce3 = resce[0:8]
    pose_pca = dataops.raw_to_pca(pose_raw, resce3)
    return (img_name, befs,
            pose_pca.flatten().T, resce)


def yank_dirtsdf(pose_local, resce):
    resce3 = resce[0:8]
    return dataops.pca_to_raw(pose_local, resce3)


def prow_truncdf(line, image_dir, caminfo):
    img_name, pose_raw = dataio.parse_line_annot(line)
    img = dataio.read_image(os.path.join(image_dir, img_name))
    pcnt, resce = dataops.fill_grid(img, pose_raw, caminfo.crop_size, caminfo)
    tdf = dataops.prop_dist(pcnt)
    resce3 = resce[0:8]
    pose_pca = dataops.raw_to_pca(pose_raw, resce3)
    return (img_name, np.expand_dims(tdf, axis=3),
            pose_pca.flatten().T, resce)


def yank_truncdf(pose_local, resce):
    resce3 = resce[0:8]
    return dataops.pca_to_raw(pose_local, resce3)


def prow_conv3d(line, image_dir, caminfo):
    img_name, pose_raw = dataio.parse_line_annot(line)
    img = dataio.read_image(os.path.join(image_dir, img_name))
    pcnt, resce = dataops.fill_grid(img, pose_raw, caminfo.crop_size, caminfo)
    resce3 = resce[0:8]
    pose_pca = dataops.raw_to_pca(pose_raw, resce3)
    return (img_name, np.expand_dims(pcnt, axis=3),
            pose_pca.flatten().T, resce)


def yank_conv3d(pose_local, resce):
    resce3 = resce[0:8]
    return dataops.pca_to_raw(pose_local, resce3)


def prow_ortho3v(line, image_dir, caminfo):
    img_name, pose_raw = dataio.parse_line_annot(line)
    img = dataio.read_image(os.path.join(image_dir, img_name))
    img_crop_resize, resce = dataops.proj_ortho3(
        img, pose_raw, caminfo)
    resce3 = resce[0:8]
    pose_pca = dataops.raw_to_pca(pose_raw, resce3)
    return (img_name, img_crop_resize,
            pose_pca.flatten().T, resce)


def yank_ortho3v(pose_local, resce):
    resce3 = resce[0:8]
    return dataops.pca_to_raw(pose_local, resce3)


def prow_cleaned(line, image_dir, caminfo):
    img_name, pose_raw = dataio.parse_line_annot(line)
    img = dataio.read_image(os.path.join(image_dir, img_name))
    img_crop_resize, resce = dataops.crop_resize_pca(
        img, pose_raw, caminfo)
    resce3 = resce[0:8]
    pose_pca = dataops.raw_to_pca(pose_raw, resce3)
    return (img_name, np.expand_dims(img_crop_resize, axis=2),
            pose_pca.flatten().T, resce)


def yank_cleaned(pose_local, resce):
    resce3 = resce[0:8]
    return dataops.pca_to_raw(pose_local, resce3)


def prow_cropped(line, image_dir, caminfo):
    img_name, pose_raw = dataio.parse_line_annot(line)
    img = dataio.read_image(os.path.join(image_dir, img_name))
    img_crop_resize, resce = dataops.crop_resize(
        img, pose_raw, caminfo)
    resce3 = resce[3:7]
    pose_local = dataops.raw_to_local(pose_raw, resce3)
    return (img_name, np.expand_dims(img_crop_resize, axis=2),
            pose_local.flatten().T, resce)


def yank_cropped(pose_local, resce):
    resce3 = resce[3:7]
    return dataops.local_to_raw(pose_local, resce3)


def put_worker(args, worker, image_dir, caminfo, batchallot):
    bi = args[0]
    line = args[1]
    img_name, frame, poses, resce = worker(line, image_dir, caminfo)
    batchallot.batch_index[bi, :] = dataio.imagename2index(img_name)
    batchallot.batch_frame[bi, ...] = frame
    batchallot.batch_poses[bi, :] = poses
    batchallot.batch_resce[bi, :] = resce


def test_puttensor(next_n_lines, worker, image_dir, caminfo, batchallot):
    import copy
    test_copy = copy.deepcopy(batchallot)
    for bi, line in enumerate(next_n_lines):
        put_worker((bi, line), worker, image_dir, caminfo, test_copy)
    print('this is TEST only!!! DO NOT forget to write using mp version')


def puttensor_mt(fanno, worker, image_dir, caminfo, batchallot):
    next_n_lines = list(islice(fanno, batchallot.store_size))
    if not next_n_lines:
        return -1
    num_line = len(next_n_lines)

    # from timeit import default_timer as timer
    # from datetime import timedelta
    # time_s = timer()
    # test_copy = test_puttensor(next_n_lines, worker, image_dir, caminfo, batchallot)
    # time_e = str(timedelta(seconds=timer() - time_s))
    # print('single tread time: {}'.format(time_e))
    # return num_line

    from functools import partial
    from multiprocessing.dummy import Pool as ThreadPool
    # time_s = timer()
    thread_pool = ThreadPool()
    thread_pool.map(
        partial(put_worker, worker=worker, image_dir=image_dir,
                caminfo=caminfo, batchallot=batchallot),
        zip(range(num_line), next_n_lines))
    # thread_pool.map(
    #     partial(worker, caminfo=caminfo, image_dir=image_dir, batchallot=batchallot),
    #     zip(range(num_line), next_n_lines))
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


def write2d(fanno, yanker, batch_index, batch_resce, batch_poses):
    for ii in range(batch_index.shape[0]):
        img_name = dataio.index2imagename(batch_index[ii, 0])
        pose_local = batch_poses[ii, :].reshape(-1, 3)
        resce = batch_resce[ii, :]
        pose_raw = yanker(pose_local, resce)
        crimg_line = ''.join("%12.4f" % x for x in pose_raw.flatten())
        fanno.write(img_name + crimg_line + '\n')
