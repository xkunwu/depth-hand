import os
import numpy as np
from itertools import islice
from . import ops as dataops
from . import io as dataio
from utils.iso_boxes import iso_cube
from utils.regu_grid import latice_image


# def prow_dirtsdf(line, image_dir, caminfo):
#     img_name, pose_raw = dataio.parse_line_annot(line)
#     img = dataio.read_image(os.path.join(image_dir, img_name))
#     pcnt, resce = dataops.fill_grid(
#         img, pose_raw, caminfo.crop_size, caminfo)
#     befs = dataops.trunc_belief(pcnt)
#     resce3 = resce[0:4]
#     pose_pca = dataops.raw_to_pca(pose_raw, resce3)
#     index = dataio.imagename2index(img_name)
#     return (index, befs,
#             pose_pca.flatten().T, resce)
#
#
# def yank_dirtsdf(pose_local, resce, caminfo):
#     resce3 = resce[0:4]
#     return dataops.pca_to_raw(pose_local, resce3)


# def prow_truncdf(line, image_dir, caminfo):
#     img_name, pose_raw = dataio.parse_line_annot(line)
#     img = dataio.read_image(os.path.join(image_dir, img_name))
#     pcnt, resce = dataops.fill_grid(
#         img, pose_raw, caminfo.crop_size, caminfo)
#     tdf = dataops.prop_dist(pcnt)
#     resce3 = resce[0:4]
#     pose_pca = dataops.raw_to_pca(pose_raw, resce3)
#     index = dataio.imagename2index(img_name)
#     return (index, np.expand_dims(tdf, axis=-1),
#             pose_pca.flatten().T, resce)
#
#
# def yank_truncdf(pose_local, resce, caminfo):
#     resce3 = resce[0:4]
#     return dataops.pca_to_raw(pose_local, resce3)


def prow_localizer2(line, image_dir, caminfo):
    img_name, pose_raw = dataio.parse_line_annot(line)
    img = dataio.read_image(os.path.join(image_dir, img_name))
    img_rescale = dataops.resize_localizer(img, caminfo)
    anchors, resce = dataops.generate_anchors_2d(
        img, pose_raw, caminfo.anchor_num, caminfo)
    index = dataio.imagename2index(img_name)
    return (index,
            np.expand_dims(img_rescale, axis=-1),
            anchors.T, resce)


def yank_localizer2_rect(index, anchors, caminfo):
    lattice = latice_image(
        np.array(caminfo.image_size).astype(float),
        caminfo.anchor_num)
    points2, wsizes = lattice.yank_anchor_single(
        index,
        anchors
    )
    z_cen = dataops.estimate_z(
        caminfo.region_size, wsizes, caminfo.focal[0])
    # print(np.append(points2, [wsizes, z_cen]).reshape(1, -1))
    centre = dataops.d2z_to_raw(
        np.append(points2, z_cen).reshape(1, -1),
        caminfo
    )
    return points2, wsizes, centre.flatten()


def yank_localizer2(pose_local, resce, caminfo):
    anchor_num = caminfo.anchor_num ** 2
    # label = np.argmax(pose_local[:anchor_num]).astype(int)
    pcnt = pose_local[:anchor_num].reshape(
        caminfo.anchor_num, caminfo.anchor_num)
    index = np.array(np.unravel_index(np.argmax(pcnt), pcnt.shape))
    # convert logits to probability, due to network design
    logits = pcnt[index[0], index[1]]
    confidence = 1 / (1 + np.exp(-logits))
    anchors = pose_local[anchor_num:]
    points2, wsizes, centre = yank_localizer2_rect(
        index, anchors, caminfo)
    return centre, index, confidence


def prow_localizer3(line, image_dir, caminfo):
    img_name, pose_raw = dataio.parse_line_annot(line)
    img = dataio.read_image(os.path.join(image_dir, img_name))
    pcnt, anchors, resce = dataops.voxelize_depth(
        img, pose_raw, caminfo.crop_size, caminfo.anchor_num, caminfo)
    index = dataio.imagename2index(img_name)
    return (index, np.expand_dims(pcnt, axis=-1),
            anchors.T, resce)


def yank_localizer3(pose_local, resce, caminfo):
    cube = iso_cube()
    cube.load(resce)
    centre = cube.cen
    return centre


# def prow_conv3d(line, image_dir, caminfo):
#     img_name, pose_raw = dataio.parse_line_annot(line)
#     img = dataio.read_image(os.path.join(image_dir, img_name))
#     pcnt, resce = dataops.fill_grid(
#         img, pose_raw, caminfo.crop_size, caminfo)
#     resce3 = resce[0:4]
#     pose_pca = dataops.raw_to_pca(pose_raw, resce3)
#     index = dataio.imagename2index(img_name)
#     return (index, np.expand_dims(pcnt, axis=-1),
#             pose_pca.flatten().T, resce)
#
#
# def yank_conv3d(pose_local, resce, caminfo):
#     resce3 = resce[0:4]
#     return dataops.pca_to_raw(pose_local, resce3)


# def prow_ortho3v(line, image_dir, caminfo):
#     img_name, pose_raw = dataio.parse_line_annot(line)
#     img = dataio.read_image(os.path.join(image_dir, img_name))
#     img_crop_resize, resce = dataops.proj_ortho3(
#         img, pose_raw, caminfo)
#     resce3 = resce[0:4]
#     pose_pca = dataops.raw_to_pca(pose_raw, resce3)
#     index = dataio.imagename2index(img_name)
#     return (index, img_crop_resize,
#             pose_pca.flatten().T, resce)
#
#
# def yank_ortho3v(pose_local, resce, caminfo):
#     resce3 = resce[0:4]
#     return dataops.pca_to_raw(pose_local, resce3)


# def prow_cleaned(line, image_dir, caminfo):
#     img_name, pose_raw = dataio.parse_line_annot(line)
#     img = dataio.read_image(os.path.join(image_dir, img_name))
#     img_crop_resize, resce = dataops.crop_resize_pca(
#         img, pose_raw, caminfo)
#     resce3 = resce[0:4]
#     pose_pca = dataops.raw_to_pca(pose_raw, resce3)
#     index = dataio.imagename2index(img_name)
#     return (index, np.expand_dims(img_crop_resize, axis=-1),
#             pose_pca.flatten().T, resce)
#
#
# def yank_cleaned(pose_local, resce, caminfo):
#     resce3 = resce[0:4]
#     return dataops.pca_to_raw(pose_local, resce3)


# def prow_cropped(line, image_dir, caminfo):
#     img_name, pose_raw = dataio.parse_line_annot(line)
#     img = dataio.read_image(os.path.join(image_dir, img_name))
#     img_crop_resize, resce = dataops.crop_resize(
#         img, pose_raw, caminfo)
#     resce3 = resce[0:4]
#     pose_local = dataops.raw_to_local(pose_raw, resce3)
#     index = dataio.imagename2index(img_name)
#     return (index, np.expand_dims(img_crop_resize, axis=-1),
#             pose_local.flatten().T, resce)
#
#
# def yank_cropped(pose_local, resce, caminfo):
#     resce3 = resce[0:4]
#     return dataops.local_to_raw(pose_local, resce3)


# def put_worker(args, worker, image_dir, caminfo, batchallot):
#     bi = args[0]
#     line = args[1]
#     index, frame, poses, resce = worker(line, image_dir, caminfo)
#     batchallot.batch_index[bi, :] = index
#     batchallot.batch_frame[bi, ...] = frame
#     batchallot.batch_poses[bi, :] = poses
#     batchallot.batch_resce[bi, :] = resce


def test_puttensor(
    next_n_lines, put_worker, image_dir, model_inst,
        caminfo, data_module, batchallot):
    import copy
    test_copy = copy.deepcopy(batchallot)
    for bi, line in enumerate(next_n_lines):
        put_worker(
            (bi, line), image_dir, model_inst,
            caminfo, data_module, test_copy)
    print('this is TEST only!!! DO NOT forget to write using mp version')


def puttensor_mt(
    fanno, put_worker, image_dir, model_inst,
        caminfo, data_module, batchallot):
    next_n_lines = list(islice(fanno, batchallot.store_size))
    if not next_n_lines:
        return -1
    num_line = len(next_n_lines)

    # from timeit import default_timer as timer
    # from datetime import timedelta
    # time_s = timer()
    # # test_copy = test_puttensor(
    # #     next_n_lines, worker, image_dir, caminfo, batchallot)
    # test_copy = test_puttensor(
    #     next_n_lines, put_worker, image_dir, model_inst,
    #     caminfo, data_module, batchallot)
    # time_e = str(timedelta(seconds=timer() - time_s))
    # print('single tread time: {}'.format(time_e))
    # return num_line

    from functools import partial
    from multiprocessing.dummy import Pool as ThreadPool
    # time_s = timer()
    thread_pool = ThreadPool()
    thread_pool.map(
        partial(
            put_worker, image_dir=image_dir, model_inst=model_inst,
            caminfo=caminfo, data_module=data_module,
            batchallot=batchallot),
        zip(range(num_line), next_n_lines))
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
