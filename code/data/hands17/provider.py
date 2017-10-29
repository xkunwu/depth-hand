import os
from itertools import islice
import ops as dataops
import io as dataio


def put2d(
        thedata, fanno, image_dir, batch_size,
        batch_index, batch_frame, batch_poses, batch_resce
):
    next_n_lines = list(islice(fanno, batch_size))
    if not next_n_lines:
        return -1
    if len(next_n_lines) < batch_size:
        return -2
    for bi, line in enumerate(next_n_lines):
        img_name, pose_raw = dataio.parse_line_annot(line)
        img = dataio.read_image(os.path.join(image_dir, img_name))
        img_crop_resize, rescen = dataops.crop_resize(
            img, pose_raw, thedata)
        # pose2d = dataops.raw_to_2d(pose_raw, thedata, rescen)
        batch_index[bi, :] = dataio.imagename2index(img_name)
        batch_frame[bi, :, :] = img_crop_resize
        batch_poses[bi, :] = pose_raw.flatten().T
        batch_resce[bi, :] = rescen
    return 0
