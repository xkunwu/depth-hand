import os
import io as dataio


def read_train(thedata):
    return open(thedata.training_annot_train, 'r')


def read_test(thedata):
    return open(thedata.training_annot_test, 'r')


def write_predict(thedata):
    return open(thedata.training_annot_predict, 'w')


def close(thedata, file):
    file.close()


def put2d(
        thedata, next_n_lines, batch_frame, batch_poses,
        image_names=None, batch_resce=None
):
    for bi, annot_line in enumerate(next_n_lines):
        img_name, pose_mat, rescen = dataio.parse_line_pose(
            annot_line)
        img = dataio.read_image(os.path.join(
            thedata.training_cropped, img_name))
        batch_frame[bi, :, :] = img
        batch_poses[bi, :] = pose_mat.flatten().T
        if image_names is not None:
            image_names.append(img_name)
        if batch_resce is not None:
            batch_resce[bi, :] = rescen
