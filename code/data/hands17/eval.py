import os
# import sys
# from importlib import import_module
import numpy as np
import matplotlib.pyplot as mpplot
from . import ops as dataops
from . import io as dataio
# from . import draw as datadraw


# def compare_error(thedata, fname_echt, fname_pred):
#     """ NOTE: the number of predictions might be smaller
#         return: FxJ, l2 error matrix
#     """
#     error_l = []
#     with open(fname_echt, 'r') as file_s, \
#             open(fname_pred, 'r') as file_t:
#         sour_lines = [x.strip() for x in file_s.readlines()]
#         targ_lines = [x.strip() for x in file_t.readlines()]
#         for li, line_t in enumerate(targ_lines):
#             name_s, pose_s = dataio.parse_line_annot(sour_lines[li])
#             name_t, pose_t = dataio.parse_line_annot(line_t)
#             if name_s != name_t:
#                 print('ERROR - different names: {} - {}'.format(
#                     name_s, name_t))
#                 return
#             p3d_s = dataops.d2z_to_raw(pose_s, thedata)
#             p3d_t = dataops.d2z_to_raw(pose_t, thedata)
#             error_l.append(np.sqrt(
#                 np.sum((p3d_s - p3d_t) ** 2, axis=1)
#             ))
#     # return np.expand_dims(np.stack(error_l, axis=0), axis=0)
#     return np.stack(error_l, axis=0)


def compare_error_h5(thedata, fname_echt, fname_pred):
    """ NOTE: the number of predictions might be smaller
        return: FxJ, l2 error matrix
    """
    import h5py
    with h5py.File(fname_echt, 'r') as file_s, \
            h5py.File(fname_pred, 'r') as file_t:
        num_line = file_t['index'].shape[0]
        # print(np.vstack((file_s['index'][:num_line], file_t['index'][:num_line])))
        if 0 != np.sum(file_s['index'][:num_line] - file_t['index'][:num_line]):
            print('ERROR - different names index!')
            return
        poses_s = file_s['poses'][:num_line, ...].reshape(-1, 3)
        poses_t = file_t['poses'][()].reshape(-1, 3)
        p3d_s = dataops.d2z_to_raw(poses_s, thedata)
        p3d_t = dataops.d2z_to_raw(poses_t, thedata)
        error = np.sqrt(
            np.sum((p3d_s - p3d_t) ** 2, axis=1)
        )
        return error.reshape(-1, thedata.join_num)


def draw_mean_error_distribution(errors, ax):
    """ errors: FxJ """
    err_mean = np.mean(errors, axis=1)
    ax.hist(
        err_mean, 100,
        weights=np.ones_like(err_mean) * 100. / err_mean.size)
    ax.set_ylabel('Percentage (%)')
    ax.set_ylim(bottom=0)
    # ax.set_ylim([0, 100])
    ax.set_xlabel('Mean error of single frame (mm)')
    # ax.set_xlim(left=0)


def draw_error_percentage_curve(errors, methods, ax, labels=None):
    """ errors: MxFxJ """
    err_max = np.max(errors, axis=-1)
    num_v = err_max.shape[1]
    num_m = err_max.shape[0]
    if len(methods) != num_m:
        print('ERROR - method dimension not matching!')
        return
    if labels is None:
        labels = methods
    percent = np.arange(num_v + 1) * 100 / num_v  # 0 .. num_v
    err_max = np.concatenate((
        np.zeros(shape=(num_m, 1)),
        np.sort(err_max, 1)),
        axis=1
    )
    maxmean = np.max(np.mean(errors, axis=1), axis=1)
    err_max_draw = 100
    for err, label in zip(err_max, labels):
        err_cut = err[err_max_draw > err]
        nc = len(err_cut)
        percent_cut = percent[:nc]
        if err_max_draw < nc:
            step = int(nc / err_max_draw)
            err_cut = err_cut[::step]
            percent_cut = percent_cut[::step]
        ax.plot(
            # err, percent,
            err_cut, percent_cut,
            linewidth=2.0,
            label=label
        )
    # ax.plot(
    #     err_max, np.tile(percent, (num_m, 1)),
    #     '-',
    #     linewidth=2.0
    # )
    ax.set_ylabel('Percentage (%)')
    ax.set_ylim([0, 100])
    ax.set_xlabel('Maximal error of single joint (mm)')
    ax.set_xlim(left=0)
    ax.set_xlim(right=err_max_draw)
    # ax.legend(labels, loc='lower right')
    handles, _ = ax.get_legend_handles_labels()
    _, handles, labels = zip(*sorted(
        zip(maxmean, handles, labels),
        key=lambda t: t[0]))
    ax.legend(handles, labels, loc='lower right')
    return maxmean


def draw_error_per_joint(
    errors, methods, ax,
        join_name=None, labels=None, draw_std=False):
    """ errors: MxFxJ """
    err_mean = np.mean(errors, axis=1)
    err_max = np.max(errors, axis=1)
    err_min = np.min(errors, axis=1)
    num_v = err_max.shape[1]
    num_m = err_max.shape[0]
    if len(methods) != num_m:
        print('ERROR - dimension not matching!')
        return
    if labels is None:
        labels = methods
    err_mean = np.append(
        err_mean,
        np.mean(err_mean, axis=1).reshape(-1, 1), axis=1)
    err_max = np.append(
        err_max,
        np.mean(err_max, axis=1).reshape(-1, 1), axis=1)
    err_min = np.append(
        err_min,
        np.mean(err_min, axis=1).reshape(-1, 1), axis=1)
    err_m2m = np.concatenate((
        np.expand_dims(err_mean - err_min, -1),
        np.expand_dims(err_max - err_mean, -1)
    ), axis=-1)

    jid = np.arange(num_v + 1)
    jtick = join_name
    if join_name is None:
        jtick = [str(x) for x in jid]
        jtick[-1] = 'Mean'
    else:
        jtick.append('Mean')
    wb = 0.2
    wsl = float(num_m - 1) * wb / 2
    jloc = jid * (num_m + 2) * wb
    for ei, err in enumerate(err_mean):
        if draw_std:
            ax.bar(
                jloc + wb * ei - wsl, err, width=wb, align='center',
                yerr=err_m2m,
                error_kw=dict(ecolor='gray', lw=1, capsize=3, capthick=2),
                label=labels[ei]
            )
        else:
            ax.bar(
                jloc + wb * ei - wsl, err, width=wb, align='center',
                label=labels[ei]
            )
    ylim_top = max(np.max(err_mean[:, 0:7]), np.max(err_mean))
    ax.set_ylabel('Mean error (mm)')
    ax.set_ylim(0, ylim_top + float(num_m) * ylim_top * 0.1)
    ax.set_xlim(jloc[0] - wsl - 0.5, jloc[-1] + wsl + 0.5)
    mpplot.xticks(jloc, jtick, rotation='vertical')
    ax.margins(0.1)
    # ax.legend(labels, loc='upper left')
    meanmean = err_mean[:, -1]
    handles, _ = ax.get_legend_handles_labels()
    _, handles, labels = zip(*sorted(
        zip(meanmean, handles, labels),
        key=lambda t: t[0]))
    ax.legend(handles, labels, loc='upper left')
    return meanmean


def evaluate_detection(thedata, model_name, predict_dir, predict_file):
    pass
    # print('evaluating {} ...'.format(model_name))
    # print('figures saved')


def evaluate_poses(thedata, model_name, predict_dir, predict_file):
    print('evaluating {} ...'.format(model_name))

    errors = compare_error_h5(
        thedata,
        thedata.annotation_test,
        predict_file
    )
    # mpplot.gcf().clear()
    fig = mpplot.figure(figsize=(2 * 5, 1 * 5))
    draw_mean_error_distribution(
        errors, mpplot.gca())
    fig.tight_layout()
    fname = '{}_error_dist.png'.format(model_name)
    mpplot.savefig(os.path.join(predict_dir, fname))
    mpplot.close(fig)
    errors = np.expand_dims(errors, axis=0)
    # mpplot.gcf().clear()
    fig = mpplot.figure(figsize=(2 * 5, 1 * 5))
    draw_error_percentage_curve(
        errors, [model_name], mpplot.gca())
    fig.tight_layout()
    fname = '{}_error_rate.png'.format(model_name)
    mpplot.savefig(os.path.join(predict_dir, fname))
    mpplot.close(fig)
    # mpplot.gcf().clear()
    fig = mpplot.figure(figsize=(2 * 5, 1 * 5))
    err_mean = draw_error_per_joint(
        errors, [model_name], mpplot.gca(), thedata.join_name)
    fig.tight_layout()
    fname = '{}_error_bar.png'.format(model_name)
    mpplot.savefig(os.path.join(predict_dir, fname))
    mpplot.close(fig)

    print('figures saved: {}'.format(fname))
    return np.max(np.mean(errors, axis=1)), err_mean

    # draw_sum = 3
    # draw_i = 1
    # fig_size = (draw_sum * 5, 5)
    # mpplot.subplots(nrows=1, ncols=draw_sum, figsize=fig_size)
    # mpplot.subplot(1, draw_sum, draw_i)
    # draw_i += 1
    # draw_error_percentage_curve(errors, mpplot.gca())
    # mpplot.subplot(1, draw_sum, draw_i)
    # draw_error_per_joint(errors, mpplot.gca(), thedata.join_name)
    # draw_i += 1
    # mpplot.subplot(1, draw_sum, draw_i)
    # draw_mean_error_distribution(errors, mpplot.gca())
    # draw_i += 1


def evaluate_hands(thedata, model_name, predict_dir, predict_file):
    # from sklearn.metrics import precision_recall_curve
    pass
