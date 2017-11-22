import numpy as np
import matplotlib.pyplot as mpplot
import ops as dataops
import io as dataio


def compare_error(thedata, fname_echt, fname_pred):
    """ NOTE: the number of predictions might be smaller
        return: FxJ, l2 error matrix
    """
    error_l = []
    with open(fname_echt, 'r') as file_s, \
            open(fname_pred, 'r') as file_t:
        sour_lines = [x.strip() for x in file_s.readlines()]
        targ_lines = [x.strip() for x in file_t.readlines()]
        for li, line_t in enumerate(targ_lines):
            name_s, pose_s = dataio.parse_line_annot(sour_lines[li])
            name_t, pose_t = dataio.parse_line_annot(line_t)
            if name_s != name_t:
                print('different names: {} - {}'.format(name_s, name_t))
                return
            p3d_s = dataops.d2z_to_raw(pose_s, thedata)
            p3d_t = dataops.d2z_to_raw(pose_t, thedata)
            error_l.append(np.sqrt(
                np.sum((p3d_s - p3d_t) ** 2, axis=1)
            ))
    # return np.expand_dims(np.stack(error_l, axis=0), axis=0)
    return np.stack(error_l, axis=0)


def draw_mean_error_distribution(errors, ax):
    """ errors: FxJ """
    err_mean = np.mean(errors, axis=1)
    mpplot.hist(
        err_mean, 100,
        weights=np.ones_like(err_mean) * 100. / err_mean.size)
    ax.set_ylabel('Percentage (%)')
    ax.set_ylim(bottom=0)
    # ax.set_ylim([0, 100])
    ax.set_xlabel('Mean error of single frame (mm)')
    # ax.set_xlim(left=0)


def draw_error_percentage_curve(errors, methods, ax):
    """ errors: MxFxJ """
    err_max = np.max(errors, axis=-1)
    num_v = err_max.shape[1]
    num_m = err_max.shape[0]
    if len(methods) != num_m:
        print('ERROR - dimension not matching!')
        return
    percent = np.arange(num_v + 1) * 100 / num_v
    err_max = np.concatenate((
        np.zeros(shape=(num_m, 1)),
        np.sort(err_max, 1)),
        axis=1
    )
    for err in err_max:
        mpplot.plot(
            err, percent,
            '-',
            linewidth=2.0
        )
    # mpplot.plot(
    #     err_max, np.tile(percent, (num_m, 1)),
    #     '-',
    #     linewidth=2.0
    # )
    ax.set_ylabel('Percentage (%)')
    ax.set_ylim([0, 100])
    ax.set_xlabel('Maximal error of single joint (mm)')
    ax.set_xlim(left=0)
    ax.set_xlim(right=100)
    mpplot.legend(methods, loc='lower right')
    mpplot.tight_layout()
    # ax.set_xlim(right=50)
    # mpplot.show()


def draw_error_per_joint(errors, methods, ax, join_name=None, draw_std=False):
    """ errors: MxFxJ """
    err_mean = np.mean(errors, axis=1)
    err_max = np.max(errors, axis=1)
    err_min = np.min(errors, axis=1)
    num_v = err_max.shape[1]
    num_m = err_max.shape[0]
    if len(methods) != num_m:
        print('ERROR - dimension not matching!')
        return
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
            mpplot.bar(
                jloc + wb * ei - wsl, err, width=wb, align='center',
                yerr=err_m2m,
                error_kw=dict(ecolor='gray', lw=1, capsize=3, capthick=2)
            )
        else:
            mpplot.bar(
                jloc + wb * ei - wsl, err, width=wb, align='center'
            )
    ylim_top = max(np.max(err_mean[:, 0:7]), np.max(err_mean))
    ax.set_ylabel('Mean error (mm)')
    ax.set_ylim(0, ylim_top + float(num_m) * ylim_top * 0.1)
    ax.set_xlim(jloc[0] - wsl - 0.5, jloc[-1] + wsl + 0.5)
    mpplot.xticks(jloc, jtick, rotation='vertical')
    mpplot.margins(0.1)
    mpplot.tight_layout()
    mpplot.legend(methods, loc='upper left')
    # mpplot.show()
