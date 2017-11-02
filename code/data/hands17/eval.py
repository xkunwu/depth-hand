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
    return np.array(error_l)


def draw_mean_error_distribution(errors, ax):
    err_mean = np.mean(errors, axis=1)
    mpplot.hist(
        err_mean, 100,
        weights=np.ones_like(err_mean) * 100. / err_mean.size)
    ax.set_ylabel('Percentage (%)')
    ax.set_ylim(bottom=0)
    # ax.set_ylim([0, 100])
    ax.set_xlabel('Mean error of single frame (mm)')
    # ax.set_xlim(left=0)


def draw_error_percentage_curve(errors, ax):
    err_max = np.max(errors, axis=1).tolist()
    num_v = len(err_max)
    percent = np.arange(num_v + 1) * 100 / num_v
    err_max = np.concatenate(([0], np.sort(err_max)))
    # fig, ax = mpplot.subplots()
    mpplot.plot(
        err_max, percent,
        '-',
        linewidth=2.0
    )
    ax.set_ylabel('Percentage (%)')
    ax.set_ylim([0, 100])
    ax.set_xlabel('Maximal error of single joint (mm)')
    ax.set_xlim(left=0)
    # ax.set_xlim(right=50)
    # mpplot.show()


def draw_error_per_joint(errors, ax, join_name=None):
    err_mean = np.mean(errors, axis=0)
    err_max = np.max(errors, axis=0)
    err_min = np.min(errors, axis=0)
    err_mean = np.append(err_mean, np.mean(err_mean))
    err_max = np.append(err_max, np.mean(err_max))
    err_min = np.append(err_min, np.mean(err_min))
    err_m2m = [
        (err_mean - err_min).tolist(),
        (err_max - err_mean).tolist()
    ]
    err_mean = err_mean.tolist()
    jid = range(len(err_mean))
    jtick = join_name
    if join_name is None:
        jtick = [str(x) for x in jid]
        jtick[-1] = 'Mean'
    else:
        jtick.append('Mean')
    # fig, ax = mpplot.subplots()
    mpplot.bar(
        jid, err_mean, yerr=err_m2m, align='center',
        error_kw=dict(ecolor='gray', lw=1, capsize=3, capthick=2)
    )
    ax.set_ylabel('Mean error (mm)')
    ax.set_ylim(bottom=0)
    ax.set_xlim([-1, 22])
    mpplot.xticks(jid, jtick, rotation='vertical')
    mpplot.margins(0.1)
    # mpplot.show()
