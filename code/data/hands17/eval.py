""" Hand in Depth
    https://github.com/xkunwu/depth-hand
"""
import numpy as np
from data.eval_abc import eval_abc
from .ops import ops as dataops


class eval(eval_abc):
    """ collect evaluation statistics
    """
    # @classmethod
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

    @classmethod
    def compare_error_h5(cls, thedata, fname_echt, fname_pred):
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
