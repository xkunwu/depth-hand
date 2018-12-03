import numpy as np
from matplotlib.mlab import PCA
from utils.iso_boxes import iso_cube


class hand_finder:
    def __init__(self, args, caminfo_ir):
        self.args = args
        self.caminfo_ir = caminfo_ir
        self.estr = ""

    def simp_crop(self, dimg):
        caminfo = self.caminfo_ir
        dlist = dimg.ravel()
        dpart10 = np.partition(
            dlist[caminfo.z_range[0] + 0.01 < dlist],
            10)
        z0 = dpart10[9]
        # print(dlist.shape, np.min(dlist), np.max(dlist))
        # print("10th closest point: {}".format(z0)) # 193.10437004605774
        if z0 > caminfo.crop_range:
            self.estr = "hand out of detection range"
            return False
        zrs = z0 + caminfo.region_size
        # zrs = z0 + caminfo.region_size * 2
        in_id = np.where(np.logical_and(z0 - 0.01 < dlist, dlist < zrs))
        xin, yin = np.unravel_index(in_id, dimg.shape)
        # p2z = np.vstack((yin, xin, dlist[in_id])).T  # TESTDATA!!
        p2z = np.vstack((xin, yin, dlist[in_id])).T
        p3d = self.args.data_ops.d2z_to_raw(
            p2z, self.caminfo_ir)
        pmax = np.max(p3d, axis=0)
        pmin = np.min(p3d, axis=0)
        self.cen = (pmax + pmin) / 2
        self.sidelen = np.max(pmax - pmin) / 2
        cube = iso_cube(
            (pmax + pmin) / 2,
            self.caminfo_ir.region_size)
        # print(cube.dump())  # [120.0000 -158.0551 -116.6658 240.0000]
        return cube

    def region_grow(self, dimg):
        dimg[::4, ::4]
        caminfo = self.caminfo_ir
        dlist = dimg.ravel()
        dpart10 = np.partition(
            dlist[caminfo.z_range[0] + 0.01 < dlist],
            10)
        z0 = dpart10[9]
        # z0 = np.min(dlist[caminfo.z_range[0] + 0.01 < dlist])
        # print("10th closest point: {}".format(z0)) # 193.10437004605774
        if z0 > caminfo.crop_range:
            self.estr = "hand out of detection range"
            return False
        zrs = z0 + caminfo.region_size * 2
        dlist = dlist[np.logical_and(z0 < dlist, dlist < zrs)]
        print(dlist.shape, np.min(dlist), np.max(dlist))
        # (69975,) 193.22935669333674 433.07873282174114
        numbin = 12
        dsub = np.random.choice(dlist, numbin * 1000)
        # print(dsub.shape, np.min(dsub), np.max(dsub))
        # (69975,) 193.22935669333674 433.07873282174114
        bins = np.linspace(np.min(dsub), np.max(dsub), numbin)
        digi = np.digitize(dsub, bins)
        eig_l = np.zeros(numbin)
        for bi in range(numbin):
            bdep = dsub[bi == digi]
