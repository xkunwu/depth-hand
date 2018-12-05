import numpy as np
from collections import deque
from matplotlib.mlab import PCA
from utils.iso_boxes import iso_cube


class MomentTrack:
    def __init__(self, steprange):
        self.track_rank = 3
        """ set probablity to 0.01 when move too far
            e^(-a*x^2) = 0.01
            a = -log(0.01) / x^2
        """
        self.steprange = steprange
        self.alpha = - np.log(0.01) / (steprange * steprange)
        self.isocen = deque(maxlen=self.track_rank)  # center of detections
        self.cendis = deque(maxlen=self.track_rank)  # center displacement
        self.delta = deque(maxlen=self.track_rank)  # displacement distance
        # for ci in range(self.track_rank):
        #     self.cendis.append(np.zeros(3))
        #     self.cendis.append(0)

    def get_prob(self, delta):
        return np.exp(-self.alpha * (delta * delta))

    def get_momentum(self, delta):
        if len(self.delta) > 0:
            return abs(delta - self.delta[-1])
        else:
            raise ValueError('should have values')
            return 0

    def get_cen_moment(self, cen, delta):
        if len(self.isocen) == 0:
            raise ValueError('should have values')
        else:
            mm = self.get_momentum(delta)
            prob = self.get_prob(mm)
            cen_m = (1 - prob) * self.isocen[-1] + prob * cen
            print(delta, mm, prob)
            return cen_m

    def update(self, cen):
        if len(self.cendis) == 0:
            self.isocen.append(cen)
            self.cendis.append(0)
            self.delta.append(0)
            return cen
        disp = cen - self.isocen[-1]
        delta = np.linalg.norm(disp)
        if self.steprange < delta:
            return False
        cen_m = self.get_cen_moment(cen, delta)
        self.isocen.append(cen_m)
        self.cendis.append(self.isocen[-1] - self.isocen[-2])
        self.delta.append(np.linalg.norm(self.cendis[-1]))
        return cen_m

    def clear(self):
        self.isocen.clear()
        self.cendis.clear()
        self.delta.clear()

    def test(self):
        fib = [1, 1, 2, 3, 5, 8, 13, 21, 34]
        cens = np.vstack((
            fib, np.zeros(len(fib)), np.zeros(len(fib)))).T
        for c in cens:
            cm = self.update(c)
            print(c, cm)
            print(self.isocen)
            print(self.cendis)
            print(self.delta)
            print("=========\n")
        self.clear()


class HandCenter:
    def __init__(self):
        pass

    def simple_mean(self, points):
        return np.mean(points, axis=0)

    def mean_shift(self, points):
        return


class hand_finder:
    def __init__(self, args, caminfo_ir):
        self.args = args
        self.caminfo_ir = caminfo_ir
        self.estr = ""
        maxm = caminfo_ir.region_size / 1  # maximum move is 12mm
        self.tracker = MomentTrack(maxm)
        # self.tracker.test()
        self.cen_ext = HandCenter()

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
            print(self.estr)
            self.tracker.clear()
            return False
        zrs = z0 + caminfo.region_size
        # zrs = z0 + caminfo.region_size * 2
        in_id = np.where(np.logical_and(z0 - 0.01 < dlist, dlist < zrs))
        xin, yin = np.unravel_index(in_id, dimg.shape)
        # p2z = np.vstack((yin, xin, dlist[in_id])).T  # TESTDATA!!
        p2z = np.vstack((xin, yin, dlist[in_id])).T
        p3d = self.args.data_ops.d2z_to_raw(
            p2z, self.caminfo_ir)
        cen = self.cen_ext.simple_mean(p3d)
        # cen_m = self.tracker.update(cen)
        # print(cen, cen_m)
        cen_m = cen
        if cen_m is False:
            # self.tracker.clear()
            return False
        cube = iso_cube(
            cen_m,
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
