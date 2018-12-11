""" Hand in Depth
    https://github.com/xkunwu/depth-hand
"""
import numpy as np
from collections import deque
import cv2
from sklearn.decomposition import PCA
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
            # print(delta, mm, prob)
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
        if 10 > points.size:
            return False
        else:
            return np.mean(points, axis=0)

    def shape_prior(self, points):
        if 10 > points.size:
            return False
        z = points[:, 2]
        zmax = np.max(z)
        zmin = np.min(z)
        num_bin = 12
        bins = np.linspace(zmax, zmin, num_bin + 1)  # 12 bins, decreasing
        pca = PCA(n_components=1, svd_solver='arpack')
        digi = np.digitize(z, bins, right=False)
        evs = np.zeros(num_bin)
        wpos = 0
        for bi in range(0, num_bin):
            psec = points[digi == (bi + 1)]
            if 10 > psec.size:
                continue
            pca.fit(psec)
            evs[bi] = pca.singular_values_[0]
            if 0 == bi:
                continue
            if (evs[bi] > evs[bi - 1] * 1.1):
                wpos = bi  # include previous section
                # wpos = bi - 1  # include previous section
                break
        # print(bins)
        # print(evs)
        # print(wpos, bins[wpos])
        points_wrist = points[digi == (wpos + 1)]
        if 10 > points_wrist.size:
            return False
        points_upper = points[z < bins[wpos]]
        if 10 > points_upper.size:
            return False
        mean_upper = np.mean(points_upper, axis=0)
        mean_wrist = np.mean(points_wrist, axis=0)
        mean_tweak = (mean_upper - mean_wrist) * 0.2
        return mean_upper + mean_tweak


class hand_locator:
    def __init__(self, args, caminfo):
        self.args = args
        self.caminfo = caminfo
        self.estr = ""
        maxm = caminfo.region_size / 1  # maximum move is 120mm
        self.tracker = MomentTrack(maxm)
        # self.tracker.test()
        self.cen_ext = HandCenter()

    def simp_crop(self, dimg):
        caminfo = self.caminfo
        dimg_f = cv2.bilateralFilter(
            dimg.astype(np.float32),
            5, 30, 30)
        dlist = dimg_f.ravel()
        if 10 > len(dlist):
            self.estr = "hand out of detection range"
            print(self.estr)
            self.tracker.clear()
            return False
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
        in_id = np.where(np.logical_and(z0 - 0.01 < dlist, dlist < zrs))
        if 10 > len(in_id[0]):
            self.estr = "not enough points in range"
            print(self.estr)
            self.tracker.clear()
            return False
        xin, yin = np.unravel_index(in_id, dimg_f.shape)
        ## FetchHands17!! {
        # p2z = np.vstack((yin, xin, dlist[in_id])).T
        ## }
        ## live stream {
        p2z = np.vstack((xin, yin, dlist[in_id])).T
        ## }
        p3d = self.args.data_ops.d2z_to_raw(
            p2z, self.caminfo)
        ## FetchHands17!! {
        # cube = iso_cube()
        # cube.extent_center(p3d)  # FetchHands17!!
        # cube.sidelen = self.caminfo.region_size
        ## }
        ## find center {
        # cen = self.cen_ext.simple_mean(p3d)
        cen = self.cen_ext.shape_prior(p3d)
        if cen is False:
            self.estr = "center not found"
            print(self.estr)
            self.tracker.clear()
            return False
        cen_m = self.tracker.update(cen)
        # cen_m = cen
        print(cen, cen_m)
        if cen_m is False:
            self.estr = "lost track"
            print(self.estr)
            self.tracker.clear()
            return False
        cube = iso_cube(
            cen_m,
            self.caminfo.region_size)
        ## }
        # # print(cube.dump())  # [120.0000 -158.0551 -116.6658 240.0000]
        return cube

    def region_grow(self, dimg):
        dimg[::4, ::4]
        caminfo = self.caminfo
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
