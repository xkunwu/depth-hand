import numpy as np


class grid_cell:
    def __init__(self, cll=np.zeros(3), sidelen=1.):
        self.cll = cll
        self.sidelen = sidelen

    def dump(self):
        return np.append(self.sidelen, self.cll)

    def load(self, args):
        self.cll = args[1:4]
        self.sidelen = args[0]

    def show_dims(self):
        print(self.cll, self.sidelen)

    def pick(self, points3):
        cmin = self.cll
        cmax = self.cll + self.sidelen
        conds = np.logical_and(
            np.all(cmin < points3, axis=1),
            np.all(cmax > points3, axis=1)
        )
        return points3[conds, :]

    def build(self, points3, m=0.1):
        pmin = np.min(points3, axis=0)
        pmax = np.max(points3, axis=0)
        cen = (pmin + pmax) / 2
        self.sidelen = np.max(pmax - pmin)
        if 1 > m and -1 < m:
            m = self.sidelen * m
        self.sidelen += m
        self.cll = cen - self.sidelen / 2

    def get_corners(self):
        cmin = self.cll
        cmax = self.cll + self.sidelen
        corners = np.array([
            cmin,
            [cmax[0], cmin[1], cmin[2]],
            [cmax[0], cmax[1], cmin[2]],
            [cmin[0], cmax[1], cmin[2]],
            [cmin[0], cmin[1], cmax[2]],
            [cmax[0], cmin[1], cmax[2]],
            cmax,
            [cmin[0], cmax[1], cmax[2]]
        ])
        return corners


class regu_grid:
    def __init__(self, cll=np.zeros(3), step=2, cellen=1.):
        self.cll = cll
        self.cellen = cellen  # cell side length
        self.step = step
        self.pcnt = np.zeros(
            shape=(self.step, self.step, self.step))

    def dump(self):
        return np.concatenate((
            np.array([self.cellen]), self.cen,
            np.array([self.step])
        ))

    def load(self, args):
        self.cellen = args[0]
        self.cen = args[1:4]
        self.step = args[4]

    def show_dims(self):
        print(self.cll, self.cellen, self.step)

    def from_cube(self, cube, step, m=0.01):
        if 1 > m and -1 < m:
            m = cube.sidelen * m
        cubelen = cube.sidelen + m
        self.cll = np.zeros(3) - cubelen
        self.step = step
        self.cellen = cubelen * 2 / step
        self.pcnt = np.zeros(shape=(self.step, self.step, self.step))

    # def subdivide(self, volume, step):
    #     self.cll = volume
    #     self.sidelen = volume.sidelen
    #     self.step = step
    #     self.pcnt = np.zeros(shape=(self.step, self.step, self.step))

    def putit(self, points3):
        return np.floor((points3 - self.cll) / self.cellen).astype(int)

    def fill(self, points3):
        indices = self.putit(points3)
        for index in indices:
            self.pcnt[index[0], index[1], index[2]] += 1.
        self.pcnt /= np.max(self.pcnt)  # normalized density
        return self.pcnt

    def voxen(self, index):
        return self.cll + self.cellen * (float(index) + 0.5)

    def fetch(self, index):
        cen = float(index) - self.cll
        return grid_cell(cen, self.cellen)

    def draw(self):
        pass


class latice_image:
    def __init__(self, image_size=np.array((6, 8)), step=2):
        self.cellen = image_size / step  # cell side length
        self.step = step
        self.pcnt = np.zeros(shape=(self.step, self.step))

    def dump(self):
        return np.concatenate((
            np.array([self.step]), self.cellen,
        ))

    def load(self, args):
        self.step = int(args[0])
        self.cellen = args[1:3]
        self.pcnt = np.zeros(shape=(self.step, self.step))

    def show_dims(self):
        print(self.step, self.cellen)

    def putit(self, points2):
        return np.floor(
            points2.astype(float) / self.cellen - 1e-6
        ).astype(int)

    def fill(self, points2):
        num_p = points2.shape[0]
        indices = self.putit(points2)
        for index in indices:
            self.pcnt[index[0], index[1]] += 1.
        self.pcnt /= num_p  # probability
        # p50 = np.where(0.5 < self.pcnt)
        pmax = np.where(np.max(self.pcnt) == self.pcnt)
        self.pcnt = np.zeros(shape=(self.step, self.step))
        self.pcnt[pmax] = 1.
        return self.pcnt

    def prow_anchor_single(self, points2, wsizes):
        # centre = self.voxen(self.putit(points2))
        # scale_base = np.max(self.cellen)
        # delta = (points2 - centre) / scale_base
        # scale = np.log(wsizes / scale_base)
        # return np.concatenate([delta.flatten(), np.array([scale])])
        scale_base = np.max(self.cellen) * self.step
        anchors = np.empty((self.step, self.step, 3))
        for rr in np.arange(self.step):
            for cc in np.arange(self.step):
                cen = self.voxen(np.array((rr, cc)))
                anchors[rr, cc, 0:2] = (points2 - cen) / scale_base
                anchors[rr, cc, 2] = np.log(wsizes / scale_base)
        return anchors.flatten()

    def yank_anchor_single(self, index, anchors):
        anchors_res = anchors.reshape(self.step, self.step, 3)
        anchors = anchors_res[index[0], index[1], :]
        centre = self.voxen(index)
        delta = anchors[0:2]
        scale = anchors[2]
        scale_base = np.max(self.cellen) * self.step
        # scale_base = np.max(self.cellen)
        points2 = (delta * scale_base) + centre
        wsizes = np.exp(scale) * scale_base
        return points2, wsizes

    def prow_anchor(self, points2, wsizes):
        pass
        # param = np.zeros(shape=(3, self.step, self.step))
        # num_p = points2.shape[0]
        # indices = self.putit(points2)
        # xx = np.repeat(np.arange(self.step), num_p).reshape(num_p, self.step)
        # xi = np.argmin(np.fabs(xx - indices[:, 0]), axis=0)
        # # xx = np.min(np.fabs(xx - indices[:, 0]), axis=0)
        # xx = np.repeat(xx[xi, :], self.step).reshape(self.step, self.step)
        # xx = np.repeat(np.arange(3), 3).reshape(3, 3)
        # yy = yy.T

    def voxen(self, index):
        return self.cellen * (index.astype(float) + 0.5)
