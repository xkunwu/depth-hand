import numpy as np


class grid_cell():
    def __init__(self, cll=np.zeros(3), len=1.):
        self.cll = cll
        self.len = len

    def dump(self):
        return np.append(self.len, self.cll)

    def load(self, args):
        self.cll = args[1:4]
        self.len = args[0]

    def show_dims(self):
        print(self.cll, self.len)

    def pick(self, points3):
        cmin = self.cll
        cmax = self.cll + self.len
        conds = np.logical_and(
            np.all(cmin < points3, axis=1),
            np.all(cmax > points3, axis=1)
        )
        return points3[conds, :]

    def build(self, points3, m=0.1):
        pmin = np.min(points3, axis=0)
        pmax = np.max(points3, axis=0)
        cen = (pmin + pmax) / 2
        self.len = np.max(pmax - pmin)
        if 1 > m and -1 < m:
            m = self.len * m
        self.len += m
        self.cll = cen - self.len / 2

    def get_corners(self):
        cmin = self.cll
        cmax = self.cll + self.len
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
    def __init__(self, cll=np.zeros(3), step=2, len=1.):
        self.cll = cll
        self.len = len  # cell side length
        self.step = step
        self.pcnt = np.zeros(
            shape=(self.step, self.step, self.step))

    def dump(self):
        return np.concatenate((
            np.array([self.len]), self.cen,
            np.array([self.step])
        ))

    def load(self, args):
        self.len = args[0]
        self.cen = args[1:4]
        self.step = args[4]

    def show_dims(self):
        print(self.cll, self.len, self.step)

    def from_cube(self, cube, step, m=0.01):
        if 1 > m and -1 < m:
            m = cube.len * m
        cubelen = cube.len + m
        self.cll = np.zeros(3) - cubelen
        self.step = step
        self.len = cubelen * 2 / step
        self.pcnt = np.zeros(shape=(self.step, self.step, self.step))

    def putit(self, points3):
        return np.floor((points3 - self.cll) / self.len).astype(int)

    def fill(self, points3):
        indices = self.putit(points3)
        for index in indices:
            self.pcnt[index[0], index[1], index[2]] += 1.
        self.pcnt /= np.max(self.pcnt)

    def voxen(self, index):
        return self.cll + self.len * (float(index) + 0.5)

    def fetch(self, index):
        cen = float(index) - self.cll
        return grid_cell(cen, self.len)

    def draw(self):
        pass
