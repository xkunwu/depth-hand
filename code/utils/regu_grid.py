""" Hand in Depth
    https://github.com/xkunwu/depth-hand
"""
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

    def build(self, points3, m=0.):
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
        # self.pcnt = np.zeros(
        #     shape=(self.step, self.step, self.step))

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

    def from_cube(self, cube, step):
        self.cll = cube.cen - cube.sidelen
        # self.cll = - np.ones(3)
        self.step = step
        self.cellen = cube.sidelen * 2. / step
        # self.cellen = 2. / step
        # self.pcnt = np.zeros(shape=(self.step, self.step, self.step))

    # def subdivide(self, volume, step):
    #     self.cll = volume
    #     self.sidelen = volume.sidelen
    #     self.step = step
    #     self.pcnt = np.zeros(shape=(self.step, self.step, self.step))

    def putit(self, points3):
        index = np.floor((points3 - self.cll) / self.cellen).astype(int)
        return index
        # return np.clip(index, 0, self.step - 1)

    def fill(self, points3):
        step = self.step
        pcnt = np.zeros((step ** 3))
        indices = self.putit(points3)
        seqid = np.ravel_multi_index(indices.T, (step, step, step))
        unid, counts = np.unique(seqid, return_counts=True)
        pcnt[unid] = counts.astype(float)
        if 1e-2 < np.max(pcnt):
            pcnt /= np.max(pcnt)  # normalized density
        pcnt = pcnt.reshape((step, step, step))
        # pcnt = np.zeros(shape=(step, step, step))
        # for index in indices:
        #     pcnt[index[0], index[1], index[2]] += 1.
        # pcnt /= np.max(pcnt)  # normalized density
        # self.pcnt = pcnt
        return pcnt

    def hit(self, points3):
        vxmap = np.zeros(shape=(self.step, self.step, self.step))
        indices = self.putit(points3)
        vxmap[indices[:, 0], indices[:, 1], indices[:, 2]] = 1.
        # self.vxmap = vxmap
        return vxmap

    def prow_anchor_single(self, points3, wsizes):
        scale_base = self.cellen * self.step
        anchors = np.empty((self.step, self.step, self.step, 4))
        for rr in np.arange(self.step):
            for cc in np.arange(self.step):
                for dd in np.arange(self.step):
                    cen = self.voxen(np.array((rr, cc, dd)))
                    anchors[rr, cc, dd, 0:3] = (points3 - cen) / scale_base
                    anchors[rr, cc, dd, 3] = np.log(wsizes / scale_base)
        return anchors.flatten()

    def yank_anchor_single(self, index, anchors):
        anchors_res = anchors.reshape(self.step, self.step, self.step, 4)
        anchors = anchors_res[index[0], index[1], index[2], :]
        centre = self.voxen(index)
        delta = anchors[0:3]
        scale = anchors[3]
        scale_base = self.cellen * self.step
        points3 = (delta * scale_base) + centre
        if 1. < scale or -10. > scale:
            print('Warning - localizer window looks bad: {}'.format(scale))
            scale = 0.
        wsizes = np.exp(scale) * scale_base
        return points3, wsizes

    def voxen(self, index):
        return self.cll + self.cellen * (index.astype(float) + 0.5)

    def fetch(self, index):
        cen = float(index) - self.cll
        return grid_cell(cen, self.cellen)

    def slice_ortho(self, vxmap, roll=0):
        ar3 = np.arange(3)
        if 0 < roll:
            ar3 = np.roll(ar3, roll)
        # each row is in an axis
        indices = np.argwhere(1e-2 < vxmap).T
        # indices = np.array(np.unravel_index(
        #     (vxhit), (self.step, self.step, self.step)))
        indices2d = indices[ar3[:2], :]
        seq2d = np.ravel_multi_index(
            indices2d, (self.step, self.step))
        seq2d_unique = np.unique(seq2d)
        coord = np.array(np.unravel_index(
            (seq2d_unique), (self.step, self.step))).T
        return coord[:, ::-1]

    def draw_slice(self, ax, coord, cellen=None):
        from utils.iso_boxes import iso_rect
        if cellen is None:
            cellen = self.cellen
        for c in coord:
            rect = iso_rect(c * cellen - (cellen / 2), cellen)
            rect.draw(ax)

    def draw_map(self, ax, vxmap):
        from utils.iso_boxes import iso_cube
        indices = np.argwhere(1e-2 < vxmap)
        for index in indices:
            cube_vox = iso_cube(
                self.voxen(index),
                (self.cellen / 2) * 0.8
            )
            corners = cube_vox.get_corners()
            # iso_cube.draw_cube_wire(ax, corners)
            iso_cube.draw_cube_face(ax, corners)

        from mayavi import mlab
        from colour import Color
        xx, yy, zz = np.where(1e-2 < vxmap)
        mlab.points3d(
            xx, yy, zz,
            mode="cube",
            color=Color('khaki').rgb,
            scale_factor=1)
        mlab.show()


class latice_image:
    def __init__(self, image_size=np.array((6, 8)), step=2):
        self.cellen = image_size / step  # cell side length
        self.step = step
        # self.pcnt = np.zeros(shape=(self.step, self.step))

    def dump(self):
        return np.concatenate((
            np.array([self.step]), self.cellen,
        ))

    def load(self, args):
        self.step = int(args[0])
        self.cellen = args[1:3]
        # self.pcnt = np.zeros(shape=(self.step, self.step))

    def show_dims(self):
        print(self.step, self.cellen)

    def putit(self, points2):
        return np.floor(
            points2.astype(float) / self.cellen - 1e-6
        ).astype(int)

    def fill(self, points2):
        """ here only one-shot """
        pcnt = np.zeros(shape=(self.step, self.step))
        num_p = points2.shape[0]
        indices = self.putit(points2)
        for index in indices:
            pcnt[index[0], index[1]] += 1.
        pcnt /= num_p  # probability
        # p50 = np.where(0.5 < pcnt)
        pmax = np.where(np.max(pcnt) == pcnt)
        pcnt = np.zeros(shape=(self.step, self.step))
        pcnt[pmax] = 1.
        # self.pcnt = pcnt
        return pcnt

    def prow_anchor_single(self, points2, wsizes):
        """ collect 3 parameters relative to each anchor
            - center locations: x, y
            - window size: s
        """
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
        if 1. < scale or -10. > scale:
            print('Warning - localizer window looks bad: {}'.format(scale))
            scale = 0.
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
