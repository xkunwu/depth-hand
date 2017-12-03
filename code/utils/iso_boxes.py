import numpy as np
# from pyquaternion import Quaternion
import matplotlib.pyplot as mpplot
import matplotlib.patches as mppatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from colour import Color


class iso_rect:
    def __init__(self, cll=np.zeros(2), sidelen=1., m=0.):
        self.cll = cll
        self.sidelen = sidelen
        self.add_margan(m)

    def dump(self):
        return np.append(self.sidelen, self.cll)

    def load(self, args):
        self.cll = args[1:3]
        self.sidelen = args[0]

    def show_dims(self):
        print(self.cll, self.sidelen)

    def pick(self, points2):
        cmin = self.cll
        cmax = self.cll + self.sidelen
        conds = np.logical_and(
            np.all(cmin < points2, axis=1),
            np.all(cmax > points2, axis=1)
        )
        return conds

    def build(self, points2, m=0.):
        pmin = np.min(points2, axis=0)
        pmax = np.max(points2, axis=0)
        cen = (pmin + pmax) / 2
        self.sidelen = np.max(pmax - pmin)
        # cen = np.mean(points2, axis=0)
        # self.sidelen = np.max(np.ptp(points2, axis=0))
        self.add_margan(m)
        self.cll = cen - self.sidelen / 2

    def add_margan(self, m=0.):
        if 1 > m and -1 < m:
            m = self.sidelen * m
        self.sidelen += m

    def print_image(self, coord, value):
        coord = np.floor(coord - self.cll + 0.5).astype(int)
        sidelen = np.ceil(self.sidelen).astype(int) + 1
        img = np.zeros((sidelen, sidelen))
        # img[coord[:, 1], coord[:, 0]] = value  # reverse coordinates!
        img[coord[:, 0], coord[:, 1]] = value
        return img

    def draw(self, color=Color('orange').rgb):
        mpplot.gca().add_patch(mppatches.Rectangle(
            self.cll, self.sidelen, self.sidelen,
            linewidth=1, facecolor='none',
            edgecolor=color
        ))


class iso_aabb:
    def __init__(self, cll=np.zeros(3), sidelen=1., m=0.):
        self.cll = cll
        self.sidelen = sidelen
        self.add_margan(m)

    def dump(self):
        return np.append(self.sidelen, self.cll)

    def load(self, args):
        self.cll = args[1:4]
        self.sidelen = args[0]

    def show_dims(self):
        print(self.cen, self.sidelen)

    def build(self, points3, m=0.):
        pmin = np.min(points3, axis=0)
        pmax = np.max(points3, axis=0)
        cen = (pmin + pmax) / 2
        self.sidelen = np.max(pmax - pmin)
        self.add_margan(m)
        self.cll = cen - self.sidelen / 2

    def add_margan(self, m=0.):
        if 1 > m and -1 < m:
            m = self.sidelen * m
        self.sidelen += m

    def transform(self, points3):
        """ world --> local """
        return (points3 - (self.cll + self.sidelen / 2)) / self.sidelen

    def transform_inv(self, points3):
        """ local --> world """
        return points3 * self.sidelen + (self.cll + self.sidelen / 2)


class iso_cube:
    def __init__(self, cen=np.zeros(3), sidelen=1., m=0.):
        self.cen = cen
        self.sidelen = sidelen  # half side length
        # self.evecs = np.eye(3)
        self.add_margan(m)

    def dump(self):
        return np.concatenate((
            np.array([self.sidelen]), self.cen,
            # Quaternion(matrix=self.evecs).elements
        ))

    def load(self, args):
        self.sidelen = args[0]
        self.cen = args[1:4]
        # self.evecs = Quaternion(args[4:8]).rotation_matrix

    def show_dims(self):
        print(self.cen, self.sidelen)
        # print(self.evecs)

    def get_sidelen(self):
        return np.ceil(2 * self.sidelen).astype(int) + 1

    def pick(self, points3):
        """ only make sense when picked in the local coordinates """
        points3_trans = self.transform(points3)
        cmin = - np.ones(3)
        cmax = np.ones(3)
        conds = np.logical_and(
            np.all(cmin < points3_trans, axis=1),
            np.all(cmax > points3_trans, axis=1)
        )
        return points3[conds, :], points3_trans[conds, :]

    def build(self, points3, m=0.):
        pmax = np.max(points3, axis=0)
        pmin = np.min(points3, axis=0)
        self.cen = (pmax + pmin) / 2
        self.sidelen = np.max(pmax - pmin) / 2
        # self.evecs = np.eye(3)
        self.add_margan(m)
        return self.transform(points3)

    # def build_pca(self, points3, m=0.6):
    #     self.cen = np.mean(points3, axis=0)
    #     evals, evecs = np.linalg.eig(np.cov(points3.T))
    #     # _, evecs = np.linalg.eig(np.cov(points3.T))
    #     idx = np.argsort(evals)[::-1]
    #     evecs = evecs[idx, :]
    #     # evals = evals[idx]
    #     # print(evals)
    #     points3_trans = np.dot(points3 - self.cen, evecs)
    #     pmin = np.min(points3_trans, axis=0)
    #     pmax = np.max(points3_trans, axis=0)
    #     self.sidelen = np.max(pmax - pmin) / 2
    #     if 0 > np.linalg.det(evecs):
    #         evecs[2, :] *= -1
    #     self.evecs = evecs
    #     # self.qrot = Quaternion(matrix=evecs)
    #     self.add_margan(m)
    #     # return points3_trans
    #     return points3_trans / self.sidelen

    def add_margan(self, m=0.):
        if 1 > m and -1 < m:
            m = self.sidelen * m
        self.sidelen += m

    def transform(self, points3):
        """ world --> local """
        # return np.dot(points3 - self.cen, self.evecs)
        # return np.dot(points3 - self.cen, self.evecs) / self.sidelen
        return (points3 - self.cen) / self.sidelen
        # return self.qrot.rotate(points3 - self.cen)

    def transform_inv(self, points3):
        """ local --> world """
        # return np.dot(points3, self.evecs.T) + self.cen
        # return np.dot(points3 * self.sidelen, self.evecs.T) + self.cen
        return (points3 * self.sidelen) + self.cen
        # return self.qrot.inverse.rotate(points3 - self.cen)

    # def trans_scale(self, points3, sizel):
    #     """ world --> image """
    #     return np.dot(points3 - self.cen, self.evecs) * sizel / self.sidelen
    #
    # def trans_scale_inv(self, points3, sizel):
    #     """ image --> world """
    #     return np.dot(points3 * self.sidelen / sizel, self.evecs.T) + self.cen

    def project_pca(self, ps3_pca, roll=0, sort=True):
        ar3 = np.arange(3)
        if 0 < roll:
            ar3 = np.roll(ar3, roll)
        cid = ar3[:2]
        did = 2
        if sort is True:
            idx = np.argsort(ps3_pca[..., did])
            ps3_pca = ps3_pca[idx, ...]
        coord = ps3_pca[:, cid]
        cll = - np.ones(2)
        coord = (coord - cll) / 2
        depth = (ps3_pca[:, did] + 1) / 2
        return coord, depth

    def print_image(self, coord, depth, sizel):
        img = np.zeros((sizel, sizel))
        coord *= 0.999999
        img[
            np.floor(coord[:, 1] * sizel).astype(int),
            np.floor(coord[:, 0] * sizel).astype(int),
        ] = depth  # reverse coordinates!
        # img[
        #     np.floor(coord[:, 0] * sizel).astype(int),
        #     np.floor(coord[:, 1] * sizel).astype(int),
        # ] = depth
        return img

    # def proj_rect(self, raw_to_2d_fn, caminfo):
    #     # c3a = np.array([
    #     #     np.append(
    #     #         self.cen[:2] - self.sidelen,
    #     #         self.cen[2] - self.sidelen),
    #     #     np.append(
    #     #         self.cen[:2] + self.sidelen,
    #     #         self.cen[2] - self.sidelen)
    #     # ])  # near z-plane
    #     c3a = np.array([
    #         np.append(self.cen[:2] - self.sidelen, self.cen[2]),
    #         np.append(self.cen[:2] + self.sidelen, self.cen[2])
    #     ])  # central z-plane
    #     c2a = raw_to_2d_fn(c3a, caminfo)
    #     cll = c2a[0, :]
    #     ctr = c2a[1, :]
    #     return iso_rect(cll, np.max(ctr - cll))

    def proj_rects_3(self, raw_to_2d_fn, caminfo):
        c3a_l = []
        c3a_l.append(np.array([
            np.append(self.cen[:2] - self.sidelen, self.cen[2]),
            np.append(self.cen[:2] + self.sidelen, self.cen[2])
        ]))  # central z-plane
        c3a_l.append(np.array([
            np.append(self.cen[:2] - self.sidelen, self.cen[2] - self.sidelen),
            np.append(self.cen[:2] + self.sidelen, self.cen[2] - self.sidelen)
        ]))  # near z-plane
        c3a_l.append(np.array([
            np.append(self.cen[:2] - self.sidelen, self.cen[2] + self.sidelen),
            np.append(self.cen[:2] + self.sidelen, self.cen[2] + self.sidelen)
        ]))  # far z-plane

        def convert(c3a):
            c2a = raw_to_2d_fn(c3a, caminfo)
            cll = c2a[0, :]
            ctr = c2a[1, :]
            return iso_rect(cll, np.max(ctr - cll))

        rects = [convert(c3a) for c3a in c3a_l]
        return rects

    def get_corners(self):
        cmin = self.cen - self.sidelen
        cmax = self.cen + self.sidelen
        # cmin = - np.ones(3) * self.sidelen
        # cmax = np.ones(3) * self.sidelen
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

    @staticmethod
    def draw_cube_face(corners, alpha='0.25'):
        faces = [
            [corners[0], corners[3], corners[2], corners[1]],
            [corners[0], corners[1], corners[5], corners[4]],
            [corners[0], corners[4], corners[7], corners[3]],
            [corners[4], corners[5], corners[6], corners[7]],
            [corners[2], corners[3], corners[7], corners[6]],
            [corners[1], corners[2], corners[6], corners[5]]
        ]
        mpplot.gca().add_collection3d(Poly3DCollection(
            faces,
            linewidths=1, edgecolors='red',
            facecolors=Color('orange').rgb, alpha=alpha
        ))
        mpplot.gca().scatter(
            corners[:, 0], corners[:, 1], corners[:, 2],
            color=Color('cyan').rgb, alpha=0.5, marker='o')

    @staticmethod
    def draw_cube_wire(corners):
        ring_b = np.array([
            corners[0], corners[1], corners[2], corners[3], corners[0]
        ])
        ring_u = np.array([
            corners[4], corners[5], corners[6], corners[7], corners[4]
        ])
        mpplot.plot(
            ring_b[:, 0], ring_b[:, 1], ring_b[:, 2],
            '-',
            linewidth=2.0,
            color=Color('orange').rgb
        )
        mpplot.plot(
            ring_u[:, 0], ring_u[:, 1], ring_u[:, 2],
            '-',
            linewidth=2.0,
            color=Color('orange').rgb
        )
        mpplot.plot(
            [corners[0, 0], corners[4, 0]],
            [corners[0, 1], corners[4, 1]],
            [corners[0, 2], corners[4, 2]],
            '-',
            linewidth=2.0,
            color=Color('orange').rgb
        )
        mpplot.plot(
            [corners[1, 0], corners[5, 0]],
            [corners[1, 1], corners[5, 1]],
            [corners[1, 2], corners[5, 2]],
            '-',
            linewidth=2.0,
            color=Color('orange').rgb
        )
        mpplot.plot(
            [corners[2, 0], corners[6, 0]],
            [corners[2, 1], corners[6, 1]],
            [corners[2, 2], corners[6, 2]],
            '-',
            linewidth=2.0,
            color=Color('orange').rgb
        )
        mpplot.plot(
            [corners[3, 0], corners[7, 0]],
            [corners[3, 1], corners[7, 1]],
            [corners[3, 2], corners[7, 2]],
            '-',
            linewidth=2.0,
            color=Color('orange').rgb
        )


if __name__ == "__main__":
    cube = iso_cube()
    # points3 = np.random.randn(1000, 3)
    points3 = np.random.rand(1000, 3) * 6
    cube.build(points3)
    fig = mpplot.figure()
    ax = Axes3D(fig)
    cube.draw()
    ax.scatter(points3[:, 0], points3[:, 1], points3[:, 2])
    # points3_trans = cube.transform(points3)
    # ax.scatter(points3_trans[:, 0], points3_trans[:, 1], points3_trans[:, 2])
    mpplot.show()
