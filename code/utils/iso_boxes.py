import numpy as np
from pyquaternion import Quaternion
import matplotlib.pyplot as mpplot
import matplotlib.patches as mppatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from colour import Color


class iso_rect:
    def __init__(self, cll=np.zeros(2), len=0., m=0.):
        self.cll = cll
        self.len = len
        self.add_margan(m)

    def dump(self):
        return np.append(self.len, self.cll)

    def load(self, args):
        self.cll = args[1:3]
        self.len = args[0]

    def show_dims(self):
        print(self.cll, self.len)

    def pick(self, points2):
        """ only meaningful when picked in the local coordinates """
        cmin = self.cll
        cmax = self.cll + self.len
        conds = np.logical_and(
            np.all(cmin < points2, axis=1),
            np.all(cmax > points2, axis=1)
        )
        return conds

    def build(self, points2, m=0.1):
        pmin = np.min(points2, axis=0)
        pmax = np.max(points2, axis=0)
        cen = (pmin + pmax) / 2
        self.len = np.max(pmax - pmin)
        # cen = np.mean(points2, axis=0)
        # self.len = np.max(np.ptp(points2, axis=0))
        self.add_margan(m)
        self.cll = cen - self.len / 2

    def add_margan(self, m=0.1):
        if 1 > m and -1 < m:
            m = self.len * m
        self.len += m

    def print_image(self, coord, value):
        coord = np.floor(coord - self.cll + 0.5).astype(int)
        sidelen = np.ceil(self.len).astype(int) + 1
        img = np.zeros((sidelen, sidelen))
        img[coord[:, 1], coord[:, 0]] = value  # reverse coordinates!
        return img

    def draw(self, color=Color('orange').rgb):
        mpplot.gca().add_patch(mppatches.Rectangle(
            self.cll, self.len, self.len,
            linewidth=1, facecolor='none',
            edgecolor=color
        ))


class iso_aabb:
    def __init__(self, cll=np.zeros(3), len=0., m=0.):
        self.cll = cll
        self.len = len
        self.add_margan(m)

    def dump(self):
        return np.append(self.len, self.cll)

    def load(self, args):
        self.cll = args[1:3]
        self.len = args[0]

    def show_dims(self):
        print(self.cen, self.len)

    def build(self, points3, m=0.1):
        pmin = np.min(points3, axis=0)
        pmax = np.max(points3, axis=0)
        cen = (pmin + pmax) / 2
        self.len = np.max(pmax - pmin)
        self.add_margan(m)
        self.cll = cen - self.len / 2

    def add_margan(self, m=0.1):
        if 1 > m and -1 < m:
            m = self.len * m
        self.len += m


class iso_cube:
    def __init__(self, cen=np.zeros(3), len=0., m=0.):
        self.cen = cen
        self.len = len  # half side length
        self.evecs = np.eye(3)
        self.add_margan(m)

    def dump(self):
        return np.concatenate((
            np.array([self.len]), self.cen,
            Quaternion(matrix=self.evecs).elements
        ))

    def load(self, args):
        self.len = args[0]
        self.cen = args[1:4]
        self.evecs = Quaternion(args[4:8]).rotation_matrix

    def show_dims(self):
        print(self.cen, self.len)
        print(self.evecs)

    def get_sidelen(self):
        return np.ceil(2 * self.len).astype(int) + 1

    def pick(self, points3):
        """ only meaningful when picked in the local coordinates """
        points3_trans = self.transform(points3)
        cmin = - np.ones(3) * self.len
        cmax = np.ones(3) * self.len
        conds = np.logical_and(
            np.all(cmin < points3_trans, axis=1),
            np.all(cmax > points3_trans, axis=1)
        )
        return points3[conds, :]

    def build(self, points3, m=0.6):
        # from sklearn.decomposition import PCA
        # pca = PCA(n_components=3)
        # points3_trans_skl = pca.fit_transform(points3)  # bias might be wrong
        # print(pca.explained_variance_)
        # print(np.ptp(points3_trans_skl, axis=0))
        pmin = np.min(points3, axis=0)
        pmax = np.max(points3, axis=0)
        self.cen = (pmin + pmax) / 2
        # self.cen = np.mean(points3, axis=0)
        C = np.cov((points3 - self.cen), rowvar=False)
        # evals, evecs = np.linalg.eigh(C)
        _, evecs = np.linalg.eigh(C)
        # idx = np.argsort(evals)[::-1]
        # evecs = evecs[idx, :]
        # evals = evals[idx]
        # print(evals)
        points3_trans = np.dot(points3 - self.cen, evecs)
        # ptp = np.ptp(points3_trans, axis=0)
        ptp = pmax - pmin
        idx = np.argsort(ptp)[::-1]
        evecs = evecs[idx, :]
        ptp = ptp[idx]
        # print(ptp)
        self.len = np.max(ptp) / 2
        # print(self.len)
        self.evecs = evecs
        if 0 > np.linalg.det(evecs):
            evecs[2, :] *= -1
        # self.qrot = Quaternion(matrix=evecs)
        self.add_margan(m)
        return points3_trans

    def add_margan(self, m=0.1):
        if 1 > m and -1 < m:
            m = self.len * m
        self.len += m

    def transform(self, points3):
        """ to local coordinates """
        return np.dot(points3 - self.cen, self.evecs)
        # return self.qrot.rotate(points3 - self.cen)

    def transform_inv(self, points3):
        """ to world coordinates """
        return np.dot(points3, self.evecs.T) + self.cen
        # return self.qrot.inverse.rotate(points3 - self.cen)

    def project_pca(self, ps3_pca, roll=0, sort=True):
        ar3 = np.roll(np.arange(3), roll)
        cid = ar3[:2]
        did = 2
        if sort is True:
            idx = np.argsort(ps3_pca[:, did])
            ps3_pca = ps3_pca[idx, :]
        coord = ps3_pca[:, cid]
        cll = np.array([-self.len, -self.len])
        coord = np.floor(coord - cll + 0.5).astype(int)
        depth = ps3_pca[:, did] + self.len
        return coord, depth

    def print_image(self, coord, depth):
        sidelen = self.get_sidelen()
        img = np.zeros((sidelen, sidelen))
        img[coord[:, 1], coord[:, 0]] = depth  # reverse coordinates!
        return img

    def get_corners(self):
        # cmin = self.cen - self.len
        # cmax = self.cen + self.len
        cmin = - np.ones(3) * self.len
        cmax = np.ones(3) * self.len
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

    def draw(self, corners, alpha='0.25'):
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

    def draw_wire(self, corners):
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
    box = iso_cube()
    # points3 = np.random.randn(1000, 3)
    points3 = np.random.rand(1000, 3) * 6
    box.build(points3)
    fig = mpplot.figure()
    ax = Axes3D(fig)
    box.draw()
    ax.scatter(points3[:, 0], points3[:, 1], points3[:, 2])
    # points3_trans = box.transform(points3)
    # ax.scatter(points3_trans[:, 0], points3_trans[:, 1], points3_trans[:, 2])
    mpplot.show()
