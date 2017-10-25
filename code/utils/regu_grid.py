import numpy as np
import matplotlib.pyplot as mpplot
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from colour import Color


class iso_box:
    def __init__(self):
        self.cen = [0, 0, 0]
        self.len = 0  # half side length
        self.evals = np.ones(3)
        self.evecs = np.eye(3)

    def pick(self, points3):
        points3_trans = self.transform(points3)
        # cmin = self.cen - self.len
        # cmax = self.cen + self.len
        cmin = - np.ones(3) * self.len
        cmax = np.ones(3) * self.len
        conds = np.logical_and(
            np.all(cmin < points3_trans, axis=1),
            np.all(cmax > points3_trans, axis=1)
        )
        return points3[conds, :]

    def build(self, points3, m=0.1):
        # from sklearn.decomposition import PCA
        # pca = PCA(n_components=3)
        # points3_trans_skl = pca.fit_transform(points3)  # bias might be wrong
        # print(pca.explained_variance_)
        # print(np.ptp(points3_trans_skl, axis=0))
        self.cen = np.mean(points3, axis=0)
        C = np.cov((points3 - self.cen), rowvar=False)
        evals, evecs = np.linalg.eigh(C)
        idx = np.argsort(evals)[::-1]
        evecs = evecs[idx, :]
        evals = evals[idx]
        print(evals)
        # print(np.linalg.norm(evecs, axis=0))
        points3_trans = self.transform(points3)
        ptp = np.ptp(points3_trans, axis=0)
        idx = np.argsort(ptp)[::-1]
        evecs = evecs[idx, :]
        ptp = ptp[idx]
        print(ptp)
        self.len = np.max(ptp) / 2
        print(self.len)
        self.evals = evals
        self.evecs = evecs
        if 1 > m:
            m = self.len * m
        self.add_margin(m)
        return points3_trans

    def transform(self, points3):
        # points3_trans = np.dot(points3 - self.cen, self.evecs) / np.sqrt(self.evals)
        points3_trans = np.dot(points3 - self.cen, self.evecs)
        return points3_trans

    def add_margin(self, m):
        self.len += m

    def project_pca(self, ps3_pca, roll=0):
        ar3 = np.roll(np.arange(3), roll)
        cid = ar3[:2]
        did = 2
        idx = np.argsort(ps3_pca[:, did])
        ps3_pca = ps3_pca[idx, :]
        coord = ps3_pca[:, cid]
        depth = ps3_pca[:, did]
        cll = np.array([-self.len, -self.len])
        sidelen = np.ceil(2 * self.len).astype(int) + 1
        img = np.zeros((sidelen, sidelen))
        coord = np.floor(coord - cll + 0.5).astype(int)
        img[coord[:, 0], coord[:, 1]] = depth + self.len
        return img

    @staticmethod
    def get_corners(cen, len):
        cmin = cen - len
        cmax = cen + len
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

    def draw(self, ax, alpha='0.25'):
        corners = iso_box.get_corners(self.cen, self.len)
        faces = [
            [corners[0], corners[3], corners[2], corners[1]],
            [corners[0], corners[1], corners[5], corners[4]],
            [corners[0], corners[4], corners[7], corners[3]],
            [corners[4], corners[5], corners[6], corners[7]],
            [corners[2], corners[3], corners[7], corners[6]],
            [corners[1], corners[2], corners[6], corners[5]]
        ]
        ax.add_collection3d(Poly3DCollection(
            faces,
            linewidths=1, edgecolors='red',
            facecolors=Color('orange').rgb, alpha=alpha
        ))
        ax.scatter(
            corners[:, 0], corners[:, 1], corners[:, 2],
            color=Color('cyan').rgb, alpha=0.5, marker='o')

    def draw_wire(self, ax):
        corners = iso_box.get_corners(
            self.transform(self.cen),
            self.len
        )
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


class regu_grid:
    pass


if __name__ == "__main__":
    box = iso_box()
    # points3 = np.random.randn(1000, 3)
    points3 = np.random.rand(1000, 3) * 6
    box.build(points3)
    fig = mpplot.figure()
    ax = Axes3D(fig)
    box.draw(ax)
    ax.scatter(points3[:, 0], points3[:, 1], points3[:, 2])
    # points3_trans = box.transform(points3)
    # ax.scatter(points3_trans[:, 0], points3_trans[:, 1], points3_trans[:, 2])
    mpplot.show()
