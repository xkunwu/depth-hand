import numpy as np
import matplotlib.pyplot as mpplot
import matplotlib.image as mpimg
import matplotlib.collections as mcoll
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tfplot
from utils.coder import file_pack
from utils.regu_grid import regu_grid


def depth_cmap(cmap):
    mycmap = cmap
    mycmap._init()
    mycmap.set_under('y')
    return mycmap


def transparent_cmap(cmap, tmax=0.99, N=255):
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:, -1] = np.linspace(0, tmax, N + 4)
    return mycmap


def draw_edt2(fig, ax, edt2):
    img_edt2 = ax.imshow(edt2, transparent_cmap(mpplot.cm.jet))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img_edt2, cax=cax)


def figure_edt2(edt2):
    fig, ax = tfplot.subplots(figsize=(4, 4))
    draw_edt2(fig, ax, edt2)
    ax.axis('off')
    return fig

tfplot_edt2 = tfplot.wrap(figure_edt2, batch=False)


def draw_dist3(vxdist, voxize_crop, scale):
    from mayavi import mlab
    # should reverser y-axis
    mlab.figure(
        bgcolor=(1, 1, 1), fgcolor=(0., 0., 0.),
        size=(800, 800))
    xx, yy, zz = np.where(1e-2 < vxdist)
    vv = vxdist[np.where(1e-2 < vxdist)]
    xx *= scale
    yy *= scale
    zz *= scale
    # vc = transparent_cmap(mpplot.cm.jet)(vv)
    yy = voxize_crop - 1 - yy
    nodes = mlab.points3d(
        xx, yy, zz, vv,
        mode="cube", opacity=0.5,
        # color=Color('khaki').rgb,
        colormap='hot',
        scale_factor=0.9)
    nodes.module_manager.scalar_lut_manager.reverse_lut = True
    nodes.glyph.scale_mode = 'scale_by_vector'
    # nodes.mlab_source.dataset.point_data.scalars = vv
    mlab.gcf().scene.parallel_projection = True
    mlab.view(0, 0)
    mlab.gcf().scene.camera.zoom(1.5)


def draw_udir(vxdist, vxunit, voxize_crop, scale):
    from mayavi import mlab
    from colour import Color
    # should reverser y-axis
    mlab.figure(
        bgcolor=(1, 1, 1), fgcolor=(0., 0., 0.),
        size=(800, 800))
    xx, yy, zz = np.where(1e-2 < vxdist)
    vv = vxdist[np.where(1e-2 < vxdist)]
    xx *= scale
    yy *= scale
    zz *= scale
    # vc = transparent_cmap(mpplot.cm.jet)(vv)
    yy = voxize_crop - 1 - yy
    nodes = mlab.points3d(
        xx, yy, zz, vv,
        mode="cube", opacity=0.5,
        # color=Color('khaki').rgb,
        colormap='hot',
        scale_factor=0.9)
    nodes.module_manager.scalar_lut_manager.reverse_lut = True
    nodes.glyph.scale_mode = 'scale_by_vector'
    # nodes.mlab_source.dataset.point_data.scalars = vv
    xx, yy, zz = np.mgrid[
        # (scale / 2):(voxize_crop + (scale / 2)):scale,
        # (scale / 2):(voxize_crop + (scale / 2)):scale,
        # (scale / 2):(voxize_crop + (scale / 2)):scale]
        0:voxize_crop:scale,
        0:voxize_crop:scale,
        0:voxize_crop:scale]
    yy = voxize_crop - 1 - yy
    mlab.quiver3d(
        xx, yy, zz,
        vxunit[..., 0], -vxunit[..., 1], vxunit[..., 2],
        mode="arrow",
        color=Color('red').rgb,
        line_width=8, scale_factor=2)
    mlab.gcf().scene.parallel_projection = True
    mlab.view(0, 0)
    mlab.gcf().scene.camera.zoom(1.5)


def draw_uomap3d(fig, ax, vxcnt_crop, uomap):
    crop_size = vxcnt_crop.shape[0]
    hmap_size = uomap.shape[0]
    map_scale = crop_size / hmap_size
    vxcnt_hmap = vxcnt_crop[::2, ::2, ::2]
    grid = regu_grid(step=32)
    coord = grid.slice_ortho(vxcnt_hmap)
    grid.draw_slice(ax, coord, 2.)
    xx, yy = np.meshgrid(
        np.arange(0, crop_size, map_scale),
        np.arange(0, crop_size, map_scale))
    ax.quiver(
        xx, yy,
        np.squeeze(uomap[..., 0]),
        -np.squeeze(uomap[..., 1]),
        color='r', width=0.004, scale=20)
    # xx, yy, zz = np.meshgrid(
    #     np.arange(0, crop_size, map_scale),
    #     np.arange(0, crop_size, map_scale),
    #     np.arange(0, crop_size, map_scale))
    # ax.quiver(
    #     xx, yy, zz,
    #     uomap[..., 0], uomap[..., 1], uomap[..., 2],
    #     color='r', length=1.2, arrow_length_ratio=0.4)
    ax.set_xlim([0, crop_size])
    ax.set_ylim([0, crop_size])
    # ax.view_init(azim=180, elev=-90)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()


def figure_uomap3d(vxcnt_crop, uomap):
    fig, ax = tfplot.subplots(figsize=(4, 4))
    draw_uomap3d(fig, ax, vxcnt_crop, uomap)
    ax.axis('off')
    return fig

tfplot_uomap3d = tfplot.wrap(figure_uomap3d, batch=False)


def draw_vxmap(
    fig, ax, vxcnt_crop, vxmap,
        voxize_hmap, reduce_fn=np.sum, roll=0):
    crop_size = vxcnt_crop.shape[0]
    hmap_size = vxmap.shape[0]
    map_scale = crop_size / hmap_size
    vxcnt_hmap = vxcnt_crop[::map_scale, ::map_scale, ::map_scale]
    grid = regu_grid(step=voxize_hmap)
    coord = grid.slice_ortho(vxcnt_hmap, roll=roll)
    grid.draw_slice(ax, coord, 1.)
    vxmap_axis = reduce_fn(vxmap, axis=(2 - roll))
    if 1 != roll:
        vxmap_axis = np.swapaxes(vxmap_axis, 0, 1)  # swap xy
    img_hit = ax.imshow(
        vxmap_axis, cmap=transparent_cmap(mpplot.cm.jet))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img_hit, cax=cax)
    ax.set_xlim([0, voxize_hmap])
    ax.set_ylim([0, voxize_hmap])
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()


def figure_vxmap(vxcnt_crop, vxmap, voxize_hmap, reduce_fn=np.sum, roll=0):
    fig, ax = tfplot.subplots(figsize=(4, 4))
    draw_vxmap(fig, ax, vxcnt_crop, vxmap, voxize_hmap, reduce_fn, roll)
    ax.axis('off')
    return fig

tfplot_vxmap = tfplot.wrap(figure_vxmap, batch=False)


def draw_vxlab(
    fig, ax, vxcnt_crop, vxlab,
        voxize_hmap, roll=0):
    vxmap = np.zeros(voxize_hmap * voxize_hmap * voxize_hmap)
    vxmap[vxlab.astype(int)] = 1
    vxmap = vxmap.reshape((voxize_hmap, voxize_hmap, voxize_hmap))
    draw_vxmap(fig, ax, vxcnt_crop, vxmap, voxize_hmap, roll=roll)
    # vxcnt_hmap = vxcnt_crop[::2, ::2, ::2]
    # grid = regu_grid(step=voxize_hmap)
    # coord = grid.slice_ortho(vxcnt_hmap, roll=roll)
    # grid.draw_slice(ax, coord, 1.)
    # vxmap_axis = np.sum(vxmap, axis=(2 - roll))
    # if 1 != roll:
    #     vxmap_axis = np.swapaxes(vxmap_axis, 0, 1)  # swap xy
    # img_hit = ax.imshow(
    #     vxmap_axis, cmap=transparent_cmap(mpplot.cm.jet))
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # fig.colorbar(img_hit, cax=cax)
    # ax.set_xlim([0, voxize_hmap])
    # ax.set_ylim([0, voxize_hmap])
    # ax.set_aspect('equal', adjustable='box')
    # ax.invert_yaxis()


def figure_vxlab(vxcnt_crop, vxlab, voxize_hmap):
    fig, ax = tfplot.subplots(figsize=(4, 4))
    draw_vxlab(fig, ax, vxcnt_crop, vxlab, voxize_hmap)
    ax.axis('off')
    return fig

tfplot_vxlab = tfplot.wrap(figure_vxlab, batch=False)


def draw_vxflt(
    fig, ax, vxcnt_crop, vxmap_flat,
        voxize_hmap, roll=0):
    vxmap = vxmap_flat.reshape((voxize_hmap, voxize_hmap, voxize_hmap))
    draw_vxmap(fig, ax, vxcnt_crop, vxmap, voxize_hmap, roll=roll)
    # vxcnt_hmap = vxcnt_crop[::2, ::2, ::2]
    # grid = regu_grid(step=voxize_hmap)
    # coord = grid.slice_ortho(vxcnt_hmap, roll=roll)
    # grid.draw_slice(ax, coord, 1.)
    # vxmap_axis = np.sum(vxmap, axis=(2 - roll))
    # if 1 != roll:
    #     vxmap_axis = np.swapaxes(vxmap_axis, 0, 1)  # swap xy
    # img_hit = ax.imshow(
    #     vxmap_axis, cmap=transparent_cmap(mpplot.cm.jet))
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # fig.colorbar(img_hit, cax=cax)
    # ax.set_xlim([0, voxize_hmap])
    # ax.set_ylim([0, voxize_hmap])
    # ax.set_aspect('equal', adjustable='box')
    # ax.invert_yaxis()


def figure_vxflt(vxcnt_crop, vxmap_flat, voxize_hmap):
    fig, ax = tfplot.subplots(figsize=(4, 4))
    draw_vxflt(fig, ax, vxcnt_crop, vxmap_flat, voxize_hmap)
    ax.axis('off')
    return fig

tfplot_vxflt = tfplot.wrap(figure_vxflt, batch=False)


def draw_hmap2(fig, ax, image_crop, hmap2):
    crop_size = image_crop.shape[0]
    hmap_size = hmap2.shape[0]
    map_scale = crop_size / hmap_size
    image_hmap = image_crop[::map_scale, ::map_scale]
    ax.imshow(image_hmap, cmap=mpplot.cm.bone_r)
    img_h2 = ax.imshow(
        hmap2,
        # cmap=transparent_cmap(mpplot.cm.jet))
        cmap=mpplot.cm.jet)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img_h2, cax=cax)


def draw_udir2(fig, ax, olmap, uomap):
    hmap_size = uomap.shape[0]
    map_scale = 2
    crop_size = hmap_size * map_scale
    xx, yy = np.meshgrid(
        np.arange(0, crop_size, map_scale),
        np.arange(0, crop_size, map_scale))
    ax.quiver(
        xx, yy,
        np.squeeze(uomap[..., 0]),
        -np.squeeze(uomap[..., 1]),
        color='r', width=0.004, scale=30)
    img_crop = np.zeros((crop_size, crop_size))
    img_crop[::map_scale, ::map_scale] = olmap
    img_h3 = ax.imshow(
        img_crop, cmap=transparent_cmap(mpplot.cm.jet))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img_h3, cax=cax)


def draw_olmap(fig, ax, image_crop, olmap):
    crop_size = image_crop.shape[0]
    hmap_size = olmap.shape[0]
    map_scale = crop_size / hmap_size
    image_hmap = image_crop[::map_scale, ::map_scale]
    ax.imshow(image_hmap, cmap=mpplot.cm.bone_r)
    img_h3 = ax.imshow(
        olmap, cmap=transparent_cmap(mpplot.cm.jet))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img_h3, cax=cax)


def draw_uomap(fig, ax, image_crop, uomap):
    crop_size = image_crop.shape[0]
    hmap_size = uomap.shape[0]
    map_scale = crop_size / hmap_size
    # assert (128 == crop_size)
    # assert (4 == map_scale)
    ax.imshow(image_crop, cmap=mpplot.cm.bone_r)
    xx, yy = np.meshgrid(
        np.arange(0, crop_size, map_scale),
        np.arange(0, crop_size, map_scale))
    ax.quiver(
        xx, yy,
        np.squeeze(uomap[..., 0]),
        -np.squeeze(uomap[..., 1]),
        color='r', width=0.004, scale=20)
    # ax.quiver(  # quiver is pointing upper-right!
    #     xx, yy,
    #     np.ones_like(xx),
    #     np.ones_like(xx),
    #     color='r', width=0.004, scale=20)


def figure_hmap2(depth_hmap, hmap2):
    fig, ax = tfplot.subplots(figsize=(4, 4))
    draw_hmap2(fig, ax, depth_hmap, hmap2)
    # ax.axis('off')
    return fig


def figure_olmap(depth_hmap, olmap):
    fig, ax = tfplot.subplots(figsize=(4, 4))
    draw_olmap(fig, ax, depth_hmap, olmap)
    # ax.axis('off')
    return fig


def figure_uomap(depth_crop, uomap):
    fig, ax = tfplot.subplots(figsize=(4, 4))
    draw_uomap(fig, ax, depth_crop, uomap)
    # ax.axis('off')
    return fig

tfplot_hmap2 = tfplot.wrap(figure_hmap2, batch=False)
tfplot_olmap = tfplot.wrap(figure_olmap, batch=False)
tfplot_uomap = tfplot.wrap(figure_uomap, batch=False)


class cam_info:
    image_size = (480, 640)
    focal = (475.065948, 475.065857)
    centre = (315.944855, 245.287079)


def fig2data(fig, show_margin=False):
    """
    @brief Convert a Matplotlib figure to a 3D numpy array with RGBA channels
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    # mpplot.gca().axis('off')
    if not show_margin:
        mpplot.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    else:
        mpplot.tight_layout()
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    ncols, nrows = fig.canvas.get_width_height()
    buf = np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def show_depth(file_name):
    """ show a depth image """
    img = mpimg.imread(file_name)
    mpplot.imshow(img, cmap=mpplot.cm.bone_r)
    mpplot.show()


def make_color_range(Color_beg, Color_end, step=10):
    return list(
        Color_beg.range_to(Color_end, step)
    )


def make_cmap_gradient(Color_beg, Color_end, step=10):
    color_range = [C.rgb for C in make_color_range(Color_beg, Color_end, step)]
    color_value = []
    for ci in range(3):
        cri = [c[ci] for c in color_range]
        color_value.append(
            zip(np.linspace(0.0, 1.0, step), cri, cri)
        )
    return dict(red=color_value[0], lime=color_value[1], blue=color_value[2])


def make_segments(x, y):
    """
    Create list of line segments in the correct format for LineCollection.
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def colorline(x, y, z=None, cmap=mpplot.get_cmap('copper'), linewidth=2):
    """
    Plot a colored line.
    """
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, linewidth=linewidth)

    ax = mpplot.gca()
    ax.add_collection(lc)

    return lc
