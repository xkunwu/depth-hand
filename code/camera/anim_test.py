""" Hand in Depth
    https://github.com/xkunwu/depth-hand
"""
import numpy as np
import matplotlib.pyplot as mpplot
from matplotlib.animation import FuncAnimation
import pyrealsense2 as pyrs


# Create a pipeline
pipeline = pyrs.pipeline()

# Create a config and configure the stream
config = pyrs.config()
config.enable_stream(pyrs.stream.depth, 640, 480, pyrs.format.z16, 30)
config.enable_stream(pyrs.stream.color, 640, 480, pyrs.format.rgb8, 30)
#config.enable_stream(pyrs.stream.color, 640, 480, pyrs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

# clip the background
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
align_to = pyrs.stream.color
align = pyrs.align(align_to)

# Create the figure canvas
fig, _ = mpplot.subplots(nrows=1, ncols=2, figsize=(2 * 5, 1 * 5))
ax1 = mpplot.subplot(1, 2, 1)
ax1.set_axis_off()
ax2 = mpplot.subplot(1, 2, 2)
ax2.set_axis_off()
mpplot.subplots_adjust(left=0, right=1, top=1, bottom=0)
im1 = ax1.imshow(np.zeros([480, 640], dtype=np.uint16), vmin=0., vmax=1., cmap=mpplot.cm.bone_r)
im2 = ax2.imshow(np.zeros([480, 640, 3], dtype=np.uint8))

def update(i):
    frames = pipeline.wait_for_frames()

    # Get frameset of only depth
    #depth = frames.get_depth_frame()
    #if not depth:
    #    continue
    #depth_image = np.asanyarray(depth)

    # Get aligned color and depth frameset
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not aligned_depth_frame or not color_frame:
        return
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    #color_image = np.asanyarray(color_frame.get_data())[..., ::-1]

    # Remove background - Set to grey
    grey_color = 159
    depth_image_3d = np.dstack(
            (depth_image, depth_image, depth_image))
    bg_removed = np.where(
            (depth_image_3d > clipping_distance) | (depth_image_3d <= 0),
            grey_color, color_image)
    np.clip(depth_image, 0, clipping_distance, out=depth_image )

    # Rendering
    im1.set_data(depth_image.astype(float) / clipping_distance)
    im2.set_data(bg_removed)
    #im2.set_data(color_image)
    
try:
    ani = FuncAnimation(fig, update, blit=False, interval=1)
    def close(event):
        if event.key == 'q':
            mpplot.close(event.canvas.figure)
    cid = fig.canvas.mpl_connect("key_press_event", close)
    mpplot.show()
finally:
    mpplot.close(fig)
    pipeline.stop()
