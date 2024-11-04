## Imports and Initialization -------------------------------------------------|

import numpy as np

# .fits file format parsing
import astropy.io.fits
from astropy.io.fits import HDUList, ImageHDU

# data visualization
import matplotlib.pyplot as plt
import matplotlib.animation
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

# plot rendering backend
matplotlib.use("TkAgg")

# internal
import flctdestretch.destretch as destretch


## Data Initialization --------------------------------------------------------|

# load our image data
data: HDUList = astropy.io.fits.open("examples/media/test.fits")
image_data_0: ImageHDU = data[0]

# test data ?? TODO explain better
test_data: np.ndarray = image_data_0.data

# process image ??
test_data /= np.mean(test_data)
test_data -= 1

# median image for y axis ??
test_median_y = np.median(test_data, axis=0)


## Visualize Image Data -------------------------------------------------------|

# create a figure and image to render our image data
figure: Figure = plt.figure(figsize=(5,4), dpi=169)
image: AxesImage = plt.imshow(test_data[0], origin="lower", cmap="copper")

# get handle to the specific subplot to render
plot_pre = plt.subplot(1, 1, 1)

# set title and limit its x and y resolution
plot_pre.set_title("Before FLCT Destretching")
plot_pre.set_xlim(0, 1000)
plot_pre.set_ylim(0, 1000)

# hide x and y axis numbers since they are pretty much meaningless for images
plot_pre.set_xticks([])
plot_pre.set_yticks([])

# define the function to animate each frame
def draw_frame_original(index: int) -> AxesImage:
	image.set_data(test_data[index])
	return image

# create an animation to view the image data in succession
animation_original = matplotlib.animation.FuncAnimation(
	figure, draw_frame_original, frames=11, interval=100
)

# show the visualization
plt.show()

## Perform Destretching -------------------------------------------------------|

# I think the sizes of each sub-image to relocate via destretch ??
kernel_sizes: np.array = np.array([32])

# ?? TODO
scene = np.moveaxis(test_data, 0, -1)

# this goes through each image in the data and applies the destretching 
# algorithm and stores the result
result = destretch.reg_loop_series(
	scene, 
	test_median_y, 
	kernel_sizes, 
	mf=0.08, 
	use_fft=True 
)

# deconstruct the results into meaningful typed variables
answer: np.ndarray[np.float64]
display: np.ndarray[np.float64]
r_display: np.ndarray[np.float64]
destretch_info: destretch.Destretch_params
(answer, display, r_display, destretch_info) = result

## Visualize new Data ---------------------------------------------------------|

# create the figure and image for rendering the new image data
figure: Figure = plt.figure(figsize=(5,4), dpi=169)
image: AxesImage = plt.imshow(answer[:, :, 0], origin="lower", cmap="copper")

# arrange sublot and get handle
plot_post = plt.subplot(1, 1, 1)
plot_post.set_title("After FLCT Destretching")
plot_post.set_xlim(0, 1000)
plot_post.set_ylim(0, 1000)

def draw_frame_destretched(index: int) -> AxesImage:
	image.set_data(answer[:, :, index])
	return image


# create a new animationto display the image data resulting from 
# the destretching
animation_destretched = matplotlib.animation.FuncAnimation(
	figure, draw_frame_destretched, frames=11, interval=100
)

# display the visualization
plt.show()