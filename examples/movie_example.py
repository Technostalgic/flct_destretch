## Imports and Initialization -------------------------------------------------|

import numpy as np

# .fits file format parsing
import astropy.io.fits
from astropy.io.fits import HDUList, ImageHDU

# data visualization
import matplotlib.pyplot as plt
import matplotlib.animation

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
figure = plt.figure(figsize=(5,4), dpi=169)
image = plt.imshow(test_data[0], origin="lower", cmap="copper")

# get handle to the specific subplot to render, and limit its x and y resolution
subplot = plt.subplot(1, 1, 1)
subplot.set_xlim(0, 1000)
subplot.set_ylim(0, 1000)

# get handle to subplot title so it can be changed
subplot_title = subplot.set_title("Before FLCT Destretching")

# define the function to animate each frame
def draw_frame(index: int):
	image.set_data(test_data[index])
	return image

# create an animation to view the image data in succession
animation = matplotlib.animation.FuncAnimation(
	figure, draw_frame, frames=11, interval=100
)

# hide x and y axis numbers since they are pretty much meaningless for images
plt.xticks([])
plt.yticks([])

# show the visualization
plt.show()
