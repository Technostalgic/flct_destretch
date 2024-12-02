import numpy as np
import astropy.io.fits as fits
from astropy.io.fits.hdu import HDUList, ImageHDU, CompImageHDU

from algorithm import reg_loop_series
from destretch_params import DestretchParams


def destretch_files(filepaths: list[str], kernel_sizes: list[int]) -> tuple[
	np.ndarray[np.float64],
	np.ndarray[np.float64],
	np.ndarray[np.float64],
	DestretchParams
]:
	"""
	Compute the destretched result of data from all given files

	Parameters
	----------
    filepaths : list[str]
        list of files in order to treat as image sequence for destretching
    kernel_sizes : ndarray (kx, ky, 2)
        the destretch kernel size, the size of each subframe to be processed

    Returns
    -------
    result : tuple[np.ndarray, np.ndarray, np.ndarray, DestretchParams]
		same output as reg_loop_series
	"""

	# used to enforce the same resolution for all data files
	image_resolution = [-1,-1]

	# stores the image hdu data from all files
	data_sequence: list[np.ndarray] = []

	# read each image HDU in each file
	print("Searching for image data in specified files...")
	for path in filepaths:
		hdus: HDUList = fits.open(path)
		for hdu_index in range(len(hdus)):
			hdu = hdus[hdu_index]
			if hdu.data is not None or hdu is ImageHDU or hdu is CompImageHDU:

				# only work with 2d images for now
				image_data: np.ndarray = hdu.data
				if len(image_data.shape) == 3:

					# ensure all data matches the first file's resolution
					if image_resolution[0] < 0:
						image_resolution[0] = image_data.shape[2]
						image_resolution[1] = image_data.shape[1]
					elif (
						image_resolution[0] != image_data.shape[2] or 
						image_resolution[1] != image_data.shape[1]
					): continue

					print(f"found {hdu_index} in {path}: {image_data.shape}")
					data_sequence.append(image_data)
	print(f"found {len(data_sequence)} image data units")

	data_cube: np.ndarray = np.zeros((
		image_resolution[0], 
		image_resolution[1], 
		len(data_sequence)
	))
	for i in range(len(data_sequence)):
		data_cube[:, :, i] = data_sequence[i]

	# median of each pixel over the image sequence
	test_median_image = np.median(data_cube, axis=2)

	print("applying destretching now...")
	return reg_loop_series(
		data_cube, 
		test_median_image, 
		kernel_sizes,
		mf=0.08,
		use_fft=True
	)
