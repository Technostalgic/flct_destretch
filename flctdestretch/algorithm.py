"""
Primary algorithm module to perform flct destretching image processing based 
on implementation by Momchil Molnar
"""

## Imports and Initialization --------------------------------------------------

import time
import numpy as np
from scipy import fft
from scipy.signal.windows import blackman
from scipy.interpolate import RectBivariateSpline

# internal
from destretch_types import DestretchParams, DestretchLoopResult

## Processing: -----------------------------------------------------------------

def bilin_values_scene(scene, coords_new, destr_info, nearest_neighbor=False) -> np.ndarray:
	"""
	Bilinear interpolation (resampling)
	of the scene s at coordinates xy

	Parameters
	----------
	scene : ndarray (nx, ny)
		Scene
	coords_new : ndarray (2, nx, ny)
		coordinates of the pixels of the output image
		on the input image (at which to interpolate the scene)
	destr_info: class Destretch_Params
		Destretch parameters

	Returns
	-------
	ans: ndarray (nx, ny)
		Bilinear interpolated (resampled) image at the xy locations

	"""

	if nearest_neighbor == True:
		x = np.array(np.round(coords_new[0, :, :]), order="F", dtype=int)
		y = np.array(np.round(coords_new[1, :, :]), order="F", dtype=int)
		
		print(scene.shape,x.shape,y.shape)
		scene_interp = scene[np.clip(x,0,x.shape[0]-1), 
							 np.clip(y,0,y.shape[1]-1)]

	else:
		x = np.array(coords_new[0, :, :], order="F")
		y = np.array(coords_new[1, :, :], order="F")

		# need to limit output coordinates so that interpolation calculations
		# don't go out of bounds (i.e. we add 1 to x and y coordinates below)
		x = np.clip(x, 0, x.shape[0]-2)
		y = np.clip(y, 0, y.shape[1]-2)

		x0 = x.astype(int)
		x1 = (x+1).astype(int)
		y0 = (y).astype(int)
		y1 = (y+1).astype(int)

		fx = x % 1.
		fy = y % 1.

		#scene  = np.array(selector_events, order="F").astype(np.float32)
		#scene_float = scene.astype(np.float32)
		scene_float = scene
		#print(scene_float.shape, scene_float.dtype)
		#scene_float = scene.copy

		ss00 = scene_float[x0, y0]
		ss01 = scene_float[x0, y1]
		ssfx00 =                (scene_float[x1, y0] - ss00) * fx
		ssfy01 = (ss01 - ss00 + (scene_float[x1, y1] - ss01) * fx - ssfx00) * fy
		scene_interp  = ss00 + ssfx00 + ssfy01

	return scene_interp

def apod_mask(nx, ny, fraction=0.08):
	"""
	Create an apodization mask over the apertures
	to reduce FFT edge effects.

	Parameters
	----------
	nx : int
		Width of window in pixels
	ny : int
		Height of window in pixels
	fraction: float
		Fraction of window over which intensity drops to zero

	Returns
	-------
	Apodization window (NumPy array)
	"""

	taper_wx = int(nx * min(fraction, 0.5))
	taper_wy = int(ny * min(fraction, 0.5))

	filt_x = blackman(2 * taper_wx)
	filt_y = blackman(2 * taper_wy)

	left = filt_x[:taper_wx]
	right = left[::-1]
	top = filt_y[:taper_wy]
	bottom = top[::-1]
	center_x = np.ones(nx - 2*taper_wx)
	center_y = np.ones(ny - 2*taper_wy)

	x_arr = np.concatenate((left, center_x, right))
	y_arr = np.concatenate((top, center_y, bottom))

	m = np.array(np.outer(x_arr, y_arr), order='F')

	return m

def smouth(nx, ny):
	"""
	Smouthing window to be applied to the 2D FFTs to
	remove HF noise.

	WORKS! Checked against IDl

	Parameters
	----------
	nx : integer
		Window size in x-direction.
	ny : integer
		Window size in y-direction.

	Returns
	-------
	mm : ndarry [nx, ny]
		smoothing mask.

	"""

	x = np.arange(nx//2)
	if nx % 2 == 1:
		x = np.concatenate([x, x[nx//2-1:nx//2], np.flip(x)])
	else:
		x = np.array([x, np.flip(x)],).flatten()
	if nx > 60:
		magic_number = nx//6
	else:
		magic_number = 10
	x = np.exp(-1*(x/(magic_number))**2)

	y = np.arange(ny//2)
	if (ny % 2) == 1:
		y = np.concatenate([y, y[ny//2-1:ny//2], np.flip(y)])
	else:
		y = np.array([y, np.flip(y)]).flatten()
	if ny > 60:
		magic_number = ny//6
	else:
		magic_number = 10
	y = np.exp(-1*(y/(magic_number))**2)

	mm = np.outer(x.T, y)

	return mm

def correlation_maxpos_vectorized(
	correlations: np.ndarray, 
	max_fit_method: int = 1
) -> tuple[np.ndarray, np.ndarray]:
	"""
	Find coordinate of the peak for each correlation, with subpixel interpolation
	"""

	kernel_width, kernel_height = correlations.shape[1:]
	correlation_count: int = correlations.shape[0]
	flat_correlations = correlations.reshape(correlation_count, -1)
	peak_value = np.amax(flat_correlations, axis=1)
	max_indices: np.ndarray = np.argmax(flat_correlations, axis=1)
	ymax_coord = max_indices % kernel_width
	xmax_coord = max_indices // kernel_width
	indices = np.arange(correlation_count)

	match max_fit_method:

		# simple interpolation
		case 1:
			xdenominators = (
				peak_value * 2 - 
				correlations[indices, np.clip(xmax_coord - 1, 0, kernel_width - 1), ymax_coord] - 
				correlations[indices, np.clip(xmax_coord + 1, 0, kernel_width - 1), ymax_coord]
			)
			xratios = (
				(xmax_coord - 0.5) + 
				(peak_value - correlations[indices, xmax_coord-1, ymax_coord]) / 
				xdenominators
			)

			ydenominators = (
				peak_value * 2 - 
				correlations[indices, xmax_coord, np.clip(ymax_coord - 1, 0, kernel_height - 1)] - 
				correlations[indices, xmax_coord, np.clip(ymax_coord + 1, 0, kernel_height - 1)]
			)
			yratios = (
				(ymax_coord - 0.5) + 
				(peak_value - correlations[indices, xmax_coord, ymax_coord-1]) / 
				ydenominators
			)
			return xratios, yratios
		
		# a more complicated interpolation
		# (from Niblack, W: An Introduction to Digital Image Processing, p 139.)
		case 2:
			# TODO vectorize:
			# a2 = (cc[xmax+1, ymax] - cc[xmax-1, ymax])/2.
			# a3 = (cc[xmax+1, ymax]/2. - cc[xmax, ymax] + cc[xmax-1, ymax]/2.)
			# a4 = (cc[xmax, ymax+1] - cc[xmax, ymax-1])/2.
			# a5 = (cc[xmax, ymax+1]/2. - cc[xmax, ymax] + cc[xmax, ymax-1]/2.)
			# a6 = (cc[xmax+1, ymax+1] - cc[xmax+1, ymax-1] 
			# 	- cc[xmax-1, ymax+1] + cc[xmax-1, ymax-1])/4.
			# xdif = (2*a2*a5 - a4*a6) / (a6**2 - 4*a3*a5)
			# ydif = (2*a3*a4 - a2*a6) / (a6**2 - 4*a3*a5)
			# xmax = xmax + xdif
			# ymax = ymax + ydif
			raise NotImplementedError
		
		case _:
			raise NotImplementedError

def surface_fit_vectorized(subwindows: np.ndarray, order: int = 0) -> np.ndarray:
	"""
	fit a polynomial surface against each subwindow in an array of 2D arrays

	WARNING: fit happens in-place, so parameters are modified
	"""
	# TODO implement order 1
	match(order):
		
		# order 0 - flat mean fit
		case 0:
			subwindows -= subwindows.mean(axis=(1, 2), keepdims=True)

		# order 1 - plane surface fit
		case 1:
			# TODO vectorize
			# # Analytical solution for plane using linear algebra
			# # grid points in X,Y
			# L, M = subwindows.shape
			# X1, X2 = np.mgrid[:L, :M]
			# # reshape independent variables into form [a, b*X1, c*X2]
			# X = np.hstack((np.ones((L*M, 1)), X1.reshape((L*M, 1)), X2.reshape((L*M, 1))))
			# # reshape dependent variable into column vector
			# YY = subwindows.reshape((L*M, 1))
			# # calculate normal vector of plane: theta = [X.T X]^-1 X.T YY
			# theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), YY)
			# # obtain fitted plane: plane_points = X ⋅ theta
			# surface_array = np.dot(X, theta).reshape((L, M))
			# return surface_array
			raise NotImplementedError()

		# higher orders are not feasible to vectorize
		case _: raise NotImplementedError()
	
	return subwindows

## Destretching ----------------------------------------------------------------

def bilin_control_points(scene, rdisp, disp):
	"""
	Compute the coordinates of the pixels in the output images to be
	sampled from the input image (using Scipy.interpolate.RectBivariate).
	Interpolate the control point displacements to infer the sampling
	coordinates.

	Parameters
	----------
	scene : ndarray (nx, ny)
		Image input
	rdisp : ndarray (kx, ky, 2)
		Reference coordinates of the control points.
	disp : ndarray (kx, ky, 2)
		Actual coordinates of the control points.

	Returns
	-------
	xy_grid : ndarray (2, nx, ny)
		Coordinates of the input image to be sampled for the output image
	"""

	scene_nx = scene.shape[0]
	scene_ny = scene.shape[1]

	# compute the control points locations
	#     this assumes the x coordinates are same for all rows 
	#     and the y-coordinates are the same for all columns
	
	# 1-D array of x-values of reference points
	cp_x_coords = rdisp[0, :, 0]
	
	# 1-D array of y-values of reference points
	cp_y_coords = rdisp[1, 0, :]

	# define an array for x and y displacement coordinates, the same size 
	# as the input scent
	xy_ref_coordinates = np.zeros((2, scene_nx, scene_ny), order="F")

	# this creates an array where values in array index 1 are constant,
	# with a value corresponding to the values of array index 12 
	xy_ref_coordinates[1, :, :] = [
		np.linspace(0, (scene_ny-1) , num=scene_ny, dtype="int")
		for el in range(scene_nx)
	]
	# this creates an array where values in array index 2 are constant,
	# with a value corresponding to the values of array index 1 
	xy_ref_coordinates[0, :, :] = [
		np.zeros(scene_ny, dtype="int") + el 
		for el in range(scene_nx)
	]

	# flip the axes
	# xy_ref_coordinates = np.swapaxes(xy_ref_coordinates, 1, 2)

	# calculate offsets between displaced and reference positions
	dd = disp - rdisp

	interp_x = RectBivariateSpline(cp_x_coords, cp_y_coords, dd[0, :, :], kx=3, ky=3, s=0)
	interp_y = RectBivariateSpline(cp_x_coords, cp_y_coords, dd[1, :, :], kx=3, ky=3, s=0)
	#interp_x = SmoothBivariateSpline((rdisp[0, :, :]).flatten(), (rdisp[1, :, :]).flatten(), (dd[0, :, :]).flatten())
	#interp_y = SmoothBivariateSpline((rdisp[0, :, :]).flatten(), (rdisp[1, :, :]).flatten(), (dd[1, :, :]).flatten())

	xy_grid = np.zeros((2, scene_nx, scene_ny))

	x_coords_output = np.linspace(0, scene_nx-1, num=scene_nx)
	y_coords_output = np.linspace(0, scene_ny-1, num=scene_ny)

	xy_grid[0, :, :] = 1. * interp_x.__call__(
		x_coords_output, y_coords_output,
		grid=True
	)
	xy_grid[1, :, :] = 1. * interp_y.__call__(
		x_coords_output, y_coords_output,
		grid=True
	)

	# if test == True:
	#    im1 = pl.imshow(xy_grid[0, :, :])
	#    pl.colorbar(im1)
	#    pl.show()
	#
	#    im2 = pl.imshow(xy_grid[1, :, :])
	#    pl.colorbar(im2)
	#    pl.show()

	xy_grid_coords = xy_grid + xy_ref_coordinates

	return xy_grid_coords, xy_grid

def destr_control_points(
	reference: np.ndarray, 
	kernel: np.ndarray, 
	border_offset: int, 
	spacing_ratio: float, 
	mf: float = 0.08
):
	"""
	this function defines a regularly spaced grid on control points, which are
	the central pixel positions of each subfield for the destretching local 
	offset determination.
	   
	Seems to work  
	Choose control point locations in the reference

	Parameters
	----------
	reference : TYPE
		Reference scene - passed only to define size of full reference image
	kernel : TYPE
		Kernel props

	Returns
	-------
	destr_info: Destr class
		Destructor info

	rcps : TYPE
		DESCRIPTION.
	"""
	define_cntl_pts_orig = 0
	destr_info = DestretchParams()

	# determine the number of pixels in the kernel
	ksz = kernel.shape
	destr_info.kx = ksz[0]
	destr_info.ky = ksz[1]

	# determine the number of pixels in the reference image
	# the assumption is that the reference is a 2D array, so we only need the 
	# x- and y-dimensions
	rsz = reference.shape
	destr_info.ref_sz_x  = rsz[0]
	destr_info.ref_sz_y  = rsz[1]

	# define size of subfield to use
	destr_info.wx = int(destr_info.kx * 2)
	destr_info.wy = int(destr_info.ky * 2)
	if (destr_info.wx % 2):
		destr_info.wx = int(destr_info.wx + 1)
	if (destr_info.wy % 2):
		destr_info.wy = int(destr_info.wy + 1)

	# [wx,wy] define the size of a border around the edge of the image, to add 
	#   an additional buffer area in which to avoid placing the control points.
	# The border_offset input variable defines this border area in relation to 
	#   the kernel size, but maybe it's better to define it as an absolute 
	#   number of pixels?
	destr_info.border_x = int(border_offset)
	destr_info.border_y = int(border_offset)
	# make sure [border_x,border_y] is divisible by 2
	if (destr_info.border_x % 2):
		destr_info.border_x = int(destr_info.border_x + 1)
	if (destr_info.border_y % 2):
		destr_info.border_y = int(destr_info.border_y + 1)
	destr_info.border_x  += destr_info.border_x % 1
			
	if destr_info.debug >= 2: 
		print('Border Size = ',destr_info.border_x, ' x ', destr_info.border_y)
	if destr_info.debug >= 2: 
		print('Kernel Size = ',destr_info.kx, ' x ', destr_info.ky)
	
	if define_cntl_pts_orig:
		cpx = int(
			(destr_info.ref_sz_x - destr_info.wx + destr_info.kx) //
			destr_info.kx
		)
		cpy = int(
			(destr_info.ref_sz_y - destr_info.wy + destr_info.ky) //
			destr_info.ky
		)
		# old way of defining the control points by looping through x and way and 
		# adding a fixed offset to the previously defined control point

		destr_info.bx = int(((destr_info.ref_sz_x - destr_info.wx + destr_info.kx) % destr_info.kx)/2)
		destr_info.by = int(((destr_info.ref_sz_y - destr_info.wy + destr_info.ky) % destr_info.ky)/2)
		rcps = np.zeros((2, cpx, cpy), order="F")

		ly = destr_info.by
		hy = ly + destr_info.wy
		for j in range(0, cpy):
			lx = destr_info.bx
			hx = lx + destr_info.wx
			for i in range(0, cpx):
				rcps[0, i, j] = (lx + hx)/2
				rcps[1, i, j] = (ly + hy)/2
				lx = lx + destr_info.kx
				hx = hx + destr_info.kx

			ly = ly + destr_info.ky
			hy = hy + destr_info.ky
	else:
		# the control points must start and end at least 1/2 kernel width away from the edges of the array
		# So that means that the allowable range of pixels available for control points 
		#     is reduced by (at minimum) one kernel width
		# it is also reduced by the size of the extra border on each side
		allowable_range_x = destr_info.ref_sz_x - destr_info.kx - (destr_info.border_x * 2)
		allowable_range_y = destr_info.ref_sz_y - destr_info.ky - (destr_info.border_y * 2)
		
		# how far apart should the sub-array control points be placed, in units of the kernel width
		# set the spacing between subarrays, making sure it is divisible by 2 (just because...)
		destr_info.spacing_x  = int(destr_info.kx * spacing_ratio)
		destr_info.spacing_y  = int(destr_info.ky * spacing_ratio)
		destr_info.spacing_x += destr_info.spacing_x % 2
		destr_info.spacing_y += destr_info.spacing_y % 2
		if destr_info.spacing_x <= 0:
			destr_info.spacing_x = 1
		if destr_info.spacing_y <= 0:
			destr_info.spacing_y = 1

		# divide the number of allowable pixels by the control points, round down to nearest integer
		num_grid_x        = int(allowable_range_x / destr_info.spacing_x) + 1
		num_grid_y        = int(allowable_range_y / destr_info.spacing_y) + 1
		destr_info.cpx    = num_grid_x
		destr_info.cpy    = num_grid_y
		
		# how far apart will the first and last control points be, in each axis
		total_range_x     = destr_info.spacing_x * (num_grid_x - 1)
		total_range_y     = destr_info.spacing_y * (num_grid_y - 1)
		# the total range will be less than the maximum possible range, in most cases
		# so allocate some of those extra pixels to each border
		destr_info.bx     = np.round((allowable_range_x - total_range_x + destr_info.kx)/2.)
		destr_info.by     = np.round((allowable_range_y - total_range_y + destr_info.ky)/2.)
		
		destr_info.mf = mf

		if destr_info.debug >= 2: print('Number of Control Points = ',num_grid_x, ' x ', num_grid_y)
		if destr_info.debug >= 2: print('Number of Border Pixels = ',destr_info.bx, ' x ', destr_info.by)
		if destr_info.debug >= 3: print('allowable range,grid spacing x, num grid x, total range x, start pos x',
										  allowable_range_x,destr_info.spacing_x,num_grid_x,total_range_x,destr_info.bx)

		rcps              = np.zeros([2, destr_info.cpx, destr_info.cpy])
		rcps[0,:,:]       = np.transpose(np.tile(np.arange(destr_info.cpx) * destr_info.spacing_x + destr_info.bx, (destr_info.cpy, 1)))
		rcps[1,:,:]       =              np.tile(np.arange(destr_info.cpy) * destr_info.spacing_y + destr_info.by, (destr_info.cpx, 1))
		destr_info.rcps = rcps
										   

	return destr_info, rcps

def controlpoint_offsets_fft(
	scene: np.ndarray, 
	subfield_fftconj: np.ndarray, 
	apod_window: np.ndarray, 
	lowpass_filter: np.ndarray, 
	destr_info: DestretchParams
) -> np.ndarray:
	"""
	Calculate the offsets of the control points in the reference frame, which 
	can be determined from the subfield fft cojugates passed in here

	Parameters
	----------
	scene : array
		a 2-dimensional array (L x M) containing the image to be registered
	subfield_fftconj : array
		the array of FFTs of all the image subfields, as cutout from the reference array
	apod_mask : array
		apodization mask, darkens edges of images to reduce FFT artifacts
	lowpass_filter : array
		reduces high-frequency noise in FFT
	destr_info : structure
		Destretch information

	Returns
	-------
	subfield_offsets : array
		X and Y offsets for control points
	"""

	kernel_width, kernel_height = destr_info.kx, destr_info.ky

	# flatten reference control point coordinates into 1d
	control_points_x: np.ndarray = destr_info.rcps[0].ravel()
	control_points_y: np.ndarray = destr_info.rcps[1].ravel()

	# find top left (subwindow start coordinate) of each subwindow for each control point
	topleft_x = (control_points_x - kernel_width // 2).astype(int)
	topleft_y = (control_points_y - kernel_height // 2).astype(int)

	# create an array to hold each subwindow
	xgrid = np.arange(kernel_width).reshape((kernel_width, 1))
	ygrid = np.arange(kernel_height).reshape((1, kernel_height))
	subwindows = scene[
		topleft_x[:, np.newaxis, np.newaxis] + xgrid[np.newaxis, :, :],
		topleft_y[:, np.newaxis, np.newaxis] + ygrid[np.newaxis, :, :]
	].copy()
	subwindow_count: int = subwindows.shape[0]

	# apply surface fit
	# TODO implement order 1
	subwindows = surface_fit_vectorized(subwindows, 0)

	# apply apodization mask
	subwindows *= apod_window[np.newaxis, :, :]

	# reshape reference image fft conjugates to match subwindows arrary
	ref_fft = (
		subfield_fftconj
			.reshape(kernel_width, kernel_height, subwindow_count)
			.transpose(2,0,1)
	)

	# apply fft to subwindows
	# TODO parallelize with scipi?
	ffts = fft.fft2(subwindows, axes=(1, 2), workers=-1) 
	ffts *= ref_fft
	ffts *= lowpass_filter[np.newaxis, :, :]

	# find cross correlation from inverse fft
	# TODO parallelize with scipi?
	correlations = np.abs(fft.ifft2(ffts, axes=(1,2), workers=-1))
	correlations = np.roll(correlations, (kernel_width // 2, kernel_height // 2), axis=(1, 2))

	# find peak for each correlation, with subpixel interpolation
	xmax, ymax = correlation_maxpos_vectorized(correlations, destr_info.max_fit_method)

	# store peak coordinates
	offsets = np.zeros((2, destr_info.cpx, destr_info.cpy), dtype=np.float32)
	offsets[0].ravel()[:] = topleft_x + xmax
	offsets[1].ravel()[:] = topleft_y + ymax

	return offsets

def reg_loop(
		scene, ref, kernel_sizes, 
		mf=0.08, use_fft=True, adf2_pad=0.25, adf_pow=2, border_offset=4, 
		spacing_ratio=0.5
	) -> DestretchLoopResult:
	"""
	Parameters
	----------
	scene : ndarray (nx, ny)
		Image to be destretched
	ref : ndarray (nx, ny)
		Reference image
	kernel_sizes : ndarray (n_kernels)
		Sizes of the consecutive kernels to be applied

	Returns
	-------
	ans : ndarray (nx, ny)
		Destretched scene
	destr_info: Destretch class
		Parameters of the destretching
	"""

	scene_nx = scene.shape[0]
	scene_ny = scene.shape[1]

	scene_temp = scene.copy()
	# start = time.time()
	# print("Spacing Ratio: ", spacing_ratio)

	disp_sum     = np.zeros((2, scene_nx, scene_ny))
	offsets_sum  = np.zeros((2, scene_nx, scene_ny))
	rdisp_sum    = np.zeros((2, scene_nx, scene_ny))
	kernel_count = 0.0

	destr_info: DestretchParams | None = None
	for kernel_dim in kernel_sizes:
		scene_temp, disp, rdisp, destr_info = reg(scene_temp, ref, kernel_dim, mf, use_fft, adf2_pad, adf_pow, border_offset, spacing_ratio)
		# remap displacements onto spatial grid of scene 
		# (i.e. the same number of pixels as the input image)
		dispmap_new, offsets_new  = bilin_control_points(scene, rdisp, disp)
		# add the displacement and offset maps to
		disp_sum     += dispmap_new
		offsets_sum  += offsets_new
		rdisp_sum    += dispmap_new - offsets_new
		kernel_count += 1
	assert destr_info is not None

	# destr_info.kx , ky - kernel size
	# use this, alongside spacing_ratio and border_size to reduce the 
	# resolution of the displacement maps

	# the displacement maps contain the pixel reference coordinates, so 
	# adding them iteratively sums those reference coordinates
	# divide by the number of maps summed to get back to the rate coordinates
	disp_sum /= kernel_count
	rdisp_sum /= kernel_count

	# end = time.time()
	# print(f"Total elapsed time {(end - start):.4f} seconds.")
	ans = scene_temp

	return DestretchLoopResult(ans, disp_sum, rdisp_sum, destr_info)

def doreg(scene, r, d, destr_info) -> np.ndarray:
	"""
	Parameters
	----------
	scene : TYPE
		Scene to be destretched
	r : TYPE
		reference displacements of the control points
	d : TYPE
		Actual displacements of the control points
	destr_info: Destr class
		Destretch information

	Returns
	-------
	ans : TYPE
		Destretched scene.

	"""

	xy, xy_offsets  = bilin_control_points(scene, r, d)
	# this was some old code for juggling the axes to match the inputs for bilin_values_scene
	# sorted out the axes in the other procedures so this should no longer be necessary
	#xy = xy[[1,0],:,:]
	#xy = np.swapaxes(xy, 1, 2)
	#scene = np.swapaxes(copy.deepcopy(scene), 0, 1)

	ans = bilin_values_scene(scene, xy, destr_info, nearest_neighbor=False)

	return ans

def reg(
	scene, ref, kernel_size, mf=0.08, 
	use_fft=True, adf_pad=0.25, adf_pow=2, 
	border_offset=4, spacing_ratio=0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray, DestretchParams]:
	# TODO: clean up control point offset calculations - move FFT specific 
	# calls (e.g. apod) into conditional
	# TODO: (here and elsewhere) rename d_info to destr_info
	# TODO: add crosscorrelation choice, other parameters to destr_info; 
	# rename destr_info.mf
	# TODO: make destr_info an optional input
	# TODO: change spacing_ratio to controlpoint_spacing (in pixels)
	# TODO: add documentation to functions, etc.!
	# TODO: process flowchart
	# TODO: testing framework - pytest?
	"""
	Register scenes with respect to ref using kernel size and
	then returns the destretched scene.

	Parameters
	----------
	scene : [nx, ny] [nx, ny, nf]
		Scene to be registered
	ref : [nx, ny]
		reference frame
	kernel_size : int
	   Kernel size (otherwise unused)!!!!!

	Returns
	-------
	ans : [nx, ny]
		Destreched scene.
	disp : ndarray (kx, ky)
		Control point locations
	rdisp : ndarray (kx, ky)
		Reference control point locations

	"""
	do_timing = False

	scene -= scene.mean()
	ref -= ref.mean()
	kernel = np.zeros((kernel_size, kernel_size))

	# compute control point locations
	destr_info, rdisp = destr_control_points(ref, kernel, border_offset, spacing_ratio, mf)
	#destr_info.subfield_correction = 0
	destr_info.subfield_correction = 1
	
	destr_info.use_fft = use_fft
	
	apod_window = apod_mask(destr_info.kx, destr_info.ky, destr_info.mf)
	smou = smouth(destr_info.kx, destr_info.ky)
	#Condition the ref

	ssz = scene.shape
	ans = np.zeros((ssz[0], ssz[1]), order="F")

	start = time.time()

	subfield_fftconj = doref(ref, apod_window, destr_info)
	disp = controlpoint_offsets_fft(scene, subfield_fftconj, apod_window, smou, destr_info)
	
	if do_timing:
		dtime = time.time() - start
		print(f"Time for a scene destretch is {dtime:.3f}")

	#disp = repair(rdisp, disp, d_info) # optional repair
	#rms = sqrt(total((rdisp - disp)^2)/n_elements(rdisp))
	#print, 'rms =', rms
	#mdisp = np.mean(rdisp-disp,axis=(1, 2))
	#disp[0, :, :] += mdisp[0]
	#disp[1, :, :] += mdisp[1]
	x = doreg(scene, rdisp, disp, destr_info)
	ans = x

	# print(f"Total destr took: {(end - start):.5f} seconds for kernel"
 	# 	+f"of size {kernel_size} px.")

	return ans, disp, rdisp, destr_info

def doref(
	ref_image: np.ndarray, 
	apod_mask: np.ndarray, 
	destr_info: DestretchParams
) -> np.ndarray:
	"""
	Setup reference window

	Parameters
	----------
	ref : a 2-dimensional array (L x M) containing the reference image
			against which the scene should be registered
	apod_mask : apodization mask to be applied to subfield image 
		mask
	destr_info : TYPE
		Destretch_info.

	Returns
	-------
	subfields_fftconj: array (kx, ky, cp_numx, cp_numy)
		Reorganized window
	"""
	k_width, k_height = destr_info.kx, destr_info.ky
	cp_x, cp_y = destr_info.cpx, destr_info.cpy
	ref_cps: np.ndarray = destr_info.rcps

	# create an array to hold a subwindow for each kernel
	topleft_x = (ref_cps[0].ravel() - k_width * 0.5).astype(int)
	topleft_y = (ref_cps[1].ravel() - k_height * 0.5).astype(int)
	xgrid = np.arange(k_width).reshape((k_width, 1))
	ygrid = np.arange(k_height).reshape((1, k_height))
	subwindows = ref_image[
		topleft_x[:, np.newaxis, np.newaxis] + xgrid[np.newaxis, :, :],
		topleft_y[:, np.newaxis, np.newaxis] + ygrid[np.newaxis, :, :]
	].copy()

	# apply suface fit and then apod mask
	# TODO implement order 1
	subwindows = surface_fit_vectorized(subwindows, 0)
	subwindows *= apod_mask[np.newaxis, :, :]
	
	# calculate and store the fft conjugate for each subwindow
	subfields_fftconj = (
		np.array(
			np.conj(fft.fft2(subwindows, axes=(1,2), workers=-1)),
			order="F"
		)
		.reshape(cp_x, cp_y, k_width, k_height)
		.transpose(2,3,0,1)
	)

	return subfields_fftconj
