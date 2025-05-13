"""
Primary algorithm module to perform flct destretching image processing based 
on implementation by Momchil Molnar
"""


## Imports and Initialization -------------------------------------------------|

import time
import numpy as np
from scipy import signal as signal
from scipy.interpolate import RectBivariateSpline
from typing import Tuple, Literal, Any

# internal
from .destretch_types import DestretchParams, DestretchLoopResult


## Processing: ----------------------------------------------------------------|

def bilin_values_scene(scene, coords_new, destr_info, nearest_neighbor=False):
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

    filt_x = signal.windows.blackman(2 * taper_wx)
    filt_y = signal.windows.blackman(2 * taper_wy)

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

def crosscor_maxpos(cc, max_fit_method=1):
    """
    TODO docstring
    """
    mx  = np.amax(cc)
    loc = cc.argmax()

    ccsz = cc.shape
    ymax = loc % ccsz[0]
    xmax = loc // ccsz[0]

    #a more complicated interpolation
    #(from Niblack, W: An Introduction to Digital Image Processing, p 139.)

    if xmax*ymax > 0 and xmax < (ccsz[0]-1) and ymax < (ccsz[1]-1):
        if max_fit_method == 1:
            denom = 2 * mx - cc[xmax-1,ymax] - cc[xmax+1,ymax]
            xfra = (xmax-1/2) + (mx-cc[xmax-1,ymax])/denom

            denom = 2 * mx - cc[xmax,ymax-1] - cc[xmax,ymax+1]
            yfra = (ymax-1/2) + (mx-cc[xmax,ymax-1])/denom

            xmax=xfra
            ymax=yfra
        elif max_fit_method == 2:
            a2 = (cc[xmax+1, ymax] - cc[xmax-1, ymax])/2.
            a3 = (cc[xmax+1, ymax]/2. - cc[xmax, ymax] + cc[xmax-1, ymax]/2.)
            a4 = (cc[xmax, ymax+1] - cc[xmax, ymax-1])/2.
            a5 = (cc[xmax, ymax+1]/2. - cc[xmax, ymax] + cc[xmax, ymax-1]/2.)
            a6 = (cc[xmax+1, ymax+1] - cc[xmax+1, ymax-1] 
                - cc[xmax-1, ymax+1] + cc[xmax-1, ymax-1])/4.
            xdif = (2*a2*a5 - a4*a6) / (a6**2 - 4*a3*a5)
            ydif = (2*a3*a4 - a2*a6) / (a6**2 - 4*a3*a5)
            xmax = xmax + xdif
            ymax = ymax + ydif

    return ymax, xmax

def surface_fit(points_array, order=0):
    """
    Fit a polynomial surface to a 2-D array of values.
    
    Parameters
    ----------
    points_array : a 2-dimensional array (L x M) of points
            to which a plane will be fit
    order : maximum exponent of polynomial
    
    Returns
    -------
    surface_array : a 2-dimensional array (L x M) of points
            representing the best-fit surface
    """
    
    if order == 0:
        # Fitting with only mean, equivalent to order 0 polynomial
        surface_array = np.ones(points_array.shape) * points_array.mean()
        return surface_array
        
    elif order == 1:
        # Analytical solution for plane using linear algebra
        # grid points in X,Y
        L, M = points_array.shape
        X1, X2 = np.mgrid[:L, :M]
        # reshape independent variables into form [a, b*X1, c*X2]
        X = np.hstack((np.ones((L*M, 1)), X1.reshape((L*M, 1)), X2.reshape((L*M, 1))))
        # reshape dependent variable into column vector
        YY = points_array.reshape((L*M, 1))
        # calculate normal vector of plane: theta = [X.T X]^-1 X.T YY
        theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), YY)
        # obtain fitted plane: plane_points = X ⋅ theta
        surface_array = np.dot(X, theta).reshape((L, M))
        
        return surface_array
        
    else:
        # Linear least-squares fitting, minimize A ⋅ x - b
        # grid points in X,Y
        L, M = points_array.shape
        x, y = np.mgrid[:L, :M]
        # define matrix of coefficients for polynomial (x in matrix equation)
        coeffs = np.ones((order+1, order+1))
        # matrix of independent variables, one term (x^m y^n) per column
        a = np.zeros((x.size, coeffs.size))
        for index, (j, i) in enumerate(np.ndindex(coeffs.shape)):
            arr = coeffs[i, j] * x**i * y**j
            a[:, index] = arr.ravel()
        # perform least-squares fit
        fit = np.linalg.lstsq(a, points_array.ravel(), rcond=None)
        # obtain polynomial coefficients in array form
        fit_coeffs = fit[0].reshape(coeffs.shape)
        # obtain surface defined by polynomial with fitted coefficients
        surface_array = np.polynomial.polynomial.polyval2d(x, y, fit_coeffs)
        
        return surface_array


## Control Points -------------------------------------------------------------|

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
    reference, kernel, border_offset, spacing_ratio, mf=0.08
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
    destr_info.border_x    = int(border_offset)
    destr_info.border_y    = int(border_offset)
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
        scene, subfield_fftconj, apod_window, 
        lowpass_filter, destr_info
    ):
    """
    Locate control points

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
    subfield_offsets = np.zeros((2, destr_info.cpx, destr_info.cpy), order="F")

    # number of array elements in each subfield
    nels = destr_info.kx * destr_info.ky

    for j in range(0, destr_info.cpy):
 
        for i in range(0, destr_info.cpx):

            sub_strt_x  = int(destr_info.rcps[0,i,j] - destr_info.kx/2)
            sub_end_x   = int(sub_strt_x + destr_info.kx - 1)

            sub_strt_y  = int(destr_info.rcps[1,i,j] - destr_info.ky/2)
            sub_end_y   = int(sub_strt_y + destr_info.ky - 1)

            #cross correlation, inline
            scene_subarr = scene[sub_strt_x:sub_end_x+1, sub_strt_y:sub_end_y+1].copy()

            scene_subarr -= surface_fit(scene_subarr, destr_info.subfield_correction)

            scene_subarr_fft = np.array(np.fft.fft2(scene_subarr * apod_window), order="F")
            scene_subarr_fft = scene_subarr_fft  * subfield_fftconj[:, :, i, j] * lowpass_filter
        
            scene_subarr_ifft = np.abs(np.fft.ifft2(scene_subarr_fft), order="F")
            cc = np.roll(scene_subarr_ifft, (int(destr_info.kx/2), int(destr_info.ky/2)),
                            axis=(0, 1))
            #cc = np.fft.fftshift(scene_subarr_ifft)
            cc = np.array(cc, order="F")

            #print("Crosscorrelation Maxpos Order: ", destr_info.max_fit_method)

            ymax, xmax = crosscor_maxpos(cc, destr_info.max_fit_method)
            #print(cc.shape, ymax, xmax)

            subfield_offsets[0,i,j] = sub_strt_x + xmax
            subfield_offsets[1,i,j] = sub_strt_y + ymax

    return subfield_offsets

def controlpoint_offsets_adf(
    scene, reference, destr_info, 
    adf_pad=0.25, adf_pow=2
):
    # TODO: check that this works, is called from reg, clean up arguments
    # TODO: make work with reference subfields
    # TODO: add a "power" option - so it can compute both ADF and ADF^2
    # #def cploc(s, w, apod_mask, smou, d_info, adf2_pad=0.25):
    """
    Locate control points

    Parameters
    ----------
    scene : array
        a 2-dimensional array (L x M) containing the image to be registered
    reference : array
        the reference array
    destr_info : structure
        Destretch information
    adf_pad : float or int
        If float between 0 and 1, fraction of subfield by which to shift
        If int > 0, number of pixels by which to shift
    adf_pow : int
        Exponent for ADF function. 1 for ADF, 2 for ADF^2

    Returns
    -------
    subfield_offsets : array
        X and Y offsets for control points

    """
    subfield_offsets = np.zeros((2, destr_info.cpx, destr_info.cpy), order="F")

    # number of array elements in each subfield
    # nels = destr_info.kx * destr_info.ky

    if adf_pad < 1:
        pad_x = int(destr_info.kx * adf_pad)
        pad_y = int(destr_info.ky * adf_pad)
    elif adf_pad > 1:
        pad_x = int(adf_pad)
        pad_y = int(adf_pad)
    else:
        raise TypeError("adf_pad must be int or float > 0")

    for j in range(0, destr_info.cpy):

        for i in range(0, destr_info.cpx):

            sub_strt_x  = int(destr_info.rcps[0,i,j] - destr_info.kx/2)
            sub_end_x   = int(sub_strt_x + destr_info.kx - 1)

            sub_strt_y  = int(destr_info.rcps[1,i,j] - destr_info.ky/2)
            sub_end_y   = int(sub_strt_y + destr_info.ky - 1)

            #scene_subarr = scene[lx-pad_x:hx+pad_x, ly-pad_y:hy+pad_y]
            scene_subarr = scene[sub_strt_x-pad_x:sub_end_x+pad_x+1,
                                 sub_strt_y-pad_y:sub_end_y+pad_y+1].copy()
            ref_subarr = reference[sub_strt_x:sub_end_x+1,
                                   sub_strt_y:sub_end_y+1].copy()
            
            #print((scene_subarr[m:m+destr_info.kx, n:n+destr_info.ky]).shape)
            #print(ref_subarr.shape)

            cc = np.zeros((2*pad_x + 1, 2*pad_y + 1), order="F")
            for m in range(2*pad_x + 1):
                for n in range(2*pad_y + 1):
                    #print(m,m+destr_info.kx, n,n+destr_info.ky )
                    cc[m, n] = -np.sum(np.abs(scene_subarr[m:m+destr_info.kx, n:n+destr_info.ky]
                                              - ref_subarr))**adf_pow
#                cc4 = np.zeros((2*pad_x + 1, 2*pad_y + 1, d_info.wx, d_info.wy))
#                for m in range(2*pad_x + 1):
#                    for n in range(2*pad_y + 1):
#                        cc4[m, n] = ss[m:m+d_info.wx, n:n+d_info.wy]
#                cc = -np.sum(np.abs(cc4 - w[:, :, i, j]), (2, 3))**2

            xmax, ymax = crosscor_maxpos(cc, destr_info.max_fit_method)

            subfield_offsets[0,i,j] = sub_strt_x + destr_info.kx/2 + xmax - pad_x
            subfield_offsets[1,i,j] = sub_strt_y + destr_info.ky/2 + ymax - pad_y

    return subfield_offsets


## Regularization -------------------------------------------------------------|

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

def reg_loop_series(
        scene, ref, kernel_sizes, mf=0.08, 
        use_fft=False, adf2_pad=0.25, border_offset=4, spacing_ratio=0.5
    ):
    """
    TODO description
    Depricated?
    
    Parameters
    ----------
    scene : ndarray (nx, ny, nt)
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

    num_scenes = scene.shape[2]
    scene_d = np.zeros((scene.shape))

    start = time()
    num_kernels = len(kernel_sizes)
    windows = {}
    destr_info_d = {}
    mm_d = {}
    smou_d = {}
    rdisp_d = {}

    # d_info, rdisp = destr_control_points(ref, kernel)
    # mm = mask(d_info.wx, d_info.wy)
    # smou = smouth(d_info.wx, d_info.wy)
    for kernel1 in kernel_sizes:
        kernel = np.zeros((kernel1, kernel1))

        destr_info, rdisp = destr_control_points(ref, kernel, border_offset, spacing_ratio, mf)
        destr_info_d[kernel1] = destr_info
        rdisp_d[kernel1] = rdisp

        mm = apod_mask(destr_info.kx, destr_info.ky, destr_info.mf)
        mm_d[kernel1] = mm

        smou = smouth(destr_info.kx, destr_info.ky)
        smou_d[kernel1] = smou

        win = doref(ref, mm, destr_info)
        # win = doref(ref, mm, destr_info, use_fft)
        windows[kernel1] = win
        
    disp_l = list(rdisp.shape)
    disp_l.append(num_scenes)
    disp_t = tuple(disp_l)
    disp_all = np.zeros(disp_t)

    for t in range(num_scenes):
        for k in kernel_sizes:
            (
                scene_d[:, :, t], disp, rdisp, destr_info
            ) = reg_saved_window(
                scene[:, :, t], 
                windows[k], 
                k, 
                destr_info_d[k], 
                rdisp_d[k], 
                mm_d[k], 
                smou_d[k], 
                use_fft, 
                adf2_pad
            )
        disp_all[:, :, :, t] = disp

    end = time()
    print(f"Total elapsed time {(end - start):.4f} seconds.")
    ans = scene_d

    return ans, disp_all, rdisp, destr_info

def doreg(scene, r, d, destr_info):
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
        use_fft=False, adf_pad=0.25, adf_pow=2, 
        border_offset=4, spacing_ratio=0.5
    ):
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

    if do_timing: start = time.time()

    if use_fft:
        subfield_fftconj, subfields_images = doref(ref, apod_window, destr_info)
        # print(scene.shape, apod_window.shape, smou.shape, destr_info)
        disp = controlpoint_offsets_fft(scene, subfield_fftconj, apod_window, smou, destr_info)
    else:
        disp = controlpoint_offsets_adf(scene, ref, destr_info, adf_pad, adf_pow)
    
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

#    print(f"Total destr took: {(end - start):.5f} seconds for kernel"
 #         +f"of size {kernel_size} px.")

    return ans, disp, rdisp, destr_info


## Window ---------------------------------------------------------------------|

def doref(ref_image, apod_mask, destr_info):
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

    subfields_fftconj = np.zeros((destr_info.kx, destr_info.ky, destr_info.cpx, destr_info.cpy),
                   dtype="complex", order="F")
    
    subfields_images = np.zeros((destr_info.kx, destr_info.ky, destr_info.cpx, destr_info.cpy),
                   dtype="float32", order="F")

    # number of elements in each subfield
    nelz = destr_info.kx * destr_info.ky
    
    # from previous method for computing subfields - see comment below for better approach
    #sub_strt_y = destr_info.by
    #sub_end_y  = sub_strt_y + destr_info.wy - 1
    
    for j in range(0, destr_info.cpy):
        #sub_strt_x = destr_info.bx
        #sub_end_x  = sub_strt_x + destr_info.wx - 1

        for i in range(0, destr_info.cpx):
            # instead of incrementing the subarray positions with a fixed step
            # size, we should instead take the reference positions 
            # from the predefined control points and extract 
            # the appropriate sized subarray around those coordinates
            # This will be more flexible going forward, especially considering 
            # the possibility of irregular sampling
            # take the reference position and define the start of the box as 
            # half the kernel size to the left (below), and then add the kernel size
            # to get the right (top) boundary
            sub_strt_x  = int(destr_info.rcps[0,i,j] - destr_info.kx/2)
            sub_end_x   = int(sub_strt_x + destr_info.kx - 1)

            sub_strt_y  = int(destr_info.rcps[1,i,j] - destr_info.ky/2)
            sub_end_y   = int(sub_strt_y + destr_info.ky - 1)
            
            ref_subarr = ref_image[sub_strt_x:(sub_end_x+1), sub_strt_y:(sub_end_y+1)].copy()
            
            ref_subarr -= surface_fit(ref_subarr, destr_info.subfield_correction)
            subfields_images[:, :, i, j] = ref_subarr
                
             # store the complex conjugate of the FFT of each reference subfield 
             #    (for later calculation of the cross correlation with the target subfield)
            subfields_fftconj[:, :, i, j] = np.array(np.conj(np.fft.fft2(ref_subarr * apod_mask)), order="F")

            #sub_strt_y = sub_strt_y + destr_info.kx
            #sub_end_y = sub_end_y + destr_info.kx
        #sub_strt_y = sub_strt_y + destr_info.ky
        #sub_end_y = sub_end_y + destr_info.ky

    return subfields_fftconj, subfields_images

def reg_saved_window(
        scene, subfield_fftconj, kernel_size, destr_info, rdisp, 
        mm, smou, use_fft=False, adf2_pad=0.25
    ):
    """
    Register scenes with respect to reference image using kernel size and
    then returns the destretched scene, using precomputed window.

    Parameters
    ----------
    scene : [nx, ny] [nx, ny, nf]
        Scene to be registered
    subfield_fftconj: [nx, ny, nf]
        FFT of the reference scene (computed with doref)
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
    kernel = np.zeros((kernel_size, kernel_size))

    # d_info, rdisp = destr_control_points(ref, kernel)
    # mm = mask(d_info.wx, d_info.wy)
    # smou = smouth(d_info.wx, d_info.wy)
    #Condition the ref
    #subfield_fftconj = doref(ref, mm, d_info)

    ssz = scene.shape
    ans = np.zeros((ssz[0], ssz[1]), order="F")

    # compute control point locations

    #start = time()
    disp = controlpoint_offsets_fft(scene, subfield_fftconj, mm, smou, destr_info)
    #end = time()
    #dtime = end - start
    #print(f"Time for a scene destretch is {dtime:.3f}")

    #disp = repair(rdisp, disp, d_info) # optional repair
    #rms = sqrt(total((rdisp - disp)^2)/n_elements(rdisp))
    #print, 'rms =', rms
    #mdisp = np.mean(rdisp-disp,axis=(1, 2))
    #disp[0, :, :] += mdisp[0]
    #disp[1, :, :] += mdisp[1]
    x = doreg(scene, rdisp, disp, destr_info)
    ans = x
        #    win = doref (x, mm); optional update of window

    #print(f"Total destr took: {(end - start):.5f} seconds for kernel"
    #      +f"of size {kernel_size} px.")

    return ans, disp, rdisp, destr_info
