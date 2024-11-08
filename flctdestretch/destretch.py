#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 16:09:55 2020

Destretching routines for removing optical defects following the
reg.pro by Phil Wiborg and Thomas Rimmele in reg.pro

@author: molnarad
"""

import numpy as np
from scipy import signal as signal
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
from time import time

class Destretch_params():
    """
    Class containing all the information about then
    TODO include field descriptors in docstring (if possible?)
    """
    def __init__(self, 
            kx, ky, wx, wy, bx, by, cpx, cpy, mf, rcps, ref_sz_x, ref_sz_y, 
            scene_sz_x, scene_sz_y, subfield_correction, 
            max_fit_method, use_fft, do_plots, debug
        ):
        """
        TODO docstring
        """
        # kernel size x,y
        self.kx = kx        
        self.ky = ky       
        
        # border offset x,y
        self.wx = wx
        self.wy = wy

        # boundary size x,y
        self.bx = bx
        self.by = by
        
        # number of control points x,y
        self.cpx = cpx
        self.cpy = cpy
        
        # apodization percentage
        self.mf = mf

        # array of control points
        self.rcps = rcps

        # TODO describe these fields
        self.ref_sz_x = ref_sz_x 
        self.ref_sz_y = ref_sz_y 
        self.scene_sz_x = scene_sz_x 
        self.scene_sz_y = scene_sz_y 

        # order of polynomial to subtract from subfields
        self.subfield_correction = subfield_correction  
        
        self.max_fit_method = max_fit_method
        self.use_fft = use_fft
        self.do_plots = do_plots
        self.debug = debug

    def print_props(self):
        print("[kx, ky, wx, wy, bx, by, cpx, cpy, mf] are:")
        print(self.kx, self.ky, self.wx, self.wy, self.bx, self.by,
              self.cpx, self.cpy, self.mf)

def plot_cps(ax_object, destr_info):
    """
    TODO
    Plot the control points for the destretching on the destretched
    image

    Parameters
    ----------
    ax_object : matplotlib ax object
        Axis object to have the control points plotted on;
    destr_info : class Destretch_params
        Destretch Parameters;

    Returns
    -------
    """
    raise NotImplementedError("This function has not been implemented yet")

def bilin_values_scene(scene, coords_new, destr_info, nearest_neighbor = False):
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
        x = np.array(coords_new[:, :, 0] + .5, order="F")
        y = np.array(coords_new[:, :, 1] + .5, order="F")
        
        scene_interp = scene[x, y]

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
        print(scene_float.shape, scene_float.dtype)
        #scene_float = scene.copy

        ss00 = scene_float[x0, y0]
        ss01 = scene_float[x0, y1]
        ssfx00 =                (scene_float[x1, y0] - ss00) * fx
        ssfy01 = (ss01 - ss00 + (scene_float[x1, y1] - ss01) * fx - ssfx00) * fy
        scene_interp  = ss00 + ssfx00 + ssfy01

    return scene_interp

def bilin_control_points(scene, rdisp, disp, test=False):
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

    #compute the control points locations
    cp_x_coords = rdisp[0, :, 0]
    cp_y_coords = rdisp[1, 0, :]

    #compute the displacements

    xy_ref_coordinates1 = np.zeros((2, scene_nx, scene_ny), order="F")
    xy_ref_coordinates = np.zeros((2, scene_nx, scene_ny), order="F")

    xy_ref_coordinates[0, :, :] = [np.linspace(0, (scene_nx-1),
                                               num=scene_ny, dtype="int")
                                   for el in range(scene_nx)]
    xy_ref_coordinates[1, :, :] = [np.zeros(scene_ny, dtype="int")+el
                                   for el in range(scene_nx)]

    xy_ref_coordinates = np.swapaxes(xy_ref_coordinates, 1, 2)

    dd = disp - rdisp

    interp_x = RectBivariateSpline(cp_x_coords, cp_y_coords, dd[0, :, :])
    interp_y = RectBivariateSpline(cp_x_coords, cp_y_coords, dd[1, :, :])

    xy_grid = np.zeros((2, scene_nx, scene_ny))

    x_coords_output = np.linspace(0, scene_nx-1, num=scene_nx)
    y_coords_output = np.linspace(0, scene_ny-1, num=scene_ny)

    xy_grid[1, :, :] = 1. * interp_x.__call__(x_coords_output, y_coords_output,
                                         grid=True)
    xy_grid[0, :, :] = 1. *interp_y.__call__(x_coords_output, y_coords_output,
                                         grid=True)
    if test == True:
        im1 = plt.imshow(xy_grid[0, :, :])
        plt.colorbar(im1)
        plt.show()

        im2 = plt.imshow(xy_grid[1, :, :])
        plt.colorbar(im2)
        plt.show()

    xy_grid += xy_ref_coordinates

    return (xy_grid)

def bspline(scene, r, dd, destr_info):
    """
    Destretch the scene using a B-spline

    Parameters
    ----------
    scene : TYPE
        Image to be destretched.
    r : TYPE
        reference displacements of control points
    dd : TYPE
        actual displacements of control points

    destr_info: Destretch_class
        info about the destretching

    Returns
    -------
    ans : TYPE
        Destretched image
    """
    # TODO validate unused parameters and calculations in this function
    # TODO (destr_info, ns, nt)

    always = 1      # exterior control points drift with interior (best)
    #always = 0     ; exterior control points fixed by ref. displacements

    # a kludgery: increases magnitude of error, since
    # curve doesn't generally pass through the tie pts.
    d = (dd-r)*1.1 + r

    ds = r[0, 1, 0]-r[0, 0, 0]
    dt = r[1, 0, 1]-r[1, 0, 0]

    dsz = d.shape

    # extend r & d to cover entire image. Two possible methods:
    if (always == True):
        # (1) this method lets boundry drift with actual displacements at
        #     edges of 'd' table.

        ns = dsz[1]
        nt = dsz[2]
        Rx, Px = extend(r[0, :, :], d[0, :, :])
        Ry, Py = extend(r[1, :, :], d[1, :, :])

        Ry = np.transpose(Ry)
        Py = np.transpose(Py)

    Ms = np.array([-1,3,-3,1, 3,-6,0,4, -3,3,3,1, 1,0,0,0],
                  order="F")/6.
    Ms = np.reshape(Ms, (4, 4))
    MsT = (Ms)

    sz = scene.shape
    nx = sz[0]
    ny = sz[1]

    ans = np.zeros((nx, ny, 2), order="F")
    for v in range(0, dsz[2]+3):
        t0 = Ry[1, v+1]
        tn = Ry[1, v+2]
        if ((tn < 0) or (t0 > ny-1)):
            break
        t0 = int(np.amax([t0, 0]))
        tn = int(np.amin([tn, ny-1]))
        t = np.arange(tn-t0)/dt + (t0-Ry[1, v+1])/dt
        for u in range(0,dsz[1]+3):
            s0 = Rx[u+1, v+1]
            sn = Rx[u+2, v+1]

            if (sn < 0) or (s0 > nx-1):
                break
            s0 = int(np.amax([s0, 0]))
            sn = int(np.amin([sn, nx-1]))
            s = np.arange(sn-s0)/ds + (s0-Rx[u+1, v+1])/ds
            compx = np.reshape(np.matmul(np.matmul(Ms,
                                                   Px[u:u+4,v:v+4]),
                                         MsT), (4, 4))

            compy = np.reshape(np.matmul(np.matmul(Ms,
                                                   Py[u:u+4,v:v+4]),
                                         MsT), (4, 4))
            ans[s0:sn, t0:tn, :] = patch(compx, compy, s, t)

def patch(compx, compy, s, t):
    """
    TBD

    Parameters
    ----------
    compx : TYPE
        DESCRIPTION.
    compy : TYPE
        DESCRIPTION.
    s : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.

    Returns
    -------
    ans: TYPE
        Description

    """
    s     = np.array(s, order="F")
    t     = np.array(t, order="F")

    len_s = len(s)
    len_t = len(t)
    ans = np.zeros((len_s, len_t, 2), order="F")

    ss = np.concatenate((s**3, s**2, s, s**0))
    ss = np.reshape(ss, (len_s, 4))
    tt = np.concatenate((t**3, t**2, t, t**0))
    tt = np.reshape(tt, (len_t, 4)).transpose()

    ans[:, :, 0] = np.matmul(np.matmul(ss, compx), tt)
    ans[:, :, 1] = np.matmul(np.matmul(ss, compy), tt)

    return ans

def extend(cntrlpts_ref, cntrlpts_actl, num_extend_pts=3):
    """Extend map of measured control points and displacements.

    Extend the maps of reference and actual control points to cover area
    outside of measured area. This is necessary to create a smooth displacement
    surface covering the whole scene

    Parameters
    ----------
    cntrlpts_ref : TYPE
        reference control points
    cntrlpts_actl : TYPE
        actual displaced position of control points

    Returns
    -------
    cntrlpts_ref_extnd : TYPE
        reference control points, extended by the appropriate border
    cntrlpts_actl_extnd : TYPE
        actual displaced position of control points, also extended
    """
    # if true, the extended border of the actual displacements will be filled
    #     with the same displacement values as at the corresponding edge.
    # if false, displacements in the extended border will be set to zero,
    #     but that may cause discontinuities in the displacement surface.
    extend_with_offsets = True

    # set the number of additional control points around the edge of the array
    # by which to extend the displacement map
    # num_extend_pts = 3

    # define the size of the extended arrays to generate
    dsz     = cntrlpts_actl.shape
    disp_nx = dsz[0] + num_extend_pts * 2
    disp_ny = dsz[1] + num_extend_pts * 2

    # First, create the entended array of control points
    cntrlpts_ref_extnd = np.zeros((disp_nx, disp_ny), order="F")

    # as currently written, one coordinate of the control point locations are
    # passed into this routine at a time. So the either the values will be varying
    # in the x-direction or the y-direction
    # We compare the differences of the change in values in the two directions
    # to identify which of the two cases we have
    step_x = cntrlpts_ref[1, 0] - cntrlpts_ref[0, 0]
    step_y = cntrlpts_ref[0, 1] - cntrlpts_ref[0, 0]

    # generally, the input arrays will contain the x- or y- coordinates of the control
    # points, so the values will only be varying in one direction if those are laid out
    # on a rectangular grid. In the other direction, the increments between points will 
    # be zero. So we just look to see in which direction is the increment bigger and 
    # use that as the step size to use for the extension.
    # But really it would be better to have the direction to use to define the step size 
    # be defined as an input, or have the step size be an input parameter.
    # This might be useful if the step size is not constant, or changes in both directions
    # simulataneously
    
    # if step_y is greater and non-zero, we'll use that to fill the rows
    if step_x > step_y:
        # define the starting point, which is num_extpts times the step size before the input position
        start_pt    = cntrlpts_ref[0, 0] - num_extend_pts * step_x
        # generate a new array of evenly spaced control points
        new_steps   = np.arange(disp_nx) * step_x + start_pt
        # replicate that array of control points into all rows of the control point array
        for i in range(disp_ny):
            cntrlpts_ref_extnd[:, i] = new_steps
    # if step_y is greater and non-zero, we'll use that to fill the columns
    else:
        # define the starting point, which is num_extpts times the step size before the input position
        start_pt    = cntrlpts_ref[0, 0] - num_extend_pts * step_y
        # generate a new array of evenly spaced control points
        new_steps   = np.arange(disp_ny) * step_y + start_pt
        # replicate that array of control points into all rows of the control point array
        for i in range(disp_nx):
            cntrlpts_ref_extnd[i, :] = new_steps

    # Next create an extended array of the displaced positions of the control points
    # and populate the center portion with the measured (input) offsets

    cntrlpts_actl_extnd = np.zeros((disp_nx, disp_ny), order="F")
    cntrlpts_actl_extnd[num_extend_pts:disp_nx-num_extend_pts, num_extend_pts:disp_ny-num_extend_pts] = cntrlpts_actl - cntrlpts_ref

    # if requested, replicate the edges of the displacement array into the extended boundaries
    if extend_with_offsets is True:
        # extend displacement array by replicating the values of the displacements along the
        #   borders of measured array. This ensures some consistencies in the values

        # take the bottom row of measured offset and replicate it into the extended array of offsets below
        # note: this could be done without a for loop...
        x = cntrlpts_actl_extnd[:, num_extend_pts]
        for i in range(num_extend_pts):
            cntrlpts_actl_extnd[:, i] = x

        # take the top row of measured offset and replicate it into the extended array of offsets above
        x = cntrlpts_actl_extnd[:, disp_ny - num_extend_pts - 1]
        for i in range(disp_ny - num_extend_pts, disp_ny):
            cntrlpts_actl_extnd[:, i] = x

        # take the left column of measured offset and replicate it into the extended array of offsets to the left
        x = cntrlpts_actl_extnd[num_extend_pts, :]
        for i in range(num_extend_pts):
            cntrlpts_actl_extnd[i, :] = x

        # take the left column of measured offset and replicate it into the extended array of offsets to the left
        x = cntrlpts_actl_extnd[disp_nx - num_extend_pts - 1, :]
        for i in range(disp_nx - num_extend_pts, disp_ny):
            cntrlpts_actl_extnd[i, :] = x
        print(cntrlpts_actl_extnd[2, :])

    # now add the control point positions back into the displacement array
    cntrlpts_actl_extnd = cntrlpts_actl_extnd + cntrlpts_ref_extnd

    return cntrlpts_ref_extnd, cntrlpts_actl_extnd

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
            
            ref_subarr = ref_image[sub_strt_x:(sub_end_x+1), sub_strt_y:(sub_end_y+1)]
            
            ref_subarr -= surface_fit(ref_subarr, destr_info.subfield_correction)
                
             # store the complex conjugate of the FFT of each reference subfield 
             #    (for later calculation of the cross correlation with the target subfield)
            subfields_fftconj[:, :, i, j] = np.array(np.conj(np.fft.fft2(ref_subarr * apod_mask)), order="F")

            #sub_strt_y = sub_strt_y + destr_info.kx
            #sub_end_y = sub_end_y + destr_info.kx
        #sub_strt_y = sub_strt_y + destr_info.ky
        #sub_end_y = sub_end_y + destr_info.ky


    return subfields_fftconj

def crosscor_maxpos(cc, order=1):
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
        if order == 1:
            denom = 2 * mx - cc[xmax-1,ymax] - cc[xmax+1,ymax]
            xfra = (xmax-1/2) + (mx-cc[xmax-1,ymax])/denom

            denom = 2 * mx - cc[xmax,ymax-1] - cc[xmax,ymax+1]
            yfra = (ymax-1/2) + (mx-cc[xmax,ymax-1])/denom

            xmax=xfra
            ymax=yfra
        elif order == 2:
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

# **********************************************************
# ******************** FUNCTION: controlpoint_offsets_fft  *******************
# **********************************************************
def controlpoint_offsets_fft(scene, subfield_fftconj, apod_mask, lowpass_filter, destr_info):
# TODO: check that this works, is called from reg
# TODO: make plane_subtraction option work
#def cploc(s, w, apod_mask, smou, d_info, adf2_pad=0.25):
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
            #ss = s[lx:hx, ly:hy]
            scene_subarr = scene[sub_strt_x:sub_end_x+1, sub_strt_y:sub_end_y+1]

            scene_subarr -= surface_fit(scene_subarr, destr_info.subfield_correction)

            #ss = (ss - np.polyfit(ss[0, :], ss[1 1))*mask
            scene_subarr_fft = np.array(np.fft.fft2(scene_subarr), order="F")
            scene_subarr_fft = scene_subarr_fft  * subfield_fftconj[:, :, i, j] * lowpass_filter
        
            scene_subarr_ifft = np.abs(np.fft.ifft2(scene_subarr_fft), order="F")
            cc = np.roll(scene_subarr_ifft, (int(destr_info.kx/2), int(destr_info.ky/2)),
                            axis=(0, 1))
            #cc = np.fft.fftshift(scene_subarr_ifft)
            cc = np.array(cc, order="F")

            xmax, ymax = crosscor_maxpos(cc, destr_info.max_fit_method)

            subfield_offsets[0,i,j] = sub_strt_x + xmax
            subfield_offsets[1,i,j] = sub_strt_y + ymax

    return subfield_offsets

# **********************************************************
# ******************** FUNCTION: controlpoint_offsets_adf  *******************
# **********************************************************
def controlpoint_offsets_adf(scene, reference, destr_info, adf_pad=0.25, adf_pow=2):
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
    nels = destr_info.kx * destr_info.ky

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
                                 sub_strt_y-pad_y:sub_end_y+pad_y+1]
            ref_subarr = reference[sub_strt_x:sub_end_x+1,
                                   sub_strt_y:sub_end_y+1]

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

def val(scene, ref, kernel):
     """
     TODO docstring
     """
     # TODO check parameters are reasonable
     # scene, ref, kernel: as defined by 'reg'

     ssz = scene.shape
     rsz = ref.shape
     ksz = kernel.shape
     errflg = 0

     if ((len(ssz) != 2) and (len(ssz) != 3)):
        print("argument 'scene' must be 2-D or 3-D")
        errflg = errflg + 1


     if len(rsz) != 2:
        print("argument 'ref' must be 2-D")
        errflg = errflg + 1

     if ((ssz[0] != rsz[0]) or (ssz[1] != rsz[2])):
         print("arguments 'scene' & 'ref' 1st 2 dimensions must agree")
         errflg = errflg + 1

     if len(ksz) != 2:
        print, "argument kernel must be 2-D"
        errflg = errflg + 1

     if errflg > 0:
         print("quitting - too many errors")

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

    xy  = bilin_control_points(scene, r, d)
    ans = bilin_values_scene(scene, xy, destr_info)

    return ans

def measure_destr_properties(scene1, scene2, destr_info):
    """
    TODO
    Measure the suitable parameters for the destretch based on the two provided
    images based on:
        1)Fourier transforms of the images to obtain the smallest
    scale (box size) on the image;
        2) Fourier transform of the image ratio (???) to measure the
    control point spacing;

    Input:
        -- scene1 -- ndarray (nx, ny)
            Image 1
        -- scene 2 -- ndarray (nx, ny)
            Image 2

    Output:
        -- destr_info -- Destretch_class
            Suggested properties of the destretch
    """
    raise NotImplementedError("This function has not been implemented yet")

def mkcps_nonuniform(ref, kernel):
    """
    TODO
    """
    raise NotImplementedError("This function has not been implemented yet")

def mkcps_overlapping(ref, kernel, box_size):
    """
    TODO
    Create control point locations in the reference with overlapping cross
    correlation regions
    """
    raise NotImplementedError("This function has not been implemented yet")

# *************************************************************************
# ********************  FUNCTION: destr_control_points  *******************
# ********************            nee mkcps             *******************
# *************************************************************************
def destr_control_points(reference, kernel, border_offset, spacing_ratio, mf=0.08):
#def mkcps(ref, kernel, mf=0.08):
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
    
    # TODO review diff (master)
    destr_info = Destretch_params(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0)

    # determine the number of pixels in the kernel
    ksz = kernel.shape
    destr_info.kx = ksz[0]
    destr_info.ky = ksz[1]

    # determine the number of pixels in the reference image
    # the assumption is that the reference is a 2D array, so we only need the x- and y-dimensions
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

    # [wx,wy] define the size of a border around the edge of the image, to add an additional 
    #     buffer area in which to avoid placing the control points.
    # The border_offset input variable defines this border area in relation to the kernel size,
    #     but maybe it's better to define it as an absolute number of pixels?
    destr_info.border_x    = int(border_offset)
    destr_info.border_y    = int(border_offset)
    # make sure [border_x,border_y] is divisible by 2
    if (destr_info.border_x % 2):
        destr_info.border_x = int(destr_info.border_x + 1)
    if (destr_info.border_y % 2):
        destr_info.border_y = int(destr_info.border_y + 1)
    destr_info.border_x  += destr_info.border_x % 1
            
    if destr_info.debug >= 2 : print('Border Size = ',destr_info.border_x, ' x ', destr_info.border_y)
    if destr_info.debug >= 2 : print('Kernel Size = ',destr_info.kx, ' x ', destr_info.ky)
    
    if define_cntl_pts_orig:
        cpx = int((destr_info.ref_sz_x - destr_info.wx + destr_info.kx)//destr_info.kx)
        cpy = int((destr_info.ref_sz_y - destr_info.wy + destr_info.ky)//destr_info.ky)
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

# ********************  END: destr_control_points  *******************

def setup(scene, reference, kernel, destr_info):
#def setup(scene, ref, kernel, d_info):

    # determine the number of pixels in the kernel
    ksz = kernel.shape
    destr_info.kx = ksz[0]
    destr_info.ky = ksz[1]

    # determine the number of pixels in the reference image
    # the assumption is that the reference is a 2D array, so we only need the x- and y-dimensions
    rsz = reference.shape
    destr_info.ref_sz_x  = rsz[0]
    destr_info.ref_sz_y  = rsz[1]

    ssz = scene.shape
    destr_info.scene_sz_x = ksz[0]
    destr_info.scene_sz_x = ksz[1]

    errflg = 0

    if ((len(ssz) != 2) and (len(ssz) != 3)):
        print("ERROR: Input 'scene' must be a 2-D or 3-D array")
        errflg = errflg + 1

    if len(rsz) != 2:
        print("ERROR: Destretching reference 'ref' must be a 2-D array")
        errflg = errflg + 1

    if ((ssz[0] != rsz[0]) or (ssz[1] != rsz[2])):
        print("ERROR: Both the x and y dimensions of the input scene must match those of the reference")
        print("arguments 'scene' & 'ref' 1st 2 dimensions must agree")
        errflg = errflg + 1

    if len(ksz) != 2:
        print("ERROR: Destretching kernel must be 2-D array")
        errflg = errflg + 1

    if errflg > 0:
         print("ERROR: Quitting - too many errors")

    return

def undo():
    """TODO"""
    raise NotImplementedError("This function has not been implemented yet")

def repair(ref, disp, destr_info):
    """
    Check if the displacements are good

    Parameters
    ----------
    ref : TYPE
        reference coordinates
    disp : TYPE
        displacements to be checked
    destr_info : TYPE
        Destr info

    Returns
    -------
    good : TYPE
        DESCRIPTION.
    """
    # TODO validate unused param (destr_info)

    TOOFAR = .5             # user may want to change this parameter
    # TOOFAR = 1.0           # user may want to change this parameter
    # TOOFAR = 1.5           # user may want to change this parameter

    sz = disp.shape
    nx = sz[1]
    ny = sz[2]

    if (len(sz) == 4):
        nf = sz[3]
    else:
        nf = 1

    kkx = ref[0, 1, 0] - ref[0, 0, 0]
    kky = ref[1, 0, 1] - ref[1, 0, 0]
    limit = (np.amax([kkx,kky])*TOOFAR)**2

    good = disp + 0

    kps = np.reshape(disp[:, :, :], (2, nx, ny))

    diff = kps - ref

    # list of bad coordinates in this frame
    bad = np.where((diff[0, :, :]**2 + diff[1, :, :]**2) > limit, 1, 0)
    bad_count = np.sum(bad)
    i = 0
    j = 0
    while i < bad_count or j < nx*ny:
        x = i % nx
        y = i // nx

        if bad[x, y] == 1:
            good [:, x, y] = ref[:, x, y]
            i += 1
        j += 1
    return good

def cps(scene, ref, kernel, adf2_pad=0.25):
    """
    Control points for sequence destretch

    Parameters
    ----------
    scene : TYPE
        Input scene [nx, ny] or [nx, ny, nf] for which
        displacements are computed
    ref : TYPE
        Reference scene [nx, ny]
    kernel : TYPE
        Kernel size [kx, ky]
    destr_info : TYPE
        DESCRIPTION.

    Returns
    -------
    ans : TYPE
        displacement array [2, cpx, cpy, nf]

    """

    destr_info, rdisp = destr_control_points(ref, kernel)


    #mm = np.zeros((d_info.wx,d_info.wy), order="F")
    #mm[:, :] = 1
    mm = apod_mask(destr_info.wx, destr_info.wy, destr_info.mf)

    smou = np.zeros((destr_info.wx,destr_info.wy), order="F")
    smou[:, :] = 1


    ref = ref/np.average(ref)*np.average(scene)
    subfield_fftconj = doref(ref, mm, destr_info)

    ssz = scene.shape
    nf = ssz[2]
    ans = np.zeros((2, destr_info.cpx, destr_info.cpy, nf), order="F")

    # compute control point locations
    if destr_info.use_fft:
        for frm in range(0, nf):
            ans[:, :, :, frm] = \
                controlpoint_offsets_fft(
                    scene[:, :, :, frm], 
                    subfield_fftconj, 
                    smou, destr_info, adf2_pad
                )
    else:
        for frm in range(0, nf):
            ans[:, :, :, frm] = controlpoint_offsets_adf(scene[:, :, :, frm], subfield_fftconj, smou, destr_info, adf2_pad)
        #ans[:, :, :, frm] = repair(rdisp, ans[:, :, :,frm], d_info)# optional repair

    if ssz[2]:
        scene = np.reshape(scene, (ssz[0], ssz[1]))
        ans = np.reshape(ans, (2, destr_info.cpx, destr_info.cpy))

    return ans

def reg(
        scene, ref, kernel_size, mf=0.08, 
        use_fft=False, adf_pad=0.25, adf_pow=2, 
        border_offset=4, spacing_ratio=0.5
    ):
# TODO: clean up control point offset calculations - move FFT specific calls (e.g. apod) into conditional
# TODO: (here and elsewhere) rename d_info to destr_info
# TODO: add crosscorrelation choice, other parameters to destr_info; rename destr_info.mf
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
    scene -= scene.mean()
    ref -= ref.mean()
    kernel = np.zeros((kernel_size, kernel_size))

    destr_info, rdisp = destr_control_points(ref, kernel, border_offset, spacing_ratio, mf)
    #destr_info.subfield_correction = 0
    destr_info.subfield_correction = 1
    
    destr_info.use_fft = use_fft
    
    apod_window = apod_mask(destr_info.kx, destr_info.ky, destr_info.mf)
    smou = smouth(destr_info.kx, destr_info.ky)
    #Condition the ref

    ssz = scene.shape
    ans = np.zeros((ssz[0], ssz[1]), order="F")

    # compute control point locations

    #start = time()
    #if use_fft: 
    #    subfield_fftconj = doref(ref, apod_window, d_info, use_fft)
    #    disp = controlpoint_offsets_fft(scene, subfield_fftconj, apod_window, smou, d_info)
    #else:
    #    disp = controlpoint_offsets_adf(scene, ref, smou, d_info,

    if use_fft:
        subfield_fftconj = doref(ref, apod_window, destr_info)
        disp = controlpoint_offsets_fft(scene, subfield_fftconj, apod_window, smou, destr_info)
    else:
        disp = controlpoint_offsets_adf(scene, ref, destr_info, adf_pad, adf_pow)
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

#    print(f"Total destr took: {(end - start):.5f} seconds for kernel"
 #         +f"of size {kernel_size} px.")

    return ans, disp, rdisp, destr_info

def reg_saved_window(
        scene, subfield_fftconj, kernel_size, destr_info, rdisp, 
        mm, smou, use_fft=False, adf2_pad=0.25
    ):
    """
    Register scenes with respect to ref using kernel size and
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
    # end = time()
    # dtime = end - start
    # print(f"Time for a scene destretch is {dtime:.3f}")

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

def reg_loop(scene, ref, kernel_sizes, mf=0.08, use_fft=False, adf2_pad=0.25):
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

    scene_temp = scene
    start = time()

    for el in kernel_sizes:
        scene_temp, disp, rdisp, destr_info = reg(scene_temp, ref, el, mf, use_fft, adf2_pad)

    end = time()
    print(f"Total elapsed time {(end - start):.4f} seconds.")
    ans = scene_temp

    return ans, disp, rdisp, destr_info

def reg_loop_series(
        scene, ref, kernel_sizes, mf=0.08, 
        use_fft=False, adf2_pad=0.25, border_offset=4, spacing_ratio=0.5
    ):
    """
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

        # TODO review diff (master)
        win = doref(ref, mm, destr_info)
        # win = doref(ref, mm, destr_info, use_fft)

        windows[kernel1] = win
        
    disp_l = list(rdisp.shape)
    disp_l.append(num_scenes)
    disp_t = tuple(disp_l)
    disp_all = np.zeros(disp_t)

    for t in range(num_scenes):
        for k in kernel_sizes:
            scene_d[:, :, t], disp, rdisp, destr_info = (reg_saved_window(
                scene[:, :, t], windows[k], k, destr_info_d[k],
                rdisp_d[k], mm_d[k], smou_d[k], use_fft, adf2_pad
            ))
        disp_all[:, :, :, t] = disp

    end = time()
    print(f"Total elapsed time {(end - start):.4f} seconds.")
    ans = scene_d

    return ans, disp_all, rdisp, destr_info

def test_destretch(scene, ref, kernel_size, plot=False):
    start = time()
    ans1, disp, rdisp, destr_info = reg_loop(scene, ref, kernel_size)
    if plot==True:
        plt.figure(dpi=250)
        plt.imshow(scene, origin=0)
        plt.title("Original scene")
        plt.show()

        plt.figure(dpi=250)
        plt.imshow(ans1, origin=0)
        plt.title("Destretched scene")
        plt.show()

        plt.figure(dpi=250)
        plt.imshow(ref, origin=0)
        plt.title("Reference")
        plt.show()
    end = time()
    print(f"Total elapsed time for test_function is {end-start}.")

def test_rotation(scene, angle):
    """
    TODO
    Test if the code can pick up a static rotation of an image
    """
    raise NotImplementedError("This function has not been implemented yet")

def test_residual_diff():
    """
    TODO
    Test if the code reduces the distortions between the two images
    """
    raise NotImplementedError("This function has not been implemented yet")
