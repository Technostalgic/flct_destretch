"""
Utility processing module for suplementing flct destretching algorithms based 
on implementation by Momchil Molnar
"""


## Imports and Initialization -------------------------------------------------|

import numpy as np
from scipy import signal as signal


## Implementation -------------------------------------------------------------|

def bilin_values_scene(
    scene, coords_new, destr_info, 
    nearest_neighbor = False
):
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
        
        ssfx00 = (scene_float[x1, y0] - ss00) * fx
        ssfy01 = (
            ss01 - ss00 + (scene_float[x1, y1] - ss01) * fx - ssfx00
        ) * fy

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

    left = np.zeros(taper_wx) # filt_x[:taper_wx]
    right = left[::-1]
    top = np.zeros(taper_wx) # filt_y[:taper_wy]
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
            a2 = (
                cc[xmax+1, ymax] - 
                cc[xmax-1, ymax]
            ) / 2.
            a3 = (
                cc[xmax+1, ymax]/2. - 
                cc[xmax, ymax] + 
                cc[xmax-1, ymax]/2.
            )
            a4 = (
                cc[xmax, ymax+1] - 
                cc[xmax, ymax-1]
            ) / 2.
            a5 = (
                cc[xmax, ymax+1]/2. - 
                cc[xmax, ymax] + 
                cc[xmax, ymax-1]/2.
            )
            a6 = (
                cc[xmax+1, ymax+1] - 
                cc[xmax+1, ymax-1] - 
                cc[xmax-1, ymax+1] + 
                cc[xmax-1, ymax-1]
            ) / 4.
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
        X = np.hstack((
            np.ones((L*M, 1)), 
            X1.reshape((L*M, 1)), 
            X2.reshape((L*M, 1))
        ))
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
