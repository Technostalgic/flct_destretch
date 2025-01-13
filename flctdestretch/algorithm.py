"""
Primary algorithm module to perform flct destretching image processing based 
on implementation by Momchil Molnar
"""


## Imports and Initialization -------------------------------------------------|

import time
import numpy as np
from scipy.interpolate import RectBivariateSpline
from typing import Tuple, Literal, Any

# internal
from destretch_params import DestretchParams
import processing


## Control Points -------------------------------------------------------------|

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

    # TODO change step size here - it is 1, we want it to be based on kernel 
    # size and resolution (remember spacing ratio is how much each kernel 
    # overlaps)
    xy_ref_coordinates[0, :, :] = [
        np.linspace(0, (scene_nx-1), num=scene_ny, dtype="int") 
        for el in range(scene_nx)
    ]
    xy_ref_coordinates[1, :, :] = [
        np.zeros(scene_ny, dtype="int") + el
        for el in range(scene_nx)
    ]

    xy_ref_coordinates = np.swapaxes(xy_ref_coordinates, 1, 2)

    dd = disp - rdisp

    interp_x = RectBivariateSpline(cp_x_coords, cp_y_coords, dd[0, :, :])
    interp_y = RectBivariateSpline(cp_x_coords, cp_y_coords, dd[1, :, :])

    xy_grid = np.zeros((2, scene_nx, scene_ny))

    x_coords_output = np.linspace(0, scene_nx-1, num=scene_nx)
    y_coords_output = np.linspace(0, scene_ny-1, num=scene_ny)

    xy_grid[1, :, :] = 1. * interp_x.__call__(
        x_coords_output, 
        y_coords_output, 
        grid=True
    )
    xy_grid[0, :, :] = 1. *interp_y.__call__(
        x_coords_output, 
        y_coords_output, 
        grid=True
    )

    # TODO implement proper test
    # if test == True:
    #     im1 = plt.imshow(xy_grid[0, :, :])
    #     plt.colorbar(im1)
    #     plt.show()
    # 
    #     im2 = plt.imshow(xy_grid[1, :, :])
    #     plt.colorbar(im2)
    #     plt.show()

    xy_grid += xy_ref_coordinates

    return (xy_grid)

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
    # the assumption is that the reference is a 2D array, so we only need 
    # the x- and y-dimensions
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

    # [wx,wy] define the size of a border around the edge of the image, 
    # to add an additional buffer area in which to avoid placing the 
    # control points.
    # The border_offset input variable defines this border area in relation to 
    # the kernel size, but maybe it's better to define it as an absolute 
    # number of pixels?
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
        # old way of defining the control points by looping through x and 
        # way and adding a fixed offset to the previously defined control point

        destr_info.bx = int((
            (destr_info.ref_sz_x - destr_info.wx + destr_info.kx) % 
            destr_info.kx
        ) / 2)
        destr_info.by = int((
            (destr_info.ref_sz_y - destr_info.wy + destr_info.ky) 
            % destr_info.ky
        ) / 2)
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
        # the control points must start and end at least 1/2 kernel width 
        # away from the edges of the array so that means that the allowable 
        # range of pixels available for control points is reduced by 
        # (at minimum) one kernel width. 
        # it is also reduced by the size of the extra border on each side
        allowable_range_x = (
            destr_info.ref_sz_x - 
            destr_info.kx - 
            (destr_info.border_x * 2)
        )
        allowable_range_y =(
            destr_info.ref_sz_y - 
            destr_info.ky - 
            (destr_info.border_y * 2)
        )
        
        # how far apart should the sub-array control points be placed, in 
        # units of the kernel width
        # set the spacing between subarrays, making sure it is divisible 
        # by 2 (just because...)
        destr_info.spacing_x  = int(destr_info.kx * spacing_ratio)
        destr_info.spacing_y  = int(destr_info.ky * spacing_ratio)
        destr_info.spacing_x += destr_info.spacing_x % 2
        destr_info.spacing_y += destr_info.spacing_y % 2

        # divide the number of allowable pixels by the control points, round 
        # down to nearest integer
        num_grid_x = int(allowable_range_x / destr_info.spacing_x) + 1
        num_grid_y = int(allowable_range_y / destr_info.spacing_y) + 1
        destr_info.cpx = num_grid_x
        destr_info.cpy = num_grid_y
        
        # how far apart will the first and last control points be, in each axis
        total_range_x = destr_info.spacing_x * (num_grid_x - 1)
        total_range_y = destr_info.spacing_y * (num_grid_y - 1)
        # the total range will be less than the maximum possible range, in 
        # most cases so allocate some of those extra pixels to each border
        destr_info.bx = np.round((
            allowable_range_x - 
            total_range_x + 
            destr_info.kx
        ) / 2.)
        destr_info.by = np.round((
            allowable_range_y - 
            total_range_y + 
            destr_info.ky
        ) / 2.)
        
        destr_info.mf = mf

        if destr_info.debug >= 2: 
            print(
                'Number of Control Points = ',
                num_grid_x, ' x ', num_grid_y
            )
        if destr_info.debug >= 2: 
            print(
                'Number of Border Pixels = ', 
                destr_info.bx , 
                ' x ', 
                destr_info.by
            )
        if destr_info.debug >= 3: 
            print(
                'allowable range,grid spacing x, num grid x' + 
                ', total range x, start pos x',
                allowable_range_x,destr_info.spacing_x,
                num_grid_x,
                total_range_x,
                destr_info.bx
            )

        rcps = np.zeros([2, destr_info.cpx, destr_info.cpy])
        rcps[0,:,:] = np.transpose(
            np.tile(
                (
                    np.arange(destr_info.cpx) * destr_info.spacing_x + 
                    destr_info.bx
                ), 
                (destr_info.cpy, 1)
            )
        )
        rcps[1,:,:] = np.tile(
            (
                np.arange(destr_info.cpy) * destr_info.spacing_y + 
                destr_info.by
            ),
            (destr_info.cpx, 1)
        )
        destr_info.rcps = rcps
        
    return destr_info, rcps

def controlpoint_offsets_fft(
        scene, subfield_fftconj, apod_mask, 
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
            #ss = s[lx:hx, ly:hy]
            scene_subarr = scene[sub_strt_x:sub_end_x+1, sub_strt_y:sub_end_y+1].copy()

            scene_subarr -= processing.surface_fit(scene_subarr, destr_info.subfield_correction)

            #ss = (ss - np.polyfit(ss[0, :], ss[1 1))*mask
            scene_subarr_fft = np.array(np.fft.fft2(scene_subarr), order="F")
            scene_subarr_fft = scene_subarr_fft  * subfield_fftconj[:, :, i, j] * lowpass_filter
        
            scene_subarr_ifft = np.abs(np.fft.ifft2(scene_subarr_fft), order="F")
            cc = np.roll(scene_subarr_ifft, (int(destr_info.kx/2), int(destr_info.ky/2)),
                            axis=(0, 1))
            #cc = np.fft.fftshift(scene_subarr_ifft)
            cc = np.array(cc, order="F")

            #print("Crosscorrelation Maxpos Order: ", destr_info.max_fit_method)

            xmax, ymax = processing.crosscor_maxpos(cc, destr_info.max_fit_method)

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
    subfield_offsets = np.zeros(
        (2, destr_info.cpx, destr_info.cpy), 
        order="F"
    )

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
            scene_subarr = scene[
                sub_strt_x-pad_x:sub_end_x+pad_x+1, 
                sub_strt_y-pad_y:sub_end_y+pad_y+1
            ]
            ref_subarr = reference[
                sub_strt_x:sub_end_x+1,
                sub_strt_y:sub_end_y+1
            ]

            cc = np.zeros(
                (2 * pad_x + 1, 2 * pad_y + 1), 
                order="F"
            )
            for m in range(2*pad_x + 1):
                for n in range(2*pad_y + 1):
                    #print(m,m+destr_info.kx, n,n+destr_info.ky )
                    cc[m, n] = -np.sum(np.abs(
                        scene_subarr[m:m+destr_info.kx, n:n+destr_info.ky] - 
                        ref_subarr
                    )) ** adf_pow
                # cc4 = np.zeros(
                #   (2*pad_x + 1, 2*pad_y + 1, d_info.wx, d_info.wy)
                # )
                # for m in range(2*pad_x + 1):
                #   for n in range(2*pad_y + 1):
                #       cc4[m, n] = ss[m:m+d_info.wx, n:n+d_info.wy]
                # cc = -np.sum(np.abs(cc4 - w[:, :, i, j]), (2, 3))**2

            xmax, ymax = processing.crosscor_maxpos(
                cc, destr_info.max_fit_method
            )

            subfield_offsets[0,i,j] = (
                sub_strt_x + destr_info.kx/2 + xmax - pad_x
            )
            subfield_offsets[1,i,j] = (
                sub_strt_y + destr_info.ky/2 + ymax - pad_y
            )

    return subfield_offsets


## Reg ------------------------------------------------------------------------|

DestretchLoopResult = Tuple[
    np.ndarray[Any, np.dtype[np.float64]], 
    np.ndarray[tuple[Literal[2], Any, Any], np.dtype[np.float64]],
    np.ndarray[tuple[Literal[2], Any, Any], np.dtype[np.float64]],
    DestretchParams
]

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
    start = time.time()

    disp_sum     = np.zeros((2,scene_nx, scene_ny))
    offsets_sum  = np.zeros((2,scene_nx, scene_ny))
    rdisp_sum    = np.zeros((2,scene_nx, scene_ny))
    kernel_count = 0.0

    for kernel_dim in kernel_sizes:
        scene_temp, disp, rdisp, destr_info = reg(
            scene_temp, ref, 
            kernel_dim, mf, 
            use_fft, adf2_pad, 
            adf_pow, border_offset, 
            spacing_ratio
        )
        # remap displacements onto spatial grid of scene 
        # (i.e. the same number of pixels as the input image)
        dispmap_new, offsets_new  = bilin_control_points(scene, rdisp, disp)
        # add the displacement and offset maps to
        disp_sum     += dispmap_new
        offsets_sum  += offsets_new
        rdisp_sum    += dispmap_new - offsets_new
        kernel_count += 1

    # the displacement maps contain the pixel reference coordinates, so 
    # adding them iteratively sums those reference coordinates
    # divide by the number of maps summed to get back to the rate coordinates
    disp_sum /= kernel_count
    rdisp_sum /= kernel_count

    end = time.time()
    # print(f"Total elapsed time {(end - start):.4f} seconds.")
    ans = scene_temp

    return ans, disp_sum, rdisp_sum, destr_info

def reg_loop_series(
        scene, ref, kernel_sizes, mf=0.08, 
        use_fft=False, adf2_pad=0.25, border_offset=4, spacing_ratio=0.5
    ):
    """
    TODO description
    
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

    start = time.time()
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

        destr_info, rdisp = destr_control_points(
            ref, kernel, 
            border_offset, spacing_ratio, 
            mf
        )
        destr_info_d[kernel1] = destr_info
        rdisp_d[kernel1] = rdisp

        mm = processing.apod_mask(destr_info.kx, destr_info.ky, destr_info.mf)
        mm_d[kernel1] = mm

        smou = processing.smouth(destr_info.kx, destr_info.ky)
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
                scene[:, :, t], 
                windows[k], 
                k, 
                destr_info_d[k],
                rdisp_d[k], 
                mm_d[k], 
                smou_d[k], 
                use_fft, 
                adf2_pad
            ))
        disp_all[:, :, :, t] = disp
    
    end = time.time()
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

    xy  = bilin_control_points(scene, r, d)
    ans = processing.bilin_values_scene(scene, xy, destr_info)

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
    scene -= scene.mean()
    ref -= ref.mean()
    kernel = np.zeros((kernel_size, kernel_size))

    destr_info, rdisp = destr_control_points(
        ref, 
        kernel, border_offset, 
        spacing_ratio, mf
    )
    #destr_info.subfield_correction = 0
    destr_info.subfield_correction = 1
    
    destr_info.use_fft = use_fft
    
    apod_window = processing.apod_mask(
        destr_info.kx, 
        destr_info.ky, 
        destr_info.mf
    )
    smou = processing.smouth(destr_info.kx, destr_info.ky)
    #Condition the ref

    ssz = scene.shape
    ans = np.zeros((ssz[0], ssz[1]), order="F")

    # compute control point locations

    #start = time()
    #if use_fft: 
    #    subfield_fftconj = doref(ref, apod_window, d_info, use_fft)
    #    disp = controlpoint_offsets_fft(
    #       scene, subfield_fftconj, apod_window, smou, d_info
    #    )
    #else:
    #    disp = controlpoint_offsets_adf(scene, ref, smou, d_info,

    if use_fft:
        subfield_fftconj = doref(ref, apod_window, destr_info)
        disp = controlpoint_offsets_fft(
            scene, subfield_fftconj, 
            apod_window, smou, destr_info
        )
    else:
        disp = controlpoint_offsets_adf(
            scene, ref, 
            destr_info, adf_pad, 
            adf_pow
        )
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
    #   + f"of size {kernel_size} px.")

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

    subfields_fftconj = np.zeros(
        (destr_info.kx, destr_info.ky, destr_info.cpx, destr_info.cpy),
        dtype="complex", order="F"
    )

    # number of elements in each subfield
    nelz = destr_info.kx * destr_info.ky
    
    # from previous method for computing subfields - see comment below 
    # for better approach
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
            # half the kernel size to the left (below), and then add the 
            # kernel size to get the right (top) boundary
            sub_strt_x  = int(destr_info.rcps[0,i,j] - destr_info.kx/2)
            sub_end_x   = int(sub_strt_x + destr_info.kx - 1)

            sub_strt_y  = int(destr_info.rcps[1,i,j] - destr_info.ky/2)
            sub_end_y   = int(sub_strt_y + destr_info.ky - 1)
            
            ref_subarr = ref_image[
                sub_strt_x :
                (sub_end_x+1), sub_strt_y:(sub_end_y+1)
            ].copy()
            # BUGFIX for darkness artifacts
            
            ref_subarr -= processing.surface_fit(
                ref_subarr, 
                destr_info.subfield_correction
            )
                
             # store the complex conjugate of the FFT of each reference 
             # subfield (for later calculation of the cross correlation with 
             # the target subfield)
            subfields_fftconj[:, :, i, j] = np.array(
                np.conj(np.fft.fft2(ref_subarr * apod_mask)), 
                order="F"
            )

            #sub_strt_y = sub_strt_y + destr_info.kx
            #sub_end_y = sub_end_y + destr_info.kx
        #sub_strt_y = sub_strt_y + destr_info.ky
        #sub_end_y = sub_end_y + destr_info.ky


    return subfields_fftconj

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
    disp = controlpoint_offsets_fft(
        scene, subfield_fftconj, 
        mm, smou, destr_info
    )
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
