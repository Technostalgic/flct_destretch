"""
Created on Tue Oct 13 16:09:55 2020

NOTE: DEPRICATED - Destretching routines for removing optical defects following the
reg.pro by Phil Wiborg and Thomas Rimmele in reg.pro

@author: molnarad
"""

import numpy as np
from scipy import signal as signal
import matplotlib.pyplot as plt
from time import time

# internal
from destretch_params import DestretchParams
import processing
import algorithm

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

    destr_info, rdisp = algorithm.destr_control_points(ref, kernel)


    #mm = np.zeros((d_info.wx,d_info.wy), order="F")
    #mm[:, :] = 1
    mm = processing.apod_mask(destr_info.wx, destr_info.wy, destr_info.mf)

    smou = np.zeros((destr_info.wx,destr_info.wy), order="F")
    smou[:, :] = 1


    ref = ref/np.average(ref)*np.average(scene)
    subfield_fftconj = algorithm.doref(ref, mm, destr_info)

    ssz = scene.shape
    nf = ssz[2]
    ans = np.zeros((2, destr_info.cpx, destr_info.cpy, nf), order="F")

    # compute control point locations
    if destr_info.use_fft:
        for frm in range(0, nf):
            ans[:, :, :, frm] = \
                algorithm.controlpoint_offsets_fft(
                    scene[:, :, :, frm], 
                    subfield_fftconj, 
                    smou, destr_info, adf2_pad
                )
    else:
        for frm in range(0, nf):
            ans[:, :, :, frm] = algorithm.controlpoint_offsets_adf(scene[:, :, :, frm], subfield_fftconj, smou, destr_info, adf2_pad)
        #ans[:, :, :, frm] = repair(rdisp, ans[:, :, :,frm], d_info)# optional repair

    if ssz[2]:
        scene = np.reshape(scene, (ssz[0], ssz[1]))
        ans = np.reshape(ans, (2, destr_info.cpx, destr_info.cpy))

    return ans

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
        scene_temp, disp, rdisp, destr_info = algorithm.reg(scene_temp, ref, el, mf, use_fft, adf2_pad)

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

        destr_info, rdisp = algorithm.destr_control_points(ref, kernel, border_offset, spacing_ratio, mf)
        destr_info_d[kernel1] = destr_info
        rdisp_d[kernel1] = rdisp

        mm = processing.apod_mask(destr_info.kx, destr_info.ky, destr_info.mf)
        mm_d[kernel1] = mm

        smou = processing.smouth(destr_info.kx, destr_info.ky)
        smou_d[kernel1] = smou

        # TODO review diff (master)
        win = algorithm.doref(ref, mm, destr_info)
        # win = doref(ref, mm, destr_info, use_fft)

        windows[kernel1] = win
        
    disp_l = list(rdisp.shape)
    disp_l.append(num_scenes)
    disp_t = tuple(disp_l)
    disp_all = np.zeros(disp_t)

    for t in range(num_scenes):
        for k in kernel_sizes:
            scene_d[:, :, t], disp, rdisp, destr_info = (algorithm.reg_saved_window(
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
