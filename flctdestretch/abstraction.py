import enum
import os
import time

import numpy as np
import astropy.io.fits as fits
from astropy.io.fits.hdu import HDUList, ImageHDU, CompImageHDU

from algorithm import reg_loop, reg_loop_series
from destretch_params import DestretchParams
import utility

def destretch_files(
        filepaths: list[os.PathLike], 
        kernel_sizes: list[int], 
        index_schema: utility.IndexSchema
    ) -> list[
        np.ndarray | DestretchParams
    ]:
    """
    Compute the destretched result of data from all given files

    Parameters
    ----------
    filepaths : list[str]
        list of files in order to treat as image sequence for destretching
    kernel_sizes : ndarray (kx, ky, 2)
        the destretch kernel size, the size of each subframe to be processed
    index_schema : reference_method.IndexSchema
        the index schema to use for the input data files

    Returns
    -------
    result : list[np.ndarray, np.ndarray, np.ndarray, DestretchParams]
        same output as reg_loop_series
    """

    time_begin = time.time()

    # used to enforce the same resolution for all data files
    image_resolution = (-1,-1)

    # store the results
    result_sequence: list[np.ndarray] = []

    # reference image to feed into next destretch loop
    reference_image: np.ndarray | None = None

    # read each image HDU in each file
    print("Searching for image data in specified files...")
    for path in filepaths:
        
        # convert all data to default index schema
        image_data: np.ndarray = utility.load_image_data(path)

        if len(image_data.shape) == 2:

            # ensure all data matches the first file's resolution
            if image_resolution[0] < 0:
                image_resolution = (
                    image_data.shape[0], 
                    image_data.shape[1]
                )
            elif (
                image_resolution[0] != image_data.shape[0] or 
                image_resolution[1] != image_data.shape[1]
            ): 
                raise Exception(
                    f"Resolution mismatch for '{os.fspath(path)}'"
                )

            # TODO seperate image data by z axis
            # image_data = np.moveaxis(image_data, 0, -1)[:,:]

            # debug log
            # print(f"found {hdu_index} in {path}: {image_data.shape}")

            # if no previous image, use self for reference and result 
            # image, do not process
            if reference_image is None:
                reference_image = image_data
                result_sequence.append(image_data)
                print("initial image to be used as reference")
                continue

            image_num = len(result_sequence) + 1
            print(f"processing image #{image_num}..")

            # perform image destretching
            result = reg_loop(
                image_data,
                reference_image,
                kernel_sizes,
                mf=0.08,
                use_fft=True
            )

            # decompose result
            answer: np.ndarray
            destretch_info: DestretchParams
            answer, _, _, destretch_info = result

            # save the processed image in the results array
            result_sequence.append(answer)

            # store previously processed frame as reference image to use
            # for nex image in series
            reference_image = result_sequence[-1]

    # output time elapsed
    time_end = time.time()
    print(f"Time elapsed: {time_end - time_begin} seconds")

    return result_sequence