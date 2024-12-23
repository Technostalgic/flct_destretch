import enum
import os
import time

import numpy as np

from algorithm import reg_loop
from destretch_params import DestretchParams
import utility
import reference_method

def destretch_files(
        filepaths: list[os.PathLike], 
        kernel_sizes: list[int], 
        index_schema: utility.IndexSchema,
        ref_method: reference_method.RefMethod = None
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
    ref_method : reference_method.RefMethod
        the method to use to apply the reference image for destretching, if not
        specified, will use reference_method.OMargin

    Returns
    -------
    result : list[np.ndarray, np.ndarray, np.ndarray, DestretchParams]
        same output as reg_loop_series
    """

    time_begin = time.time()

    # set default ref method
    if ref_method is None:
        ref_method = reference_method.OMargin.create(
            filepaths, 
            index_schema, 
            5, 5
        )

    # used to enforce the same resolution for all data files
    image_resolution = (-1,-1)

    # store the results
    result_sequence: list[np.ndarray] = []

    # reference image to feed into next destretch loop
    reference_image: np.ndarray | None = None

    # read each image HDU in each file
    print("Searching for image data in specified files...")
    for i in range(len(filepaths)):
        path = filepaths[i]
        
        # pass data to reference method so it can do its thing
        match ref_method:
            case reference_method.OMargin():
                ref_method.pass_params(i)
        
        # get the image data from ref method if available otherwise load it
        image_data: np.ndarray = ref_method.get_original_data(i)
        if image_data is None:
            image_data = utility.load_image_data(path)

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

        # debug log
        # print(f"found {hdu_index} in {path}: {image_data.shape}")

        # get the reference image from the specified reference method
        reference_image = ref_method.get_reference(i)

        # perform image destretching
        print(f"processing image #{i}..")
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
        # use final displacement sum 'disp_sum' - 'rdisp_sum'
        answer, disp_sum, rdisp_sum, destretch_info = result

        # save the processed image in the results array
        result_sequence.append(answer)

    # output time elapsed
    time_end = time.time()
    print(f"Time elapsed: {time_end - time_begin} seconds")

    return result_sequence