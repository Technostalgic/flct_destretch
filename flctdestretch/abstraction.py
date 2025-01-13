import enum
import os
import time
from typing import Callable, TypedDict

import numpy as np
from astropy.io import fits

from algorithm import reg_loop, DestretchLoopResult
from destretch_params import DestretchParams
from utility import IndexSchema, load_image_data
from reference_method import RefMethod, OMargin, PreviousRef

class IterProcessArgs(TypedDict):
    """
    All available keyword arguments for relevant functions

    Attributes
    ----------
    kernel_sizes : list[int]
        the destretch kernel size, the size of each subframe to be processed
    index_schema : reference_method.IndexSchema
        the index schema to use for the input data files
    ref_method : reference_method.RefMethod
        the method to use to apply the reference image for destretching, if not
        specified, will use reference_method.OMargin in most cases
    """
    kernel_sizes: list[int]
    index_schema: IndexSchema
    ref_method: RefMethod

def fits_file_process_iter(
        in_filepaths: list[os.PathLike],
        iter_func: Callable[[DestretchLoopResult], None],
        ** kwargs: IterProcessArgs
    ):
    """
    iterates over each data from each specified file and applies destretching
    to each of them, calling the specified iter_func and passing the result of
    destretching to that funciton

    Parameters
    ----------
    in_filepaths:
        the list of file paths to process
    iter_func:
        the function that is called for each frame after the image data has 
        been processed it
    """

    # get required kwargs or default values if not specified
    kernel_sizes: list[int] = kwargs.get("kernel_sizes", [64, 32])
    index_schema: IndexSchema = kwargs.get("index_schema", IndexSchema.XYT)
    ref_method: RefMethod = kwargs.get("ref_method", OMargin(in_filepaths))

    # used to enforce the same resolution for all data files
    image_resolution = (-1,-1)

    # reference image to feed into next destretch loop
    reference_image: np.ndarray | None = None

    # iterate through each file
    print("Searching for image data in specified files...")
    for i in range(len(in_filepaths)):
        path = in_filepaths[i]
        
        # pass data to reference method so it can do its thing
        match ref_method:
            case OMargin():
                ref_method.pass_params(i)
        
        # get the image data from ref method if available otherwise load it
        image_data: np.ndarray = ref_method.get_original_data(i)
        if image_data is None:
            image_data = load_image_data(path, index_schema)

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
            raise Exception(f"Resolution mismatch for '{os.fspath(path)}'")

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

        # call the function specified by caller
        iter_func(result)

def destretch_files(
        filepaths: list[os.PathLike], 
        ** kwargs: IterProcessArgs
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
    ** kwargs: 
        see IterProcessArgs class for all available arguments

    Returns
    -------
    result : list[np.ndarray, np.ndarray, np.ndarray, DestretchParams]
        same output as reg_loop_series
    """

    # start timing 
    time_begin = time.time()

    # store the results
    result_sequence: list[np.ndarray] = []

    # function to handle processing of destretch loop results for each image
    # frame found in data files
    def iter_process(result: DestretchLoopResult):
        # save the processed image in the results array
        result_sequence.append(result[0])

    # iterate through file datas and call iter_process on destretched results
    fits_file_process_iter(filepaths, iter_process, ** kwargs)

    # output time elapsed
    time_end = time.time()
    print(f"Time elapsed: {time_end - time_begin} seconds")

    return result_sequence

def calc_offset_vectors(
        in_filepaths: list[os.PathLike],
        out_dir: os.PathLike,
        out_filename: str = "offsets",
        ** kwargs: IterProcessArgs
    ) -> None:
    """
    calcuates the offset vectors from destretching and outputs each of them as 
    .fits files

    Parameters
    ----------
    in_filepaths:
        the list of fits filepaths to process data from
    out_dir:
        the directory where the offset vector files will be stored
    out_filename:
        the base name of each file (numbers will be appended to the end of them)
    ** kwargs: 
        see IterProcessArgs class for all available arguments
    """

    # number of digits needed to accurately order the output files
    out_name_digits: int = len(str(len(in_filepaths)))
    index: int = 0

    # ensure output directory exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # define the processing to calculate and store the offset files
    def process_iter(result: DestretchLoopResult):
        nonlocal index

        # use final displacement sum 'disp_sum' - 'rdisp_sum'
        _, disp_sum, rdisp_sum, _ = result
        offsets = disp_sum - rdisp_sum
        offsets = IndexSchema.convert(offsets, IndexSchema.TYX, IndexSchema.XYT)

        # output the vectors as a new fits file
        out_num = f"{index:0{out_name_digits}}"
        out_path = os.path.join(out_dir, out_filename + f"{out_num}.off.fits")
        fits.writeto(out_path, offsets, overwrite=True)
        index += 1
    
    # use default previousRef method if not specified
    kwargs["ref_method"] = kwargs.get("ref_method", PreviousRef(in_filepaths))

    # iterate over data in files
    fits_file_process_iter(in_filepaths, process_iter, ** kwargs)

def calc_offset_vector_averages(
        
) -> None:
    pass