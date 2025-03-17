import os
import time
from typing import Callable, TypedDict, Any, Union

import numpy as np
from astropy.io import fits

from algorithm import (
    doreg, destr_control_points, reg_loop,
    DestretchLoopResult
)
from utility import IndexSchema, load_image_data
from reference_method import RefMethod, RollingWindow, PreviousRef, WindowEdgeBehavior

## Utility Types ---------------------------------------------------------------

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

## Utility Funcs ---------------------------------------------------------------

def fits_file_destretch_iter(
        in_filepaths: list[os.PathLike],
        iter_func: Callable[[DestretchLoopResult], None],
        ** kwargs: IterProcessArgs
    ) -> None:
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
    ref_method: RefMethod = kwargs.get("ref_method", RollingWindow(in_filepaths))

    # meta info from files
    (_, _, image_resolution) = get_filepaths_info(in_filepaths)

    # reference image to feed into next destretch loop
    reference_image: np.ndarray | None = None

    # iterate through each file
    print("Searching for image data in specified files...")
    for i in range(len(in_filepaths)):
        path = in_filepaths[i]
        
        # get the image data from ref method if available otherwise load it
        image_data: np.ndarray = ref_method.get_original_data(i)
        if image_data is None:
            image_data = load_image_data(path)

        # ensure all data matches the first file's resolution
        if (
            image_resolution[-1] != image_data.shape[-1] or 
            image_resolution[-2] != image_data.shape[-2]
        ):
            print(image_resolution, image_data.shape)
            raise Exception(f"Resolution mismatch for '{os.fspath(path)}'")

        # debug log
        # print(f"found {hdu_index} in {path}: {image_data.shape}")

        # get the reference image from the specified reference method
        reference_image = ref_method.get_reference(i)

        print(f"ref image checksum: {reference_image.sum()}")

        # perform image destretching
        print(f"processing image #{i}.." + in_filepaths[i])
        result = reg_loop(
            image_data,
            reference_image,
            kernel_sizes,
        )

        # ???? why does it change after the first time? why is the change 
        # different than just ref -= ref.mean()? 
        print(f"ref image checksum: {reference_image.sum()}")

        # call the function specified by caller
        iter_func(result)

def fits_file_process_iter(
        in_data_files: list[os.PathLike],
        in_off_files: list[os.PathLike],
        iter_func: Callable[[DestretchLoopResult], None],
        in_avg_files: list[os.PathLike] | None = None,
        ** kwargs: IterProcessArgs
    ) -> None:
    """
    iterates over each data from each specified file and applies destretching
    to each of them, calling the specified iter_func and passing the result of
    destretching to that funciton

    Parameters
    ----------
    in_data_files:
        the list of file paths to process
    iter_func:
        the function that is called for each frame after the image data has 
        been processed it
    """

    # get required kwargs or default values if not specified
    kernel_sizes: list[int] = kwargs.get("kernel_sizes", [64, 32])
    index_schema: IndexSchema = kwargs.get("index_schema", IndexSchema.XYT)

    # ensure there is an offset for each data
    if(len(in_data_files) != len(in_off_files)):
        raise Exception("Each data file must have a corresponging offset")

    # meta info from files
    (_, _, image_resolution) = get_filepaths_info(in_data_files)
    
    # iterate through each file
    print("Searching for image data in specified files...")
    for i in range(len(in_data_files)):
        path_data = in_data_files[i]
        path_off = in_off_files[i]
        path_avg = None if in_avg_files is None else in_avg_files[i]
        
        # get the image data from ref method if available otherwise load it
        image_data = load_image_data(path_data)
        off_data = load_image_data(path_off, z_index=None)
        avg_data = None if path_avg is None else load_image_data(path_avg, z_index=None)

        # ensure all data matches the first file's resolution
        if (
            image_resolution[-1] != image_data.shape[-1] or 
            image_resolution[-2] != image_data.shape[-2] or
            image_resolution[-1] != off_data.shape[-1] or
            image_resolution[-2] != off_data.shape[-2] or
            (False if avg_data is None else image_resolution[-1] != avg_data.shape[-1]) or
            (False if avg_data is None else image_resolution[-2] != avg_data.shape[-2])
        ): 
            print(
                image_resolution, 
                image_data.shape, 
                off_data.shape, 
                avg_data.shape
            )
            raise Exception(
                f"Resolution mismatch for '{os.fspath(path_data), path_avg}'"
            )

        # generate destretch and reference displacement vectors based on offsets that we know 
        # have a 1:1 kernel:pixel scale
        destr_params, rdisp = destr_control_points(
            image_data,
            np.zeros((1,1)), # kernel?
            border_offset=0, # should be default?
            spacing_ratio=0 # should be defailt?
        )

        # invert the offset data and apply it as a correction to destretch the original image
        corrected_off_data = -off_data if avg_data is None else avg_data - off_data 
        result = (
            doreg(
                image_data, 
                rdisp,
                rdisp - corrected_off_data * 2,
                destr_params
            ),
            None,
            None,
            destr_params,
        )

        # call the function specified by caller
        iter_func(result)

def write_sequential_file(
    index: int, 
    digits: int, 
    out_dir: str, 
    base_name: str,
    data: np.ndarray
) -> os.PathLike:
    out_num = f"{index:0{digits}}"
    out_path = os.path.join(
        out_dir, 
        base_name + f"{out_num}.avg.fits"
    )
    fits.writeto(out_path, data, overwrite=True)
    return out_path

def get_filepaths_info(
    in_filepaths: list[os.PathLike]
) -> tuple[int, int, Union[int, tuple[int, ...]]]:
    
    file_count = len(in_filepaths)
    out_name_digits: int = len(str(file_count))
    image_resolution = load_image_data(in_filepaths[-1], z_index=None).shape
    return (file_count, out_name_digits, image_resolution)

## Module Funcitonality --------------------------------------------------------

def destretch_files(
        in_data_files: list[os.PathLike],
        in_off_files: list[os.PathLike],
        out_dir: os.PathLike,
        out_filename: str = "destretched",
        in_avg_files: list[os.PathLike] | None = None,
        ** kwargs: IterProcessArgs
    ) -> list[os.PathLike]:
    """
    Compute the destretched result of data from all given files, and export to 
    new files

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
    """

    out_paths: list[os.PathLike] = []

    # start timing 
    time_begin = time.time()

    # number of digits needed to accurately order the output files
    out_name_digits: int = len(str(len(in_data_files)))
    index: int = 0

    # ensure output directory exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # function to handle processing of destretch loop results for each image
    # frame found in data files
    def iter_process(result: DestretchLoopResult):
        nonlocal index

        # output the result as a new fits file
        out_num = f"{index:0{out_name_digits}}"
        out_path = os.path.join(out_dir, out_filename + f"{out_num}.fits")
        fits.writeto(out_path, result[0], overwrite=True)
        out_paths.append(out_path)
        print(f"destretched {out_path}")
        index += 1

    # iterate through file datas and call iter_process on destretched results
    fits_file_process_iter(
        in_data_files, 
        in_off_files, 
        iter_process, 
        in_avg_files=in_avg_files, 
        ** kwargs
    )

    # calculate time in appropriate units
    time_end = time.time()
    time_taken = time_end - time_begin
    time_units = "seconds"
    if time_taken >= 120:
        time_taken /= 60
        time_units = "minutes"
        if time_taken >= 120:
            time_taken /= 60
            time_units = "hours"
    
    # output time
    print(
        "Destretching complete! Time elapsed: " +
        f"{time_taken:.3f} {time_units}"
    )
    return out_paths

def calc_offset_vectors(
        in_filepaths: list[os.PathLike],
        out_dir: os.PathLike,
        out_filename: str = "offsets",
        ** kwargs: IterProcessArgs
    ) -> list[os.PathLike]:
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

    out_paths: list[os.PathLike] = []

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

        # output the vectors as a new fits file
        out_num = f"{index:0{out_name_digits}}"
        out_path = os.path.join(out_dir, out_filename + f"{out_num}.off.fits")
        fits.writeto(out_path, offsets, overwrite=True)
        out_paths.append(out_path)
        index += 1
    
    # use default previousRef method if not specified
    kwargs["ref_method"] = kwargs.get("ref_method", PreviousRef(in_filepaths))

    # iterate over data in files
    fits_file_destretch_iter(in_filepaths, process_iter, ** kwargs)
    return out_paths

def calc_rolling_mean(
        in_filepaths: list[os.PathLike],
        out_dir: os.PathLike,
        out_filename: str = "avg",
        window_left: int = 5,
        window_right: int = 5,
        end_behavior: WindowEdgeBehavior = WindowEdgeBehavior.KEEP_RANGE,
        ** kwargs: IterProcessArgs
    )-> list[os.PathLike]:
    """
    TODO this function needs to be revised, end behavior of trim margin seems 
    to break it

    calculate a rolling mean for the given input filepaths (which should be 
    files generated by the calc_offset_vectors or similar function), and output
    a corresponding mean matrix with the specified range, as a fits file in the 
    specified output directory

    Paramaters
    ----------
    in_filepaths:
        the list of fits filepaths to process data from
    out_dir:
        the directory where the offset vector files will be stored
    out_filename:
        the base name of each file (numbers will be appended to the end of them)
    window_left:
        number of images before current image to include in ref margin
    window_right:
        number of images after current image to include in ref margin
    end_behavior:
        how the function decides what data to use for the margin when it 
        collides with an edge of the data list
    ** kwargs: 
        see IterProcessArgs class for all available arguments
    """

    # store output paths to return
    out_paths: list[os.PathLike] = []

    # ensure output directory exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # data structures to store info about original datas
    file_count = len(in_filepaths)
    original_data: list[np.ndarray] = []
    avg_data: list[np.ndarray] = []
    original_data_off: int = 0

    # number of digits needed to accurately order the output files
    out_name_digits: int = len(str(file_count))
    index_avged: int = 0
    index_written: int = 0

    # track the data average from the previous iterations
    data_avg: np.ndarray | None = None
    data_avg_start: int = 0
    data_avg_end: int = 0
    data_avg_range: int = 0

    # define the local function that iterates over each data frame
    iter_count: int = file_count + window_right
    for i0 in range(iter_count):
        path: str | None = None if i0 >= file_count else in_filepaths[i0]
        data: np.ndarray | None = None if path is None else load_image_data(path, z_index=None)

        # calculate the local margin values
        margin_min, margin_max = end_behavior.clamp(
            file_count,
            index_avged - window_left, 
            index_avged + window_right + 1
        )
        margin_range: int = margin_max - margin_min
        
        # append current data to data list
        if data is not None:
            if i0 - original_data_off >= len(original_data):
                original_data.append(data)
        
        # if the entire margin is within the data list
        local_marg_max = margin_max - original_data_off
        if local_marg_max <= len(original_data):
            local_marg_min = margin_min - original_data_off

            # calculate the average from scratch if it doesn't exist yet
            if data_avg is None:
                data_avg = original_data[local_marg_min]
                for i1 in range(local_marg_min + 1, local_marg_max):
                    data_avg += original_data[i1].copy()
                data_avg /= margin_range
            
            # if it does exist, remove the preceding datas, and add the datas 
            # that need to be added
            else:
                # remove all preceding datas from average
                local_avg_start = data_avg_start - original_data_off
                for i1 in range(local_avg_start, local_marg_min):
                    data_avg -= original_data[i1] / data_avg_range

                # adjust current avg weight to new range if changed
                local_avg_end = data_avg_end - original_data_off
                if data_avg_range != margin_range:
                    avg_weight_prev_numerator = data_avg_range - (
                        local_avg_start -
                        local_marg_min
                    )
                    avg_weight_numerator = margin_range - (
                        local_marg_max - 
                        local_avg_end
                    )
                    data_avg /= avg_weight_prev_numerator / data_avg_range
                    data_avg *= avg_weight_numerator

                # add new datas to average
                for i1 in range(local_avg_end, local_marg_max):
                    data_avg += original_data[i1] / margin_range

            # remove unneeded datas from beginning of data list
            for _ in range(local_marg_min):
                original_data.pop(0)
                original_data_off += 1
            
            # update avg metadata
            data_avg_start = margin_min
            data_avg_end = margin_max
            data_avg_range = margin_range
            avg_data.append(data_avg.copy())
            index_avged += 1

        # output the averaged vectors as a new fits file
        while len(avg_data) > 0:
            out_num = f"{index_written:0{out_name_digits}}"
            out_path = os.path.join(
                out_dir, 
                out_filename + f"{out_num}.fits"
            )
            fits.writeto(out_path, avg_data[0], overwrite=True)
            out_paths.append(out_path)
            index_written += 1

            # pop data if not last iteration, so we rewrite the same file
            # multiple times at the end according to margin settings
            if not(i0 == iter_count - 1 and index_written <= i0 - window_right):
                avg_data.pop(0)

    return out_paths

def calc_cumulative_sums(
    in_filepaths: list[os.PathLike],
    out_dir: os.PathLike,
    out_filename: str = "cumulative_off"
) -> list[os.PathLike]:
    
    # meta info from files
    (file_count, out_name_digits, image_resolution) = get_filepaths_info(in_filepaths)
    paths: list[os.PathLike] = []

    # ensure output directory exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # iterate from 0 to index at each iteration, and sum all data from the start to nth iteration
    data_sum = np.zeros(image_resolution)
    for index0 in range(file_count):

        # sum all data from 0 to iteration by adding it to our last data_sum
        data = load_image_data(in_filepaths[index0], z_index=None)
        data_sum += data
        
        # write file and append path to output
        path = write_sequential_file(index0, out_name_digits, out_dir, out_filename, data_sum)
        paths.append(path)

    return paths

def calc_difs(
    in_filepaths1: list[os.PathLike],
    in_filepaths2: list[os.PathLike],
    out_dir: os.PathLike,
    out_filename: str = "dif"
) -> list[os.PathLike]:
    """
    Subtract the data in in_filepaths2 from the data in in_filepaths1, store 
    results at specified directory and filename
    """
    
    # meta info from files
    (file_count, out_name_digits, image_resolution) = get_filepaths_info(in_filepaths1)
    paths: list[os.PathLike] = []

    # ensure output directory exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for index in range(file_count):
        data_a = load_image_data(in_filepaths1[index], z_index=None)
        data_b = load_image_data(in_filepaths2[index], z_index=None)
        write_sequential_file(index, out_name_digits, out_dir, out_filename, data_a - data_b)