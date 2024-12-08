import enum
from typing import Self

import numpy as np
import astropy.io.fits as fits
from astropy.io.fits.hdu import HDUList, ImageHDU, CompImageHDU

from algorithm import reg_loop, reg_loop_series
from destretch_params import DestretchParams

class IndexSchema(enum.Enum):
    """
    Represents an index schema for an np.ndarray, showing which index in the 
    array corresponds to which axis.\n
    X - Spatial X axis, \n
    Y - Spatial Y axis, \n
    T - Temporal time axis. \n
    For example, an IndexSchema.XYT represents an 3 dimensional array where
    the 1st index is X, 2nd index is Y, and the 3rd index is T, accessed by 
    `my_array[x_coord, y_coord, timestamp]`
    """
    XY = 0
    YX = 1
    XYT = 0
    YXT = 1
    TXY = 2
    TYX = 3
    XTY = 4
    YTX = 5

    # Axis permutations for each schema
    __SCHEMA_TO_AXES = {
        XYT: (0, 1, 2),
        YXT: (1, 0, 2),
        TXY: (2, 0, 1),
        TYX: (2, 1, 0),
        XTY: (0, 2, 1),
        YTX: (1, 2, 0),
    }

    @staticmethod
    def convert(
            input: np.ndarray, 
            from_schema: 'IndexSchema', 
            to_schema: 'IndexSchema'
        ) -> np.ndarray:

        # don't need to do anything if the schemas are the same
        if from_schema == to_schema: return input

        # Get the axis order for each schema
        from_axes = IndexSchema.__SCHEMA_TO_AXES[from_schema]
        to_axes = IndexSchema.__SCHEMA_TO_AXES[to_schema]

        # ensure it works for 2d arrays
        if len(input.shape) == 2:
            if to_axes > 1 or from_axes > 1:
                raise Exception(
                    "2D arrays can only be converted with 2d " +
                    "Schemas (IndexSchema.XY, IndexSchema.YX)"
                )
            from_axes = (from_axes[0], from_axes[1])
            to_axes = (to_axes[0], to_axes[1])

        # Create the permutation to transform from from_axes to to_axes
        permute_order = [from_axes.index(axis) for axis in to_axes]

        # Transpose the array accordingly
        return np.transpose(input, axes=permute_order)

def destretch_files(
        filepaths: list[str], 
        kernel_sizes: list[int], 
        index_schema: IndexSchema
    ) -> list[
        np.ndarray
    ]:
    """
    Compute the destretched result of data from all given files

    Parameters
    ----------
    filepaths : list[str]
        list of files in order to treat as image sequence for destretching
    kernel_sizes : ndarray (kx, ky, 2)
        the destretch kernel size, the size of each subframe to be processed
    index_schema : IndexSchema
        the index schema to use for the input data files

    Returns
    -------
    result : tuple[np.ndarray, np.ndarray, np.ndarray, DestretchParams]
        same output as reg_loop_series
    """

    # used to enforce the same resolution for all data files
    image_resolution = (-1,-1)

    # store the results
    result_sequence: list[np.ndarray] = []

    # reference image to feed into next destretch loop
    reference_image: np.ndarray | None = None

    # read each image HDU in each file
    print("Searching for image data in specified files...")
    for path in filepaths:
        hdus: HDUList = fits.open(path)
        for hdu_index in range(len(hdus)): 
            hdu = hdus[hdu_index]
            if hdu.data is not None or hdu is ImageHDU or hdu is CompImageHDU:

                # convert all data to default index schema
                image_data: np.ndarray = IndexSchema.convert(
                    hdu.data, 
                    index_schema, 
                    IndexSchema.XYT
                )

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
                    ): continue

                    # TODO seperate image data by z axis
                    # image_data = np.moveaxis(image_data, 0, -1)[:,:]

                    # debug log
                    print(f"found {hdu_index} in {path}: {image_data.shape}")

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
                    print(f"processed image #{image_num}")

                    # store previously processed frame as reference image to use
                    # for nex image in series
                    reference_image = result_sequence[-1]

    return result_sequence
