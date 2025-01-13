import os
import re
import enum

import numpy as np
import astropy.io.fits as fits
from astropy.io.fits.hdu import HDUList, ImageHDU, CompImageHDU

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
        """
        Create and return a new window for the data array to rearrange the 
        indices based on the specified schemas.

        Parameters
        ----------
        input: np.ndarray
            the array to rearrange the indices of
        from_schema: IndexSchema
            the scheme that the array is currently
        to_schema: IndexSchema
            the schema you wish to change it to
        """

        # don't need to do anything if the schemas are the same
        if from_schema == to_schema: return input

        # Get the axis order for each schema
        from_axes = IndexSchema.__SCHEMA_TO_AXES[from_schema.value]
        to_axes = IndexSchema.__SCHEMA_TO_AXES[to_schema.value]

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

def load_image_data(
        path: os.PathLike, 
        schema: IndexSchema = IndexSchema.XY,
        hdu_index: int | None = None,
        z_index: int = 0
    ) -> np.ndarray:
    """
    extract image data from a file in the form of an XY np.ndarray

    Parameters
    ----------
    path : os.PathLike
        the path to the file which we want to extract the image data from
    schema : IndexSchema
        the index schema of the input data
    hdu_index : int
        which hdu to select from the file. Will select the first valid hdu if
        none specified
    z_index : int
        which z-slice to use if the specified data file contains a 3 dimensional 
        data block (after applying the IndexSchema conversion)
    """
    # select the correct hdu by the specified hdu index
    hdus: HDUList = fits.open(path)
    hdu: ImageHDU | CompImageHDU = None
    if hdu_index is None:
        for unit in hdus:
            if (
                unit.data is not None or 
                unit is ImageHDU or 
                unit is CompImageHDU
            ):
                hdu = unit
                break
    else: hdu = hdus[hdu_index]
    
    # transform the image data from the data in the fits file
    image_data: np.ndarray = IndexSchema.convert(
        hdu.data, 
        schema, 
        IndexSchema.XYT
    )
    if len(image_data.shape) == 3:
        image_data = image_data[:, :, z_index]
    
    return image_data

fits_regex = re.compile("^.*\.fits$")

def get_fits_paths(in_dir: os.PathLike) -> list[str]:
    return [
        os.path.join(in_dir, filename)
        for filename in os.listdir(in_dir)
        if fits_regex.match(filename)
    ]