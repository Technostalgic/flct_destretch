import os
import abc
import enum

import astropy.io.fits as fits

import numpy as np

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

class RefMethod(abc.ABC):
    """
    Data structure to define how the reference image in the destretching 
    algorithm is defined or calculated - abstract class
    """
    def __init__(self, filepaths: list[os.PathLike] = []):
        """
        Do not use this - instead use the static 'create' method

        Parameters
        ----------
        filepaths : list[os.PathLike]
            list of paths to be destretched
        """
        super().__init__()
        self.filepaths: list[os.PathLike] = filepaths

    @abc.abstractmethod
    def get_reference(
        current_index: int
    ) -> np.ndarray:
        """
        returns the image that should be used as the reference for calculating
        the destretched result from an original data image

        Parameters
        ----------
        current_index: int
            the index of the current image that is to be processed
        """
        pass

class MarginEndBehavior(enum.Enum):
    KEEP_RANGE: int = 0
    TRIM_MARGINS: int = 1

class OMargin(RefMethod):
    """
    A Reference Method which takes `self.margin_left` preceeding images of the 
    original dataset (with respect to the image currently being processed), and 
    `self.margin_right` of the images after. A composite median image is then 
    created from those images and returned as a reference image
    """

    def __init__(self, filepaths = []):
        super().__init__(filepaths)
        self.end_behavior: MarginEndBehavior = MarginEndBehavior.KEEP_RANGE
        self.margin_left: int = 1
        self.margin_right: int = 1
        self.original_data: list[np.ndarray] = []
        self.original_data_off = 0
        self.input_schema: IndexSchema = IndexSchema.XY

    @staticmethod
    def create(
            filepaths: list[os.PathLike], 
            input_schema: IndexSchema = IndexSchema.XY,
            margin_left:int = 1, 
            margin_right: int = 1,
        ) -> 'OMargin':
        """
        create a new instance of the OMargin reference method

        Parameters
        ----------
        filepaths : list[os.PathLike]
            list of paths to be destretched
        margin_left : int
            how many original images before the current image should be included
        margin_left : int
            how many original images after the current image should be included
        """
        result: OMargin = OMargin(filepaths)
        result.input_schema = input_schema
        result.margin_left = margin_left
        result.margin_right = margin_right
        return result

    def pass_params(self,
        current_index: int,
        original_image: np.ndarray
    ):
        # calculate the local margin values
        margin_min = current_index - self.margin_left
        margin_max = current_index + self.margin_right + 1
        file_count = len(self.filepaths)
        if margin_min < 0 or margin_max > file_count:
            match self.end_behavior:
                case MarginEndBehavior.KEEP_RANGE:
                    margin_range = margin_max - margin_min
                    if file_count < margin_range:
                        margin_min = 0
                        margin_max = file_count
                    elif margin_min < 0:
                        margin_min = 0
                        margin_max = margin_range
                    elif margin_max > file_count:
                        margin_max = file_count
                        margin_min = file_count - margin_range
                        pass
                case MarginEndBehavior.TRIM_MARGINS:
                    margin_min = max(0, margin_min)
                    margin_max = min(file_count, margin_max)
                    pass
        
        # remove image datas who are no longer needed
        off_increment = margin_min - self.original_data_off
        self.original_data_off = margin_min
        for _ in range(off_increment):
            self.original_data.pop(0)

        # iterate through each index in the margin
        for i in range(margin_min, margin_max):
            local_index = i - self.original_data_off
            if local_index >= len(self.original_data):
                data: np.ndarray = None
                if i == current_index:
                    data = original_image
                else:
                    # TODO
                    pass
                self.original_data.append(data)

    def get_reference(
        current_index: int
    ) -> np.ndarray:
        pass