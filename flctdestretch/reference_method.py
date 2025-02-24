import os
import abc
import enum

import numpy as np

from utility import IndexSchema, load_image_data

## Utility Types ---------------------------------------------------------------

class WindowEdgeBehavior(enum.Enum):
    KEEP_RANGE: int = 0
    TRIM_MARGINS: int = 1

    def clamp(self, max_range: int, min: int, max: int) -> tuple[int, int]:
        """
        Clamps (clamp = keep within range) min and max between 0 and max_range
        based on the specified edge behavior. Returns new valuse for min and max
        """
        if min < 0 or max > max_range:
            match self:
                case WindowEdgeBehavior.KEEP_RANGE:
                    margin_range = max - min
                    if max_range < margin_range:
                        min = 0
                        max = max_range
                    elif min < 0:
                        min = 0
                        max = margin_range
                    elif max > max_range:
                        max = max_range
                        min = max_range - margin_range
                case WindowEdgeBehavior.TRIM_MARGINS:
                    min = max(0, min)
                    max = min(max_range, max)
        return (min, max)

## Reference Method Definitions ------------------------------------------------

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
    def get_reference(self, current_index: int) -> np.ndarray:
        """
        returns the image that should be used as the reference for calculating
        the destretched result from an original data image

        Parameters
        ----------
        current_index : int
            the index of the current image that is to be processed
        """
        pass

    def get_original_data(self, current_index: int) -> np.ndarray | None:
        """
        will get the original data from the filepaths if precalculated, 
        otherwise returns None

        Parameters
        ----------
        current_index : int
            the index of the current image that is to be processed
        """
        return None

class PreviousRef(RefMethod):
    """
    A simple reference method, it just uses the previous image as the reference.
    NOTE This should not be used for destretching a final image sequence, as it 
    will not account for distortions in any meaningful way, it's used mainly 
    for preprocessing the data and then performing additional calculations
    """

    def __init__(self, filepaths: list[os.PathLike]):
        super().__init__(filepaths)
        self.previous_data: np.ndarray | None = None
        self.cur_data: np.ndarray | None = None

    def get_reference(self, current_index) -> np.ndarray:

        # get the current frame data
        self.cur_data = load_image_data(self.filepaths[current_index])

        # just use current data as reference if first frame
        if self.previous_data is None:
            self.previous_data = self.cur_data
        
        # cycle current data to previous
        reference_data = self.previous_data
        self.previous_data = self.cur_data
        self.cur_data = None

        return reference_data

class RollingWindow(RefMethod):
    """
    A Reference Method which takes `self.window_left` preceeding images of the 
    original dataset (with respect to the image currently being processed), and 
    `self.window_right` of the images after. A composite median image is then 
    created from those images and returned as a reference image

    Parameters
    ----------
    filepaths:
        the paths to the files being processed,
    left:
        number of images before current image to include in ref margin
    right:
        number of images after current image to include in ref margin
    """

    edge_behavior: WindowEdgeBehavior = WindowEdgeBehavior.KEEP_RANGE
    window_left: int
    window_right: int
    original_data: list[np.ndarray] = []
    original_data_off: int = 0
    input_schema: IndexSchema = IndexSchema.XY

    def __init__(self, filepaths = [], left: int = 5, right: int = 5):
        super().__init__(filepaths)
        self.edge_behavior = WindowEdgeBehavior.KEEP_RANGE
        self.window_left = left
        self.window_right = right

    @staticmethod
    def create(
            filepaths: list[os.PathLike], 
            input_schema: IndexSchema = IndexSchema.XY,
            margin_left:int = 1, 
            margin_right: int = 1,
        ) -> 'RollingWindow':
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
        result: RollingWindow = RollingWindow(filepaths)
        result.input_schema = input_schema
        result.window_left = margin_left
        result.window_right = margin_right
        return result

    def pass_params(self,
        current_index: int,
        original_image: np.ndarray = None
    ):
        # calculate the local margin values
        margin_min = current_index - self.window_left
        margin_max = current_index + self.window_right + 1
        file_count = len(self.filepaths)
        if margin_min < 0 or margin_max > file_count:
            match self.edge_behavior:
                case WindowEdgeBehavior.KEEP_RANGE:
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
                case WindowEdgeBehavior.TRIM_MARGINS:
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
                if i == current_index and original_image is not None:
                    data = original_image
                else:
                    data = load_image_data(self.filepaths[i])
                self.original_data.append(data)

    def get_reference(self, current_index: int) -> np.ndarray:
        return np.median(np.stack(self.original_data, axis=2), axis=2)

    def get_original_data(self, current_index) -> np.ndarray | None:
        local_index = current_index - self.original_data_off
        if len(self.original_data) > local_index:
            return self.original_data[local_index]
        return None
