import os
import abc
import enum

import numpy as np

import utility

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
        self.cur_data = utility.load_image_data(self.filepaths[current_index])

        # just use current data as reference if first frame
        if self.previous_data is None:
            self.previous_data = self.cur_data
        
        # cycle current data to previous
        reference_data = self.previous_data
        self.previous_data = self.cur_data
        self.cur_data = None

        return reference_data
     
class MarginEndBehavior(enum.Enum):
    KEEP_RANGE: int = 0
    TRIM_MARGINS: int = 1

    def get_margin_range(self, max_range: int, min: int, max: int) -> tuple[int, int]:
        if min < 0 or max > max_range:
            match self:
                case MarginEndBehavior.KEEP_RANGE:
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
                case MarginEndBehavior.TRIM_MARGINS:
                    min = max(0, min)
                    max = min(max_range, max)
        return (min, max)


class OMargin(RefMethod):
    """
    A Reference Method which takes `self.margin_left` preceeding images of the 
    original dataset (with respect to the image currently being processed), and 
    `self.margin_right` of the images after. A composite median image is then 
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

    def __init__(self, filepaths = [], left: int = 5, right: int = 5):
        super().__init__(filepaths)
        self.end_behavior: MarginEndBehavior = MarginEndBehavior.KEEP_RANGE
        self.margin_left: int = left
        self.margin_right: int = right
        self.original_data: list[np.ndarray] = []
        self.original_data_off = 0
        self.input_schema: utility.IndexSchema = utility.IndexSchema.XY

    @staticmethod
    def create(
            filepaths: list[os.PathLike], 
            input_schema: utility.IndexSchema = utility.IndexSchema.XY,
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
        original_image: np.ndarray = None
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
                if i == current_index and original_image is not None:
                    data = original_image
                else:
                    data = utility.load_image_data(self.filepaths[i])
                self.original_data.append(data)

    def get_reference(self, current_index: int) -> np.ndarray:
        return np.median(np.stack(self.original_data, axis=2), axis=2)

    def get_original_data(self, current_index) -> np.ndarray | None:
        local_index = current_index - self.original_data_off
        if len(self.original_data) > local_index:
            return self.original_data[local_index]
        return None
