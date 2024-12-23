import enum

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
