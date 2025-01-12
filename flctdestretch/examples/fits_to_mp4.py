import os
import cv2
import numpy as np
import utility
from astropy.io import fits

def fits_to_mp4(
        in_dir_path: os.PathLike | list[os.PathLike], 
        out_path: os.PathLike,
        fps: float = 24.0,
        index_schema: utility.IndexSchema = utility.IndexSchema.XY,
        relative_min: float = 0,
        relative_max: float = 1,
    ):
    """
    convert a directory of .fits files into an mp4 video

    Parameters
    ----------
    in_dir_path : os.PathLike
        Path to the directory containing .fits files, or a list of filepaths to
        fits files to be used as frames
    out_path : os.PathLike
        Path to save the output video
    fps : float
        Frame rate of the output video
    relative_min : float
        normalized min value to adjust the pixel brightness to for each data 
        point, between 0 and 1 - 0 being the lowest value
    relative_max : float
        normalized min value to adjust the pixel brightness to for each data 
        point, between 0 and 1 - 1 being the highest value
        
    """
    # Get a sorted list of all .fits files in the directory
    fits_files: list[str]
    if not hasattr(in_dir_path, "append"):
        fits_files = sorted([
            os.path.join(in_dir_path, f)
            for f in os.listdir(in_dir_path) 
            if f.lower().endswith('.fits')
        ])
    else: fits_files = in_dir_path

    if not fits_files:
        print("No .fits files found in the directory.")
        return

    # Initialize variables for video dimensions
    video_writer = None

    # store data range to normalize data
    data_min: float | None = None
    data_max: float | None = None
    data_range: float | None = None

    for path in fits_files:

        # Open the FITS file and extract image data
        data = utility.load_image_data(path, index_schema)

        # ensure data is 2D
        if data is None or data.ndim != 2:
            print(f"Skipping {path}: Data is not 2D.")
            continue

        # get range of first frame and use it for the rest of the frames
        if data_min is None:
            data_min = np.min(data)
            data_max = np.max(data)
            data_range = (data_max - data_min)
            data_min += data_range * relative_min
            data_max -= data_range * (1 - relative_max)
            data_range *= (relative_max - relative_min)
        
        # normalize the data to 8-bit range (0-255) for visualization
        thresh_data = np.clip((data - data_min) / data_range, 0, 1)
        norm_data = (thresh_data * 255.0).astype(np.uint8)
        frame = norm_data.astype(np.uint8)

        # convert grayscale to BGR for video
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        # initialize video writer if not already initialized
        if video_writer is None:
            height, width = frame_bgr.shape[:2]
            video_writer = cv2.VideoWriter(
                out_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height)
            )

        # Write the frame to the video
        video_writer.write(frame_bgr)

    # Release the video writer
    if video_writer is not None:
        video_writer.release()
        print(f"Video saved to {out_path}")
    else:
        print("No valid frames to create a video.")