import os
import cv2
import numpy as np
import math
from typing import Callable


from matplotlib import cm

from utility import IndexSchema, load_image_data

def write_frame(
    data: np.ndarray, 
    video_writer: cv2.VideoWriter, 
    map_rgb: Callable | None, 
    data_range: float, 
    data_min: float, 
    normal_mode: bool = False,
    scale_factor: float = 1.0
) -> bool:

    #ensure data is 2D
    if not normal_mode and (data is None or data.ndim != 2):
        return False
    if normal_mode:
        if data.ndim != 3:
            return False
        if data.shape[2] > 3:
            return False
    
    # normalize the data to 8-bit range (0-255) for visualization
    norm_data = np.clip((data - data_min) / data_range, 0, 1)
    frame: np.ndarray | None = None

    # encode each image dimension as normal map colors if specified
    if normal_mode:

        # separate rgb channels by data z slice
        shape = norm_data.shape
        r = norm_data[:,:,0]
        g = (
            np.zeros((shape[0], shape[1]))
                if shape[2] <= 1 else 
            norm_data[:,:,1] 
        )
        b = (
            np.zeros((shape[0], shape[1]))
                if shape[2] <= 2 else 
            norm_data[:,:,2] 
        )
        
        # write rgb to frame
        rgb = np.zeros((shape[0], shape[1], 3))
        rgb[:,:,0] = r
        rgb[:,:,1] = g
        rgb[:,:,2] = b
        frame = (
            IndexSchema.convert(rgb, IndexSchema.XYT, IndexSchema.XYT) * 
            255.0
        ).astype(np.uint8)
    else:
        frame = (map_rgb(norm_data) * 255.0).astype(np.uint8)
    
    # convert to BGR for video
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Write the frame to the video
    video_writer.write(frame_bgr)
    return True

def data_to_mp4(
    datas: list[np.ndarray],
    out_path: os.PathLike,
    fps: float = 24.0,
    color_map: str = "copper",
    relative_min: float = 0,
    relative_max: float = 1,
    normal_mode: bool = False,
) -> None:

    # grab a test frame to initialize some data
    test_data = datas[-1]

    # Initialize video writer
    height, width = test_data.shape[:2]
    video_writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )
    frames_written = 0

    # get range of last frame and use it for the rest of the frames
    data_min = np.min(test_data)
    data_max = np.max(test_data)
    data_range = (data_max - data_min)
    data_min += data_range * relative_min
    data_max -= data_range * (1 - relative_max)
    data_range *= (relative_max - relative_min)

    # get the colormap mapping function
    map = cm.get_cmap(color_map)

    # write each data as a frame in the video
    for data in datas:
        written = write_frame(data, video_writer, map, data_range, data_min, normal_mode)
        if written: frames_written += 1

    # Release the video writer to save the video
    if frames_written > 0:
        video_writer.release()
        print(f"Video saved to {out_path}, with {frames_written} frames")
    else:
        print("No valid frames to create a video.")

def fits_to_mp4(
    in_files: list[os.PathLike], 
    out_path: os.PathLike,
    fps: float = 24.0,
    color_map: str = "copper",
    index_schema: IndexSchema = IndexSchema.TYX,
    relative_min: float = 0,
    relative_max: float = 1,
    normal_mode: bool = False,
    z_index: int | None = None
) -> None:
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
    process : Callable
        a funcition that is called that allows the passed in data to be modified
        before it is rendered as a frame in the video
    normal_mode : bool
        whether or not the image data is treated as a multidimensional array for 
        each pixel, if so the frames are rendered as a normal map, using each z
        slice as a seperate R or G or B channel
    """

    # grab a test frame to initialize some data
    test_data = IndexSchema.convert(
        load_image_data(in_files[math.floor(len(in_files) * .5)], z_index=z_index), 
        index_schema,
        index_schema.XYT
    )

    # Initialize video writer
    height, width = test_data.shape[:2]
    video_writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )
    frames_written = 0

    # get range of last frame and use it for the rest of the frames
    data_min = np.min(test_data)
    data_max = np.max(test_data)
    data_range = (data_max - data_min)
    data_min += data_range * relative_min
    data_max -= data_range * (1 - relative_max)
    data_range *= (relative_max - relative_min)

    # get the colormap mapping function
    map = cm.get_cmap(color_map)

    for path in in_files:

        # Open the FITS file and extract image data
        data = load_image_data(path, z_index=z_index)
        data = IndexSchema.convert(data, index_schema, index_schema.XYT)

        written = write_frame(data, video_writer, map, data_range, data_min, normal_mode)
        if written: frames_written += 1

    # Release the video writer
    if video_writer is not None:
        video_writer.release()
        print(f"Video saved to {out_path}, with {frames_written} frames")
    else:
        print("No valid frames to create a video.")