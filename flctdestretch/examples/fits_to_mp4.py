import os
import cv2
import numpy as np
from typing import Callable

from astropy.io import fits
from matplotlib import cm

from utility import IndexSchema, load_image_data

def fits_to_mp4(
        in_dir_path: os.PathLike | list[os.PathLike], 
        out_path: os.PathLike,
        fps: float = 24.0,
        color_map: str = "copper",
        index_schema: IndexSchema = IndexSchema.TYX,
        relative_min: float = 0,
        relative_max: float = 1,
        normal_mode: bool = False,
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
    process : Callable
        a funcition that is called that allows the passed in data to be modified
        before it is rendered as a frame in the video
    normal_mode : bool
        whether or not the image data is treated as a multidimensional array for 
        each pixel, if so the frames are rendered as a normal map, using each z
        slice as a seperate R or G or B channel
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
    frames_written = 0

    # store data range to normalize data
    data_min: float | None = None
    data_max: float | None = None
    data_range: float | None = None

    # get the colormap mapping function
    map = cm.get_cmap(color_map)

    for path in fits_files:

        # Open the FITS file and extract image data
        data = load_image_data(path, z_index=None)
        data = IndexSchema.convert(data, index_schema, index_schema.XYT)

        ## TODO fix offsets increasing magnitude along y axis
        # if pixel_adjust_y:
        #     print("adjusting for pixel offsets... " + str(frames_written))
        #     sub = utility.IndexSchema.convert(np.transpose(np.indices((data.shape[0], data.shape[1], data.shape[2]))[0], (2,0,1)), utility.IndexSchema.TYX, utility.IndexSchema.YXT)
        #     data -= sub

        # ensure data is 2D
        if not normal_mode and (data is None or data.ndim != 2):
            print(f"Skipping {path}: Data is not 2D.")
            continue
        if normal_mode:
            if data.ndim != 3:
                print(
                    f"Skipping {path}: " +
                    "normal_mode on, but data has dimensions not equal to 3"
                )
                continue
            if data.shape[0] > 3 :
                print(
                    f"Skipping {path}: " +
                    "Data has to manny z levels."
                )
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
        norm_data = np.clip((data - data_min) / data_range, 0, 1)
        frame: np.ndarray | None = None

        # encode each image dimension as normal map colors if specified
        if normal_mode:

            # separate rgb channels by data z slice
            shape = norm_data.shape
            r = norm_data[0,:,:]
            g = (
                np.zeros((shape[-2], shape[-1]))
                    if shape[0] <= 1 else 
                norm_data[1,:,:] 
            )
            b = (
                np.zeros((shape[-2], shape[-1]))
                    if shape[0] <= 2 else 
                norm_data[2,:,:] 
            )
            
            # if no b channel, use it to show magnitude
            # if shape[0] <= 2:
            #     b = (r ** 2 + b ** 2) ** .5
            
            # write rgb to frame
            rgb = np.zeros((3, shape[-2], shape[-1]))
            rgb[0,:,:] = r
            rgb[1,:,:] = g
            rgb[2,:,:] = b
            frame = (
                IndexSchema.convert(rgb, IndexSchema.TXY, IndexSchema.XYT) * 
                255.0
            ).astype(np.uint8)
            # print(data[frames_written * 90, 0, 1])

        # convert to BGR for video
        else:
            frame = (map(norm_data) * 255.0).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

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
        # cv2.imwrite(out_path + f"frame_{frames_written}.png", frame_bgr)
        video_writer.write(frame_bgr)
        frames_written += 1

    # Release the video writer
    if video_writer is not None:
        video_writer.release()
        print(f"Video saved to {out_path}, with {frames_written} frames")
    else:
        print("No valid frames to create a video.")