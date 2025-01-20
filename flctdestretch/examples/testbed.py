## Imports and Initialization -------------------------------------------------|
print("Importing dependencies... ")

import os, os.path
import re

# third party
import numpy as np

# internal
import abstraction
from utility import IndexSchema, get_fits_paths
from examples.fits_to_mp4 import fits_to_mp4
from reference_method import OMargin
import algorithm as destretch


## Fetch Data ------------------------------------------------------------------

files_dir = os.path.join(
    ".", "examples", "media", "large"
)
files_regex = re.compile(
    "^whitelight\.cln\.\d{8}_\d{6}\.im\d{5}\.seq\d{3}\.ext\d{3}\.fits$"
    #"^test_1k_\d{2}\.fits$"
)
files = [
    os.path.join(files_dir, filename)
    for filename in os.listdir(files_dir)
    if files_regex.match(filename)
][610:620]
print(f"{len(files)} files found")


## Perform Destretching -------------------------------------------------------|

print("Destretching images... ")
kernel_sizes: np.ndarray[np.int64] = np.array([64, 32])

# calculate offset vectors
offs_out_dir = os.path.join(files_dir, "off")
print(f"calculating offsets... {offs_out_dir}")
abstraction.calc_offset_vectors(
    files,
    offs_out_dir,
    "offset",
    kernel_sizes=kernel_sizes,
    IndexSchema=IndexSchema.XY
)

# calculate rolling mean
out_dir = os.path.join(files_dir, "avg")
print(f"calculating offset rolling mean... {out_dir}")
abstraction.calc_rolling_mean(
    get_fits_paths(offs_out_dir),
    out_dir,
    "average"
)

# do actual destretch
out_dir = os.path.join(files_dir, "destretched")
result = abstraction.destretch_files(
    files,
    out_dir,
    kernel_sizes=kernel_sizes,
    index_schema=IndexSchema.XY,
    ref_method=OMargin(files, 10, 10)
)

# output results as video files
out_file_orig_vid = os.path.join(files_dir, "video_original.mp4")
out_file_destr_vid = os.path.join(files_dir, "video_destretched.mp4")
fits_to_mp4(files, out_file_orig_vid, 60, "copper", IndexSchema.XY, 0.2, 1.25)
fits_to_mp4(out_dir, out_file_destr_vid,60, "copper", IndexSchema.XY, 0.2, 1.25)

print(
    "Demo Complete! output videos at:\n"+
    f"{out_file_orig_vid} \n{out_file_destr_vid}"
)