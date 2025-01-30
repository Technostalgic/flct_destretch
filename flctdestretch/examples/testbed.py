## Imports and Initialization -------------------------------------------------|
print("Importing dependencies... ")

import os, os.path
import re

# third party
import numpy as np
from astropy.io import fits

# internal
import abstraction
from utility import IndexSchema, get_fits_paths
from examples.fits_to_mp4 import fits_to_mp4
from reference_method import OMargin

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
][610:722]
print(f"{len(files)} files found")


## Perform Destretching -------------------------------------------------------|

print("Destretching images... ")
kernel_sizes: np.ndarray[np.int64] = np.array([64, 32])

offs_out_dir = os.path.join(files_dir, "off")
out_avg_dir = os.path.join(files_dir, "avg")
out_dir = os.path.join(files_dir, "destretched")

# # calculate offset vectors
# print(f"calculating offsets... {offs_out_dir}")
# abstraction.calc_offset_vectors(
#     files,
#     offs_out_dir,
#     "offset",
#     kernel_sizes=kernel_sizes,
#     IndexSchema=IndexSchema.XY
# )
# 
# # calculate rolling mean
# print(f"calculating offset rolling mean... {out_avg_dir}")
# abstraction.calc_rolling_mean(
#     get_fits_paths(offs_out_dir),
#     out_avg_dir,
#     "average"
# )
# 
# # do actual destretch
# result = abstraction.destretch_files(
#     files,
#     out_dir,
#     kernel_sizes=kernel_sizes,
#     index_schema=IndexSchema.XY,
#     ref_method=OMargin(files, 10, 10)
# )

# output results as video files
out_file_orig_vid = os.path.join(files_dir, "video_original.mp4")
fits_to_mp4(files, out_file_orig_vid, 60, "copper", IndexSchema.XY, 0.2, 1.25)

out_file_destr_vid = os.path.join(files_dir, "video_destretched.mp4")
fits_to_mp4(out_dir, out_file_destr_vid, 60, "copper", IndexSchema.XY, 0.2, 1.25)

out_file_flow_vid = os.path.join(files_dir, "video_flowmap.mp4")
fits_to_mp4(out_avg_dir, out_file_flow_vid, 60, "copper", IndexSchema.XY, 0.2, 1.25, True)

out_file_off_vid = os.path.join(files_dir, "video_offmap.mp4")
fits_to_mp4(offs_out_dir, out_file_off_vid, 60, "copper", IndexSchema.XY, 0.2, 1.25, True)

print(
    "Demo Complete! output videos at:\n" +
    f"{out_file_orig_vid} \n{out_file_destr_vid} \n{out_file_flow_vid}"
)