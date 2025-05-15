## Imports and Initialization -------------------------------------------------|
print("Importing dependencies... ")

import sys, time
import os, os.path
sys.path.append(os.path.abspath(".."))  # Adjust path to package root

# third party
import numpy as np

# internal
from ..abstraction import *
from ..utility import get_fits_paths

## Fetch Data ------------------------------------------------------------------

files_dir = os.path.abspath(os.path.join(".", "flctdestretch", "examples", "media"))
files = get_fits_paths(files_dir)
files = files[0:10]
# print("\n".join(files))
print(f"{len(files)} files found")

# variables for destretching config
kernel_sizes: np.ndarray[np.int64] = np.array([128, 64])
rolling_mean_window_size: int = 5
flowmap_window_size: int = 5

# in case it fails partway through destretching, you can set this to start at 
# the nth file, where n is the index of the last file that was successfully 
# destretched. naming conventions with offset indices will remain consistent 
# with previously destretched files, so you can always pick up where you left off.
start_index: int = 0 

# filepaths for output files
out_off_dir = os.path.join(files_dir, "off")
out_sum_dir = os.path.join(files_dir, "sum")
out_rolling_sum_dir = os.path.join(files_dir, "rolling_sum")
out_off_control_dir = os.path.join(files_dir, "off_control")
out_avg_dir = os.path.join(files_dir, "avg")
out_flow_dir = os.path.join(files_dir, "flow")
out_dir = os.path.join(files_dir, "destretched")
out_dir_control = os.path.join(files_dir, "destretched_control")
out_off_final_dir = os.path.join(files_dir, "off_final")
print(f"beginning destretch, end result will be output to {out_dir}")

# begin timer
start = time.time()

# calculate offset vectors
print(f"calculating offsets... {out_off_dir}")
calc_offset_vectors(
	files,
	out_off_dir,
	"off",
	kernel_sizes=kernel_sizes,
    start_at=start_index,
)

# calculate the cumulative sum offsets
print(f"calculating cumulative sums... {out_sum_dir}")
calc_cumulative_sums(
    get_fits_paths(out_off_dir),
    out_sum_dir,
    start_at=start_index,
)

# calculate the rolling sums of the cumulative sum offsets
print(f"calculating rolling mean... {out_rolling_sum_dir}")
calc_rolling_mean(
    get_fits_paths(out_sum_dir),
    out_rolling_sum_dir,
    window_left=rolling_mean_window_size,
    window_right=rolling_mean_window_size,
    start_at=start_index,
)

# calculate the difs
print(f"calculating final offs... {out_off_final_dir}")
calc_difs(
    get_fits_paths(out_sum_dir),
    get_fits_paths(out_rolling_sum_dir),
    out_off_final_dir,
    "final",
    start_at=start_index,
)

# calculate the flowmap
print(f"generating flow map data... {out_flow_dir}")
calc_change_rate(
    get_fits_paths(out_rolling_sum_dir),
    out_flow_dir,
    window_size=flowmap_window_size,
    start_at=start_index,
)

# apply the final offset vectors to destretch the image data
print(f"generating destretched image data... {out_dir}")
result = destretch_files(
	files,
	get_fits_paths(out_off_final_dir),
	out_dir,
	"destr",
	#in_avg_files=get_fits_paths(out_avg_dir),
    start_at=start_index,
)


elapsed = time.time() - start
print(f"Demo complete! \nTotal elapsed time: {elapsed}")