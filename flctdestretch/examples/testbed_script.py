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


# create variable paths for destretching files

kernel_sizes: np.ndarray[np.int64] = np.array([128, 64])
out_off_dir = os.path.join(files_dir, "off")
out_sum_dir = os.path.join(files_dir, "sum")
out_rolling_sum_dir = os.path.join(files_dir, "rolling_sum")
out_off_control_dir = os.path.join(files_dir, "off_control")
out_avg_dir = os.path.join(files_dir, "avg")
out_ref_dir = os.path.join(files_dir, "ref")
out_flow_dir = os.path.join(files_dir, "flow")
out_dir = os.path.join(files_dir, "destretched")
out_dir_control = os.path.join(files_dir, "destretched_control")
out_off_final_dir = os.path.join(files_dir, "off_final")
print(out_off_dir)
print(out_avg_dir)
print(out_dir)

# calculate offset vectors
print(f"calculating offsets... {out_off_dir}")
start = time.time()
calc_offset_vectors(
	files,
	out_off_dir,
	"off",
	kernel_sizes=kernel_sizes,
	# ref_method=ExternalRefs(files, get_fits_paths(out_ref_dir))
)
elapsed = time.time() - start
print(f"elapsed time: {elapsed}")

# calculate the cumulative sum offsets
print(f"calculating cumulative sums... {out_sum_dir}")
calc_cumulative_sums(
    get_fits_paths(out_off_dir),
    out_sum_dir
)

# calculate the rolling sums of the cumulative sum offsets
print(f"calculating tolling mean... {out_rolling_sum_dir}")
calc_rolling_mean(
    get_fits_paths(out_sum_dir),
    out_rolling_sum_dir
)

# calculate the difs
print(f"calculating final offs... {out_off_final_dir}")
calc_difs(
    get_fits_paths(out_sum_dir),
    get_fits_paths(out_rolling_sum_dir),
    out_off_final_dir,
    "final"
)

# calculate the flowmap
print(f"generating flow map data... {out_flow_dir}")
calc_change_rate(
    get_fits_paths(out_rolling_sum_dir),
    out_flow_dir
)

# apply the final offset vectors to destretch the image data
print(f"generating destretched image data... {out_dir}")
result = destretch_files(
	files,
	get_fits_paths(out_off_final_dir),
	out_dir,
	"destr",
	#in_avg_files=get_fits_paths(out_avg_dir),
)