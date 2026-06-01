"""
TODO
"""

## Imports and Initialization --------------------------------------------------

import numpy as np

## Filtering Functionality -----------------------------------------------------

def get_psrs(
	correlations: np.ndarray, 
	mask_radius: float | None = None, 
	peak_coords: np.ndarray | None = None
) -> np.ndarray:
	"""
	Compute Peak-to-Sidelobe Ratio (PSR).

	PSR = (peak - mean_sidelobe) / std_sidelobe  
	where sidelobes exclude a small window around the peak.
	
	Parameters
	----------
	correlations : ndarray (count, kernel_width, kernel_height)
	mask_radius : float
	peak_coords : ndarray (count, 2)

	Reference
	---------
	Savvides, M., & Kumar, B. V. K. V. (2004).
    Face Verification Using Correlation Filters,
    http://mathdesc.fr/documents/facerecog/mace.pdf
	"""
	# mask_radius default is one eighth of the kernel size
	if mask_radius is None:
		mask_radius = min(min(correlations.shape[1:]) * 0.125, 2)
	
	# peak coordinates are recalcuated by default if not provided
	kwidth, kheight = correlations.shape[1], correlations.shape[2]
	count = correlations.shape[0]
	if peak_coords is None:
		peak_coords = np.zeros((count, 2), dtype=int)
		coords = np.argmax(correlations.reshape(count, -1), axis=1)
		peak_coords[:, 0] = coords % kwidth
		peak_coords[:, 1] = coords // kwidth
	
	# create the masks for the sidelobe regions of the correlations:
	# create homogenous coordinate grid for each correlation
	gy, gx = np.meshgrid(np.arange(kheight), np.arange(kwidth), indexing='ij')
	gx = gx[np.newaxis]
	gy = gy[np.newaxis]

	# broadcast peak positions into grid
	px = peak_coords[:, 0, np.newaxis, np.newaxis]
	py = peak_coords[:, 1, np.newaxis, np.newaxis]
	dist_sq = (gx - px) ** 2 + (gy - py) ** 2

	# get masks by distance check from broadcasted peak
	sidelobe_masks = dist_sq > mask_radius ** 2

	# apply sidelobe mask to separate correlation sidelobes from peak
	masked_correlations = np.where(sidelobe_masks, correlations, np.nan)
	sidelobe_means = np.nanmean(masked_correlations, axis=(1, 2))
	sidelobe_stds = np.nanstd(masked_correlations, axis=(1, 2))
	
	# calculate psrs from the value at the correlation peak
	indices = np.arange(count)
	peak_vals = correlations[indices, peak_coords[:, 0], peak_coords[:, 1]]
	psrs = (peak_vals - sidelobe_means) / sidelobe_stds

	return psrs

def psr_filter_weights(
	correlations: np.ndarray, 
	psr_min: float = 2.0, 
	psr_max: float | None = None
) -> np.ndarray:
	"""
	Filter out bad correlations based on their peak-to-sidelobe ratio (PSR). 
	Low PSR values are considered bad, so correlations with corresponding bad 
	PSRs will be removed. A mask is returned that determines which correlations 
	should be used and which should not. 

	- 1 = use correlation offset
	- 0 = discard correlation offset
	- between 0 and 1 = a sort of weight for how reliable that correlation is
	according to the PSR)

	Parameters
	----------
	correlations : ndarray (count, kernel_width, kernel_height)
		The correlations to apply the filter to
	psr_min : float
		Any correlation with a PSR lower than this is 0
	psr_max : float
		default = psr_min.
		Any correlation with a PSR above this is 1

	Returns
	-------
	weights : ndarray (count)
		a mask of weights that describes how reliable an offset from the given 
		correlations is
	"""
	# apply default value for psr max
	if psr_max is None: psr_max = psr_min
	assert psr_max is not None

	# calculate psrs and apply stretching / clipping
	weights = get_psrs(correlations)
	threshold_range = psr_max - psr_min
	weights = np.clip((weights - psr_min) * threshold_range, 0, 1)

	return weights
	