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
	count = correlations[0]
	if peak_coords is None:
		kernel_width = correlations[1]
		peak_coords = np.zeros((count, 2))
		coords = np.argmax(correlations.reshape(count, -1), axis=1)
		peak_coords[:, 0] = coords % kernel_width
		peak_coords[:, 1] = coords // kernel_width
	
	# create the masks for the sidelobe regions of the correlations
	sidelobe_masks = np.ones(correlations.shape, dtype=np.bool)
	# TODO mask out peak and surrounding radius

	# apply sidelobe mask to separate correlation sidelobes from peak
	sidelobe_means = np.mean(correlations, where=sidelobe_masks, axis=0)
	sidelobe_stds = np.std(correlations, where=sidelobe_masks, axis=0)
	
	# calculate psrs from the value at the correlation peak
	indices = np.arange(count)
	peak_vals = correlations[indices, peak_coords[1], peak_coords[2]]
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
	