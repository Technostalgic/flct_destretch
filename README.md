# flctDestretch

Python package containing algorithms for destretching, or removing optical defects, from 
image sequences in heliophysics datasets (probably works for other datasets too).

Destretching algorithms were implemented by [@momomolnar](https://github.com/momomolnar) in the
original 
[Destretching Algorithms repository](https://github.com/momomolnar/Destretching_Algorithms) meant
to showcase the effectiveness of the algorithms. The source has been modified so that it can be
packaged more easily.

For a more in-depth explanation on what exactly the destretching allgorithms are doing, and how
and why they work, I highly recommend cloning the repository linked above and walking through the 
jupyter notebooks provided in it.

This package is meant just for the purpose of utilizing the algorithms to improve image quality and
not intended to explain the math behind it.

## Installation

TODO

## Usage

TODO

### Examples

TODO see ./examples

from repository root, run `python -m examples/movie_example/py` to see a before and after applying
our algorithm to a 12 frame image sequence.