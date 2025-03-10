"""
Utility module for storing information about flct destretching algorithm based 
on implementation by Momchil Molnar
"""

import numpy as np
from typing import Tuple, Literal, Any
from dataclasses import dataclass

## Definitions: ---------------------------------------------------------------|

@dataclass(frozen=True)
class DestretchLoopResult:
    """
    Data type that's returned by the destretch loop, for tracking destretch state
    """
    result: np.ndarray[Any, np.dtype[np.float64]]
    displace_sum: np.ndarray[tuple[Literal[2], Any, Any], np.dtype[np.float64]]
    ref_displace_sum: np.ndarray[tuple[Literal[2], Any, Any], np.dtype[np.float64]]
    destr_info: 'DestretchParams'

    def __iter__(self):
        yield self.result
        yield self.displace_sum
        yield self.ref_displace_sum
        yield self.destr_info

class DestretchParams():
    """
    structure for storing information needed to perform destretching
    """

    # kernel size x,y
    kx: int = 0
    ky: int = 0
        
    # border offset x,y
    wx: int = 0
    wy: int = 0

    # boundary size x,y
    bx: int = 0
    by: int = 0
        
    # number of control points x,y
    cpx: int = 0
    cpy: int = 0
        
    # apodization percentage
    mf: int = 0

    # array of control points
    rcps: int = 0

    # TODO describe these fields
    ref_sz_x: int = 0
    ref_sz_y: int = 0
    scene_sz_x: int = 0
    scene_sz_y: int = 0

    # order of polynomial to subtract from subfields
    subfield_correction: int = 0
        
    # TODO describe these fields
    max_fit_method: int = 1
    use_fft: bool = 0
    do_plots: bool = False
    debug: bool = False

    def print_props(self):
        """ print all the property values of this object """
        print("[kx, ky, wx, wy, bx, by, cpx, cpy, mf] are:")
        print(
            self.kx, self.ky, self.wx, self.wy, self.bx, self.by,
            self.cpx, self.cpy, self.mf
        )