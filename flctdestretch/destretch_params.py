"""
Utility module for storing information about flct destretching algorithm based 
on implementation by Momchil Molnar
"""

class DestretchParams():
    """
    structure for storing information needed to perform destretching
    
    Attributes
    ----------
    kx : int
        the kernel X size in pixels
    ky : int
        the kernel Y size in pixels
    wx : int
        the border X offset in pixels
    wy : int
        the border Y offset in pixels
    cpx : int
        amount of X control points
    cpy : int
        amount of X control points
    mf : float
        amount of apodization to apply in the apod mask
    rcps : Any
        TODO
    ref_sz_x : Any
        TODO
    ref_sz_y : Any
        TODO
    scene_sz_x : Any
        TODO
    scene_sz_y : Any
        TODO
    subfield_correction : Any
        order of polynomial to subtract from subfields
    max_fit_method : Any
        TODO
    use_fft : bool
        use fast fourier transform
    do_plots : Any
        TODO
    debug : Any
        TODO
    """
    def __init__(self,
            kx: int = 0, ky: int = 0, 
            wx: int = 0, wy: int = 0, 
            bx: int = 0, by: int = 0, 
            cpx: int = 0, cpy: int = 0, 
            mf: float = 0, rcps = 0, 
            ref_sz_x = 0, ref_sz_y = 0, 
            scene_sz_x = 0, scene_sz_y =0 , 
            subfield_correction = 0, 
            max_fit_method: int = 1, 
            use_fft: bool = 0, 
            do_plots = 0, debug = 0
        ):
        """    
        create a DestretchParams object
    
        Attributes
        ----------
        kx : int
            the kernel X size in pixels
        ky : int
            the kernel Y size in pixels
        wx : int
            the border X offset in pixels
        wy : int
            the border Y offset in pixels
        cpx : int
            amount of X control points
        cpy : int
            amount of X control points
        mf : float
            amount of apodization to apply in the apod mask
        rcps : Any
            TODO
        ref_sz_x : Any
            TODO
        ref_sz_y : Any
            TODO
        scene_sz_x : Any
            TODO
        scene_sz_y : Any
            TODO
        subfield_correction : Any
            order of polynomial to subtract from subfields
        max_fit_method : Any
            TODO
        use_fft : bool
            use fast fourier transform
        do_plots : Any
            TODO
        debug : Any
            TODO
        """
        # kernel size x,y
        self.kx = kx        
        self.ky = ky       
        
        # border offset x,y
        self.wx = wx
        self.wy = wy

        # boundary size x,y
        self.bx = bx
        self.by = by
        
        # number of control points x,y
        self.cpx = cpx
        self.cpy = cpy
        
        # apodization percentage
        self.mf = mf

        # array of control points
        self.rcps = rcps

        # TODO describe these fields
        self.ref_sz_x = ref_sz_x 
        self.ref_sz_y = ref_sz_y 
        self.scene_sz_x = scene_sz_x 
        self.scene_sz_y = scene_sz_y 

        # order of polynomial to subtract from subfields
        self.subfield_correction = subfield_correction  
        
        # TODO describe these fields
        self.max_fit_method = max_fit_method
        self.use_fft = use_fft
        self.do_plots = do_plots
        self.debug = debug

    def print_props(self):
        """ print all the property values of this object """
        print("[kx, ky, wx, wy, bx, by, cpx, cpy, mf] are:")
        print(
            self.kx, self.ky, self.wx, self.wy, self.bx, self.by,
            self.cpx, self.cpy, self.mf
        )