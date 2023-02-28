import numpy as np

from definitions import *

class ChannelEvent:
    """
    Contains only the init method, which.
    The remamining methods (plots) are only used to debug.
    """    

    def __init__(self, mode = 'AGGRADATION', 
        nit = 100, dt = 0.1, saved_ts = 10,
        cr_dist = 200, cr_wind = 1500,
        Cf = 0.02, kl = 60.0, kv = 0.01, number_layers=3, #Dennis: added here
        ch_depth = lambda slope: -20 * slope,
        ch_width = lambda slope: 700 * np.exp(0.80 * slope) + 95, 
        dep_height = lambda slope: -20 * slope * 1/4,
        dep_props = lambda slope: (0.3, 0.0, 0.0, 0.5, 0.0, 0.0, 0.2, 0.0), # initializing 7 layers + 1 for separator
        dep_sigmas = lambda slope: (0.25, BIG_NUMBER, BIG_NUMBER, 0.5, BIG_NUMBER, BIG_NUMBER, 2, BIG_NUMBER),
        aggr_props = lambda slope: (0.333, 0.0, 0.0, 0.333, 0.0, 0.0, 0.333, 0.0),
        aggr_sigmas = lambda slope: (2, BIG_NUMBER, BIG_NUMBER, 5, BIG_NUMBER, BIG_NUMBER, 10, BIG_NUMBER),        
        sep_thickness = 0, #dennis: separator thickness from the SEPARADOR mode
        sep_type = 'CONDENSED_SECTION', #dennis: separator type from the SEPARADOR mode
        dens = 1000, aggr_factor = 2):
        """       
        Initializes a ChannelEvent object.
        Channel width and depth: parameters to determine the shape of the channel used to cut the terrain.
        Deposition height, proportions and sigmas: parameters used in the deposition process that takes place after the channel cut step.
        Aggradation Proportions, and Sigmas: parameters used in the aggradation process that is quite similar to the deposition process above.
        However, in contrast to the deposition, the aggradation only is performed when there is a elevation in the basin profile.

        :param mode: event type (Incision, Aggradation or Separator).
        :param nit: number of iterations.
        :param dt: time of each iteration (in years).
        :param saved_ts: number of iterations interval which the mesh is saved.
        :param cr_dist: TODO
        :param cr_wind: TODO
        :param Cf: TODO
        :param kl: Meandering modulation (m/years).
        :param kv: aggradation/incision modulation (m/years).
        :param number_layers: number of layers
        :param ch_depth: channel depth in meters.
        :param ch_width: channel width in meters.
        :param dep_height: deposition height in meters per slope.
        :param dep_props: deposition proportions of the material filling the deposition depth.
        :param dep_sigmas: TODO
        :param aggr_props: proportions of the materials used in aggradation cycle.
        :param aggr_sigmas: TODO
        :param sep_thickness: thickness of the event separator
        :param dens: water density in kg/m³ = 1000 kg/m³
        :param aggr_factor: aggradation factor. According to Sylvester (meanderpy.py, line 201) it should be larger than 1 so that this leads to overall aggradation
        """        
        
        #dennis: Initialize unused variables for the events
        # CHECK HERE  
        '''
        if (mode == 'INCISION'):
            aggr_props = lambda slope: (0, 0, 0, 0, 0, 0, 0, 0)
            aggr_sigmas = lambda slope: (0, 0, 0, 0, 0, 0, 0, 0)
            sep_thickness = 0
        elif (mode == 'AGGRADATION'):
            sep_thickness = 0                
        elif (mode == 'SEPARATOR'):
            dep_props = lambda slope: (0, 0, 0, 0, 0, 0, 0, 1)
            dep_sigmas = lambda slope: (0, 0, 0, 0, 0, 0, 0, 1)
            aggr_props = lambda slope: (0, 0, 0, 0, 0, 0, 0, 1)
            aggr_sigmas = lambda slope: (0, 0, 0, 0, 0, 0, 0, 1)
        '''        

        self.mode = mode
        self.nit = nit
        self.dt = dt
        self.saved_ts = saved_ts
        self.cr_dist = cr_dist
        self.cr_wind = cr_wind
        self.Cf = Cf
        self.kl = kl
        self.kv = kv
        self.number_layers = number_layers

        self.ch_depth = ch_depth
        self.ch_width = ch_width
        self.dep_height = dep_height
        self.dep_props = dep_props
        self.dep_sigmas = dep_sigmas

        self.aggr_props = aggr_props
        self.aggr_sigmas = aggr_sigmas
        
        self.sep_thickness = sep_thickness
        self.sep_type = sep_type
        
        self.dens = dens
        self.aggr_factor = aggr_factor
        self.start_time = -1