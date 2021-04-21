import bisect
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.interpolate
import pyvista as pv
from scipy.spatial import distance
from scipy import ndimage, stats
from scipy.signal import savgol_filter
from PIL import Image, ImageDraw, ImageFilter
from skimage import measure
from skimage import morphology
from matplotlib.colors import LinearSegmentedColormap
import time, sys
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

OMEGA = -1.0 # constant in curvature calculation (Howard and Knutson, 1984)
GAMMA = 2.5  # from Ikeda et al., 1981 and Howard and Knutson, 1984
K = 4.0 # constant in HK equation
YEAR = 365*24*60*60.0

def update_progress(progress):
    """progress bar from https://stackoverflow.com/questions/3160699/python-progress-bar
    update_progress() : Displays or updates a console progress bar
    Accepts a float between 0 and 1. Any int will be converted to a float.
    A value under 0 represents a 'halt'.
    A value at 1 or bigger represents 100%"""
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

def compute_migration_rate(r0, Cf, d, dl, L):
    """compute migration rate as weighted sum of upstream curvatures
    pad - padding (number of nodepoints along centerline)
    ns - number of points in centerline
    ds - distances between points in centerline
    omega - constant in HK model
    gamma - constant in HK model
    R0 - nominal migration rate (dimensionless curvature * migration rate constant)"""
    NS = len(dl)
    r1 = np.zeros(NS) # preallocate adjusted channel migration rate
    for i in range(0, NS):
        SIGMA_2 = np.hstack((np.array([0]), np.cumsum(dl[i-1::-1])))  # distance along centerline, backwards from current point
        if d[i] > 1:
            G = np.exp(-2.0 * K * Cf / d[i] * SIGMA_2) # convolution vector
            r1[i] = OMEGA*r0[i] + GAMMA*np.sum(r0[i::-1]*G)/np.sum(G) # main equation
        else:
            r1[i] = r0[i]
    return r1

def find_cutoffs(x, y, crdist, diag):
    # distance matrix for centerline points:
    dist = distance.cdist(np.array([x,y]).T,np.array([x,y]).T)
    dist[dist>crdist] = np.NaN # set all values that are larger than the cutoff threshold to NaN
    # set matrix to NaN along the diagonal zone:
    rows, cols = np.diag_indices_from(dist)

    for k in range(-diag,0):
        dist[rows[:k], cols[-k:]] = np.NaN
    dist[rows, cols] = np.NaN
    for k in range(1, diag + 1):
        dist[rows[k:], cols[:-k]] = np.NaN

    i1, i2 = np.where(~np.isnan(dist))
    ind1 = i1[np.where(i1<i2)[0]] # get rid of unnecessary indices
    ind2 = i2[np.where(i1<i2)[0]] # get rid of unnecessary indices

    return ind1, ind2 # return indices of cutoff points and cutoff coordinates

def find_cutoffs_R(R, W = 5, T = 1):
    '''
        R - curvature * width (dimensionless curvature)
        W - window size (in elements) that will be cut
        T - threshold for cut
    '''
    indexes = np.where(np.abs(R) > T)[0][-1:]

    if len(indexes) == 0:
        return -1, -1

    ind1, ind2 = indexes[0] - W, indexes[0] + W
    
    for i in indexes:
        if i > ind1:
            ind1 = i - W
        
    return max(ind1, 0), min(ind2, len(R) -1)

class Basin:    
    def __init__(self, x, z):
        self.x = x
        self.z = z

    def copy(self):
        return Basin(self.x.copy(), self.z.copy())

    def fit_elevation(self, x):
        return scipy.interpolate.interp1d(self.x, self.z, kind='cubic', fill_value='extrapolate')(x)

    def fit_slope(self, x, ws = 2500):
        return scipy.interpolate.interp1d(self.x, self.slope(ws), kind='cubic', fill_value='extrapolate')(x)

    def slope(self, ws = 2500, degrees = True):
        slope = np.gradient(self.z, self.x)
        NS = len(self.x)
        sl = np.zeros(NS)

        for i in range(0, NS):
            t = (self.x[i:] - self.x[i]) / ws
            G = np.exp(-t ** 2) 
            sl[i] = np.sum(slope[i:] * G)/np.sum(G)
        
        if not degrees:
            return sl
        else:
            return np.arctan(sl) * 180 / np.pi

    def aggradate(self, density, kv, dt, aggr_factor):
        slope = self.slope(degrees=False)
        K = kv * density * 9.81 * dt
        self.z += K *(slope - aggr_factor*np.mean(slope))

    def incise(self, density, kv, dt):
        slope = self.slope(degrees=False)
        K = kv * density * 9.81 * dt
        self.z += K *slope 

    def plot(self, axis = plt, color=sns.xkcd_rgb["ocean blue"], points = False):
        axis.plot(self.x, self.z)

class Channel:
    """class for Channel objects"""
    def __init__(self, x, y, z = [], d = [], w = []):
        """initialize Channel object
        x, y, z  - coordinates of centerline
        W - channel width
        D - channel depth"""
        self.x = x
        self.y = y
        self.z = z
        self.d = d
        self.w = w

    def copy(self):
        return Channel(self.x.copy(), self.y.copy(), self.z.copy(), self.d.copy(), self.w.copy())

    def margin_offset(self):
        d = self.w / 2

        dx = np.gradient(self.x)
        dy = np.gradient(self.y)

        n = np.stack((dy, -dx))
        l = np.sqrt(np.sum(np.conj(n) * n, axis = 0))
            
        xo = d * (  dy / l )
        yo = d * ( -dx / l )

        return xo, yo

    def derivatives(self):
        dx = np.gradient(self.x)
        dy = np.gradient(self.y)
        dz = np.gradient(self.z)
        ds = np.sqrt(dx**2 + dy**2 + dz**2)
        s = np.hstack((0,np.cumsum(ds[1:])))

        return dx, dy, dz, ds, s

    def curvature(self):
        dx = np.gradient(self.x) 
        dy = np.gradient(self.y)      
        
        ddx = np.gradient(dx)
        ddy = np.gradient(dy) 

        return (dx*ddy-dy*ddx)/((dx**2+dy**2)**1.5)
        
    def refit(self, basin, ch_width, ch_depth):
        slope = basin.fit_slope(self.x)

        self.z = basin.fit_elevation(self.x)
        self.w  = ch_width(slope)
        self.d  = ch_depth(slope)

    def resample(self, target_ds):
        _, _, _, _, s = self.derivatives()
        N = 1 + int(round(s[-1]/target_ds))

        tck, _ = scipy.interpolate.splprep([self.x, self.y], s=0)
        u = np.linspace(0,1,N)
        self.x, self.y = scipy.interpolate.splev(u,tck)

    def migrate(self, Cf, kl, dt):
        curv = self.curvature()
        dx, dy, _, ds, s = self.derivatives()
        sinuosity = s[-1]/(self.x[-1]-self.x[0])
        # Velocity is proportial to cross section area
        # Friction force is proportial to contact surface area
        area = np.clip(self.d, a_min = 0, a_max = None) * self.w / 2

        R0 = kl * self.w * curv 
        R1 = compute_migration_rate(R0, Cf, self.d, ds, s[-1])

        RN = sinuosity**(-2/3.0) * R1 * (area / np.max(area))
        #plt.plot(self.x, R0, self.x, R1, self.x, RN);plt.show()
        #plt.plot(self.x, (area / np.max(area)));plt.show()
        
        self.x += RN * (dy/ds) * dt 
        self.y -= RN * (dx/ds) * dt
 
    def cut_cutoffs(self, crdist, ds):     
        cuts = []   
        
        diag_blank_width = int((crdist+20*ds)/ds)
        # UPPER MARGIN
        xo, yo = self.margin_offset()
        ind1, ind2 = find_cutoffs(self.x+xo, self.y+yo, crdist, diag_blank_width)
        while len(ind1)>0:

            xc = self.x[ind1[0]:ind2[0]+1] # x coordinates of cutoff
            yc = self.y[ind1[0]:ind2[0]+1] # y coordinates of cutoff
            zc = self.z[ind1[0]:ind2[0]+1] # z coordinates of cutoff
            dc = self.d[ind1[0]:ind2[0]+1] # d coordinates of cutoff
            wd = self.w[ind1[0]:ind2[0]+1] # w coordinates of cutoff

            cuts.append(Channel(xc, yc, zc, dc, wd))

            self.x = np.hstack((self.x[:ind1[0]+1],self.x[ind2[0]:])) # x coordinates after cutoff
            self.y = np.hstack((self.y[:ind1[0]+1],self.y[ind2[0]:])) # y coordinates after cutoff
            self.z = np.hstack((self.z[:ind1[0]+1],self.z[ind2[0]:])) # z coordinates after cutoff
            self.w = np.hstack((self.w[:ind1[0]+1],self.w[ind2[0]:])) # z coordinates after cutoff
            self.d = np.hstack((self.d[:ind1[0]+1],self.d[ind2[0]:])) # z coordinates after cutoff

            xo, yo = self.margin_offset()
            ind1, ind2 = find_cutoffs(self.x+xo, self.y+yo, crdist, diag_blank_width)

        # LOWER MARGIN
        xo, yo = self.margin_offset()
        ind1, ind2 = find_cutoffs(self.x-xo, self.y-yo, crdist, diag_blank_width)
        while len(ind1)>0:

            xc = self.x[ind1[0]:ind2[0]+1] # x coordinates of cutoff
            yc = self.y[ind1[0]:ind2[0]+1] # y coordinates of cutoff
            zc = self.z[ind1[0]:ind2[0]+1] # z coordinates of cutoff
            dc = self.d[ind1[0]:ind2[0]+1] # d coordinates of cutoff
            wd = self.w[ind1[0]:ind2[0]+1] # w coordinates of cutoff

            cuts.append(Channel(xc, yc, zc, dc, wd))

            self.x = np.hstack((self.x[:ind1[0]+1],self.x[ind2[0]:])) # x coordinates after cutoff
            self.y = np.hstack((self.y[:ind1[0]+1],self.y[ind2[0]:])) # y coordinates after cutoff
            self.z = np.hstack((self.z[:ind1[0]+1],self.z[ind2[0]:])) # z coordinates after cutoff
            self.w = np.hstack((self.w[:ind1[0]+1],self.w[ind2[0]:])) # z coordinates after cutoff
            self.d = np.hstack((self.d[:ind1[0]+1],self.d[ind2[0]:])) # z coordinates after cutoff

            xo, yo = self.margin_offset()
            ind1, ind2 = find_cutoffs(self.x-xo, self.y-yo, crdist, diag_blank_width)

        return cuts

    def cut_cutoffs_R(self, cut_window, ds):
        D = int(cut_window / (2 * ds))
        ind1, ind2 = find_cutoffs_R(self.w / 2 * self.curvature(), D)
        if ind1 != -1:
            self.x = np.hstack((self.x[:ind1+1],self.x[ind2:])) # x coordinates after cutoff
            self.y = np.hstack((self.y[:ind1+1],self.y[ind2:])) # y coordinates after cutoff
            self.z = np.hstack((self.z[:ind1+1],self.z[ind2:])) # z coordinates after cutoff
            self.w = np.hstack((self.w[:ind1+1],self.w[ind2:])) # z coordinates after cutoff
            self.d = np.hstack((self.d[:ind1+1],self.d[ind2:])) # z coordinates after cutoff
            ind1, ind2 = find_cutoffs_R(self.w * self.curvature(), D)

    def plot(self, axis = plt, color=sns.xkcd_rgb["ocean blue"], points = False):
        x = self.x
        y = self.y

        if self.w == []:
            axis.plot(x, y)
            return

        xo, yo = self.margin_offset()
        
        xm = np.hstack((x + xo, (x - xo)[::-1]))
        ym = np.hstack((y + yo, (y - yo)[::-1]))
        
        if points:
            axis.plot(x, y)
        else:
            axis.fill(xm, ym, color=color, edgecolor='k', linewidth=0.25)

class ChannelMapper:
    def __init__(self, xmin, xmax, ymin, ymax, xsize, ysize, downscale = 4, sigma = 2):
        self.xmin = xmin
        self.ymin = ymin
        
        self.downscale = downscale
        self.sigma = sigma

        #grid size
        self.xsize = int(xsize / downscale)
        self.ysize = int(ysize / downscale)

        self.dx = xmax - xmin
        self.dy = ymax - ymin

        self.width = int(self.dx / self.xsize)
        self.height = int(self.dy / self.ysize)

        self.rwidth = int(self.width/self.downscale)
        self.rheight = int(self.height/self.downscale)

    def __repr__(self):
        return 'GRID-SIZE: ({};{})\nIMAGE-SIZE: ({};{})\n PIXELS: {}'.format(self.xsize, self.ysize, self.width, self.height, self.width * self.height)

    def map_size(self):
        return (self.xsize, self.ysize)

    def post_processing(self, _map):
        return self.downsize(self.filter(_map))

    def filter(self, _map):
        return scipy.ndimage.gaussian_filter(_map, sigma = self.sigma)

    def downsize(self, _map):
        return np.array(Image.fromarray(_map).resize((self.rwidth, self.rheight), Image.BILINEAR))

    def create_maps(self, channel, basin):
        ch_map = self.create_ch_map(channel)
        
        cld_map = self.create_cld_map(channel)
        md_map = self.create_md_map(channel)
        cz_map = self.create_z_map(channel)
        bz_map = self.create_z_map(basin)
        sl_map = self.create_sl_map(basin)

        hw_inside = cld_map + md_map
        hw_outside = cld_map - md_map

        hw_inside[np.array(np.logical_not(ch_map).astype(bool))] = 0.0
        hw_outside[np.array(ch_map.astype(bool))] = 0.0
        hw_map = hw_inside + hw_outside

        return (ch_map, cld_map, md_map, cz_map, bz_map, sl_map, hw_map)

    def create_md_map(self, channel):
        xo, yo = channel.margin_offset()

        upper_pixels = self.to_pixels(channel.x + xo, channel.y + yo)
        lower_pixels = self.to_pixels(channel.x - xo, channel.y - yo)

        img = Image.new("1", (self.width, self.height), 1)
        draw = ImageDraw.Draw(img)
        draw.line(upper_pixels, fill=0) 
        draw.line(lower_pixels, fill=0)

        md_map = ndimage.distance_transform_edt(np.array(img), sampling=[self.xsize, self.ysize])

        return self.post_processing(md_map)

    def create_cld_map(self, channel):
        pixels = self.to_pixels(channel.x, channel.y)
        img = Image.new("1", (self.width, self.height), 1)
        draw = ImageDraw.Draw(img)
        draw.line(pixels, fill=0)
    
        cld_map = ndimage.distance_transform_edt(np.array(img), sampling=[self.xsize, self.ysize])

        return self.post_processing(cld_map)

    def create_ch_map(self, channel):
        x, y = channel.x, channel.y
        xo, yo = channel.margin_offset()

        xm = np.hstack(((x + xo), (x - xo)[::-1]))
        ym = np.hstack(((y + yo), (y - yo)[::-1]))

        xy = self.to_pixels(xm, ym)

        img = Image.new("1", (self.width, self.height), 0)
        draw = ImageDraw.Draw(img)
        draw.polygon(xy, fill=1)

        return self.downsize(np.array(img))

    def create_z_map(self, basin):
        x_p = ((basin.x - self.xmin) / self.dx) * self.width

        tck, _ = scipy.interpolate.splprep([x_p, basin.z], s = 0)
        u = np.linspace(0,1,self.width)
        _, z_level = scipy.interpolate.splev(u, tck)

        return self.post_processing(np.tile(z_level, (self.height, 1)))

    def create_sl_map(self, basin):
        x_p = ((basin.x - self.xmin) / self.dx) * self.width
        
        tck, _ = scipy.interpolate.splprep([x_p, basin.slope()], s = 0)
        u = np.linspace(0,1,self.width)
        _, z_level = scipy.interpolate.splev(u, tck)

        return self.post_processing(np.tile(z_level, (self.height, 1)))

    def plot_map(self, _map):
        plt.matshow(_map)
        plt.colorbar()
        plt.show()

    def to_pixels(self, x, y):
        x_p = ((x - self.xmin) / self.dx) * self.width
        y_p = ((y - self.ymin) / self.dy) * self.height

        xy = np.vstack((x_p, y_p)).astype(int).T
        return tuple(map(tuple, xy))

def topostrat(topo):
    """function for converting a stack of geomorphic surfaces into stratigraphic surfaces
    inputs:
    topo - 3D numpy array of geomorphic surfaces
    returns:
    strat - 3D numpy array of stratigraphic surfaces
    """
    r,c,ts = np.shape(topo)
    strat = np.copy(topo)
    for i in (range(0,ts)):
        strat[:,:,i] = np.amin(topo[:,:,i:], axis=2)
    return strat

def topostrat_evolution(topo):
    """function for converting a stack of geomorphic surfaces into stratigraphic surfaces
    inputs:
    topo - 3D numpy array of geomorphic surfaces
    returns:
    strat - 3D numpy array of stratigraphic surfaces
    """
    N = 4
    r,c,ts = np.shape(topo)
    strat = np.zeros((r,c,int(ts/N)))
    for i in (range(0,ts, N)):
        strat[:,:,int((i+1)/N)] = np.amin(topo[:,:,i:i+N], axis=2)
    return strat

def plot3D(Z, grid_size = 1):
    X, Y = np.meshgrid(np.linspace(0, Z.shape[1] * grid_size, Z.shape[1]), np.linspace(0, Z.shape[0] * grid_size, Z.shape[0]))
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z)

    max_range = np.array([X.max()-X.min(), Y.max()-Y.min()]).max() / 2.0

    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range/4, mid_y + max_range/4)

    return fig

def erosional_surface(cld_map, z_map, hw_map, cd_map):
    return cd_map * ((cld_map / hw_map) ** 2 - 1) + z_map

def gausian_surface(sigma_map, cld_map, hw_map):
    return np.exp(- 1 / 2 * ((cld_map / hw_map) / sigma_map) ** 2)

class ChannelEvent:
    '''
        mode: 'INCISION' | 'AGGRADATION' 
    '''
    def __init__(self, mode = 'AGGRADATION', 
        nit = 100, dt = 0.1, saved_ts = 10,
        cr_dist = 200, cr_wind = 1500,
        Cf = 0.02, kl = 60.0, kv = 0.01,

        ch_depth = lambda slope: -20 * slope, ch_width = lambda slope: 700 * np.exp(0.80 * slope) + 95, 
        dep_height = lambda slope: -20 * slope * 1/4 , dep_props = lambda slope: (0.3, 0.5, 0.2), dep_sigmas = lambda slope: (0.25, 0.5, 2),
        aggr_props = lambda slope: (1, 1, 1), aggr_sigmas = lambda slope: (2, 5, 10), 

        dens = 1000, aggr_factor = 2):

        self.mode = mode
        self.nit = nit
        self.dt = dt
        self.saved_ts = saved_ts
        self.cr_dist = cr_dist
        self.cr_wind = cr_wind
        self.Cf = Cf
        self.kl = kl
        self.kv = kv

        self.ch_depth = ch_depth
        self.ch_width = ch_width
        self.dep_height = dep_height
        self.dep_props = dep_props
        self.dep_sigmas = dep_sigmas
        self.aggr_props = aggr_props
        self.aggr_sigmas = aggr_sigmas

        self.dens = dens
        self.aggr_factor = aggr_factor
        self.start_time = -1

    def plot_ch_depth(self, slope = np.linspace(-5, 0, 20), axis = None):
        if axis is None:
            fig, axis = plt.subplots(1, 1)
        else:
            fig = None

        axis.set_xlabel('Slope(°)')
        axis.set_ylabel('Channel Depth')
        axis.plot(slope, self.ch_depth(slope))
        return fig

    def plot_ch_width(self, slope = np.linspace(-5, 0, 20), axis = None):
        if axis is None:
            fig, axis = plt.subplots(1, 1)
        else:
            fig = None
            
        axis.set_xlabel('Slope(°)')
        axis.set_ylabel('Channel Width(m)')
        axis.plot(slope, self.ch_width(slope))
        return fig

    def plot_dep_height(self, slope = np.linspace(-5, 0, 20), axis = None):
        if axis is None:
            fig, axis = plt.subplots(1, 1)
        else:
            fig = None
            
        axis.set_xlabel('Slope(°)')
        axis.set_ylabel('Deposition Height(m)')
        axis.plot(slope, self.dep_height(slope))
        return fig

    def plot_dep_props(self, slope = np.linspace(-5, 0, 20), axis = None):
        if axis is None:
            fig, axis = plt.subplots(1, 1)
        else:
            fig = None
        
        L = len(slope)
        gr_p, sa_p, si_p = self.dep_props(slope)
        t_p = gr_p + sa_p + si_p
        axis.set_ylim(0, 1) 
        axis.set_xlabel('Slope(°)')
        axis.set_ylabel('Deposition Proportions')
        axis.plot(slope, gr_p / t_p * np.ones(L), slope, sa_p / t_p * np.ones(L), slope, si_p / t_p * np.ones(L))
        axis.legend(['% gravel', '% sand', '% silt'])
        return fig

    def plot_dep_sigmas(self, slope = np.linspace(-5, 0, 20), axis = None):
        if axis is None:
            fig, axis = plt.subplots(1, 1)
        else:
            fig = None
        
        L = len(slope)
        gr_s, sa_s, si_s = self.dep_sigmas(slope)
        axis.set_xlabel('Slope(°)')
        axis.set_ylabel('Deposition Sigmas')
        axis.plot(slope, gr_s * np.ones(L), slope, sa_s * np.ones(L), slope, si_s * np.ones(L))
        axis.legend(['gravel', ' sand', 'silt'])
        return fig

    def plot_aggr_props(self, slope = np.linspace(-5, 0, 20), axis = None):
        if axis is None:
            fig, axis = plt.subplots(1, 1)
        else:
            fig = None
        
        L = len(slope)
        gr_p, sa_p, si_p = self.aggr_props(slope)
        t_p = gr_p + sa_p + si_p
        axis.set_ylim(0, 1)
        axis.set_xlabel('Slope(°)')
        axis.set_ylabel('Aggradation Proportions')
        axis.plot(slope, gr_p / t_p * np.ones(L), slope, sa_p / t_p * np.ones(L), slope, si_p / t_p * np.ones(L))
        axis.legend(['gravel', ' sand', 'silt'])
        return fig

    def plot_aggr_sigmas(self, slope = np.linspace(-5, 0, 20), axis = None):
        if axis is None:
            fig, axis = plt.subplots(1, 1)
        else:
            fig = None
        
        L = len(slope)
        gr_s, sa_s, si_s = self.aggr_sigmas(slope)
        axis.set_xlabel('Slope(°)')
        axis.set_ylabel('Aggradation Sigmas')
        axis.plot(slope, gr_s * np.ones(L), slope, sa_s * np.ones(L), slope, si_s * np.ones(L))
        axis.legend(['gravel', ' sand', 'silt'])
        return fig

    def plot_all_relations(self):
        fig, axes = plt.subplots(4, 2)

        self.plot_ch_depth(axis = axes[0][0])
        self.plot_ch_width(axis = axes[0][1])
        self.plot_dep_height(axis = axes[1][0])
        self.plot_dep_props(axis = axes[2][0])
        self.plot_dep_sigmas(axis = axes[2][1])
        self.plot_aggr_props(axis = axes[3][0])
        self.plot_aggr_sigmas(axis = axes[3][1])

        return fig

class ChannelBelt:
    def __init__(self, channel, basin):
        """
            Times in years.
        """
        self.channels = [channel.copy()]
        self.basins = [basin.copy()]
        self.times = [0.0]
        self.events = []

    def simulate(self, event):
        last_time = self.times[-1]
        event.start_time = last_time + event.dt

        if len(self.events) == 0:
            channel = self.channels[0]
            basin = self.basins[0]
            self.events.append(event)
            channel.refit(basin, event.ch_width, event.ch_depth)
            _, _, _, ds, _ = channel.derivatives()
            self.ds = np.mean(ds)
            event.start_time = 0

        channel = self.channels[-1].copy()
        basin = self.basins[-1].copy()
        last_time = self.times[-1]


        for itn in range(event.nit):
            update_progress(itn/event.nit)

            channel.migrate(event.Cf, event.kl / YEAR, event.dt * YEAR)
            channel.cut_cutoffs(event.cr_dist, self.ds)
            channel.cut_cutoffs_R(event.cr_wind, self.ds)
            channel.resample(self.ds)
            channel.refit(basin, event.ch_width, event.ch_depth)
            
            if event.mode == 'INCISION':
                basin.incise(event.dens, event.kv / YEAR, event.dt * YEAR)
            if event.mode == 'AGGRADATION':
                basin.aggradate(event.dens, event.kv / YEAR, event.dt * YEAR, event.aggr_factor)

            if itn % event.saved_ts == 0:

                self.times.append(last_time + (itn+1) * event.dt)
                self.channels.append(channel.copy())
                self.basins.append(basin.copy())
                self.events.append(event)

    def plot_basin(self, evolution = True):
        fig, axis = plt.subplots(1, 1)
        if not evolution:
            self.basins[-1].plot(axis)
        else:
            legends = []
            uniques = set()
            self.basins[0].plot(axis)
            legends.append('initial')
            for evt in self.events:
                i = self.times.index(evt.start_time)
                if not i in uniques:
                    uniques.add(i)
                    self.basins[i + int(evt.nit / evt.saved_ts) - 1].plot(axis)
                    legends.append('event-{}'.format(len(uniques)))
            axis.legend(legends)
        axis.set_xlabel('X (m)')
        axis.set_ylabel('Elevation (m)')
        return fig

    def plot(self, start_time=0, end_time = 0, points = False):
        start_index = 0
        if start_time > 0:
            start_index = bisect.bisect_left(self.times, start_time)

        end_index = len(self.times)
        if end_time > 0:
            end_index = bisect.bisect_right(self.times, end_time)
            
        fig, axis = plt.subplots(1, 1)
        axis.set_aspect('equal', 'datalim')

        for i in range(start_index, end_index):
            color = sns.xkcd_rgb["ocean blue"] if i == end_index - 1 else sns.xkcd_rgb["sand yellow"]
            self.channels[i].plot(axis, color, points)

        return fig

    def build_3d_model(self, dx, margin = 500):
        xmax, xmin, ymax, ymin = [], [], [], []
        for channel in self.channels:
            xmax.append(max(channel.x))
            xmin.append(min(channel.x))
            ymax.append(max(channel.y))
            ymin.append(min(channel.y))

        xmax = max(xmax)
        xmin = min(xmin)
        ymax = max(ymax)
        ymin = min(ymin)

        mapper = ChannelMapper(xmin + margin, xmax - margin, ymin - margin, ymax + margin, dx, dx)

        channel = self.channels[0]
        basin = self.basins[0]
        ch_map, cld_map, md_map, cz_map, bz_map, sl_map, hw_map = mapper.create_maps(channel, basin)

        surface = bz_map

        N = len(self.channels)
        L = 3 + 1

        topo = np.zeros((mapper.rheight, mapper.rwidth, N*L))

        for i in range(0, N):
            update_progress(i/N)
            event = self.events[i]
            # Last iteration 
            aggr_map = bz_map - surface
            aggr_map[aggr_map < 0] = 0


            # channel, centerline distance, channel z, basin z, slope, half width
            ch_map, cld_map, md_map, cz_map, bz_map, sl_map, hw_map = mapper.create_maps(self.channels[i], self.basins[i])
            # channel depth
            dh_map = event.dep_height(sl_map)
            cd_map = event.ch_depth(sl_map)

            channel_surface = erosional_surface(cld_map, cz_map, hw_map, cd_map)
            
            gr_p, sa_p, si_p = event.dep_props(sl_map)
            gr_s, sa_s, si_s = event.dep_sigmas(sl_map)
            t_p = gr_p + sa_p + si_p

            gravel_surface = (gr_p / t_p) * dh_map * gausian_surface(gr_s, cld_map, hw_map)
            sand_surface = (sa_p / t_p) * dh_map * gausian_surface(sa_s, cld_map, hw_map)
            silt_surface = (si_p / t_p) * dh_map * gausian_surface(si_s, cld_map, hw_map)

            gr_p, sa_p, si_p = event.aggr_props(sl_map)
            gr_s, sa_s, si_s = event.aggr_sigmas(sl_map)
            t_p = gr_p + sa_p + si_p

            gravel_surface += (gr_p / t_p) * aggr_map
            sand_surface += (sa_p / t_p) * aggr_map
            silt_surface += (si_p / t_p) * aggr_map

            # CUTTING CHANNEL
            surface = scipy.ndimage.gaussian_filter(np.minimum(surface, channel_surface), sigma = 10 / dx)

            topo[:,:,i*L + 0] = surface

            # DEPOSITING SEDIMENT
            surface += gravel_surface
            topo[:,:,i*L + 1] = surface
            surface += sand_surface
            topo[:,:,i*L + 2] = surface
            surface += silt_surface
            topo[:,:,i*L + 3] = surface
            
        return ChannelBelt3D(topo, xmin, ymin, dx, dx)

class ChannelBelt3D():
    def __init__(self, topo, xmin, ymin, dx, dy):
        self.strat = topostrat(topo)
        self.topo = topo

        self.xmin = xmin
        self.ymin = ymin

        zmin, zmax = np.amin(self.strat[:,:,0]), np.amax(self.strat[:,:,-1])
        dz = zmax - zmin
        
        self.zmin = zmin - dz * 0.1 
        self.zmax = zmax + dz * 0.1
    
        self.dx = dx
        self.dy = dy

    def plot_xsection(self, xsec, ve = 5, substrat = True, silt_color = [51/255, 51/255, 0], sand_color = [255/255, 204/255, 0], gravel_color = [255/255, 102/255, 0]):
        strat = self.strat
        sy, sx, sz = np.shape(strat)
        
        xindex = int(xsec * sx)

        fig1 = plt.figure(figsize=(20,5))
        ax1 = fig1.add_subplot(111)
        ax1.set_title('({:.3f}) - {:.3f} km'.format(xsec, xindex * self.dx + self.xmin))

        Xv = np.linspace(self.ymin, self.ymin + sy * self.dy, sy)
        X1 = np.concatenate((Xv, Xv[::-1]))
        
        if substrat:
            Yb = np.ones(sy) * self.zmin
            ax1.fill(X1, np.concatenate((Yb, strat[::-1,xindex,0])), facecolor=[192/255, 192/255, 192/255])
        
        for i in range(0, sz, 4):
            Y1 = np.concatenate((strat[:,xindex,i],   strat[::-1,xindex,i+1])) 
            Y2 = np.concatenate((strat[:,xindex,i+1], strat[::-1,xindex,i+2]))
            Y3 = np.concatenate((strat[:,xindex,i+2], strat[::-1,xindex,i+3]))

            ax1.fill(X1, Y1, facecolor=gravel_color)
            ax1.fill(X1, Y2, facecolor=sand_color) 
            ax1.fill(X1, Y3, facecolor=silt_color)
        
        #ax1.set_aspect(ve, adjustable='datalim')
        ax1.set_xlim(self.ymin, self.ymin + sy * self.dy)
        ax1.set_ylim(-100, 800)
        
        return fig1

    def plot(self, ve = 1, curvature = False, save = False):
        sy, sx, sz = np.shape(self.strat)
        x = np.linspace(self.xmin, self.xmin + sx * self.dx, sx)
        y = np.linspace(self.ymin, self.ymin + sy * self.dy, sy)

        xx, yy = np.meshgrid(x, y)
        zz = self.strat[:,:,-1 - 4] * ve

        grid = pv.StructuredGrid(xx, yy, zz)

        if curvature:
            grid.plot_curvature()
        else:
            grid.plot()

        if save:
            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(grid, color = 'brown')

            plotter.show(screenshot='airplane.png')

    def render(self, ve = 3, name = 'TEST.gif'):
        sy, sx, sz = np.shape(self.strat)
        x = np.linspace(self.xmin, self.xmin + sx * self.dx, sx)
        y = np.linspace(self.ymin, self.ymin + sy * self.dy, sy)
        
        xx, yy = np.meshgrid(x, y)

        zz = self.topo[:,:,0] * ve

        grid = pv.StructuredGrid(xx, yy, zz)

        plotter = pv.Plotter()
        plotter.add_mesh(grid)

        plotter.show(auto_close=False)
        plotter.open_gif(name)

        pts = grid.points.copy()
        
        for i in range(4, sz-1, 4):
            strat = topostrat(self.topo[:,:,0:i+1])
            zz = strat[:,:,i] * ve
            pts[:, -1] = zz.T.ravel()

            plotter.update_coordinates(pts, render=False)

            plotter.write_frame()  # this will trigger the render
            plotter.render()

        plotter.close()

    def export_obj(self, file_name = 'test.obj', ve = 3):
        sy, sx, sz = np.shape(self.strat)
        x = np.linspace(self.xmin, self.xmin + sx * self.dx, sx)
        y = np.linspace(self.ymin, self.ymin + sy * self.dy, sy)
        
        xx, yy = np.meshgrid(x, y)

        zz = self.topo[:,:,0] * ve

        grid = pv.StructuredGrid(xx, yy, zz)

        plotter = pv.Plotter()
        plotter.add_mesh(grid)
        plotter.show(auto_close=False)

        plotter.export_obj(file_name)

    def export(self, ve = 3):
        zz = topostrat_evolution(self.topo)
        np.save("terrain.npy", zz)