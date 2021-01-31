from mpl_toolkits.mplot3d import Axes3D
import bisect
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.interpolate
from scipy.spatial import distance
from scipy import ndimage, stats
from scipy.signal import savgol_filter
from PIL import Image, ImageDraw
from skimage import measure
from skimage import morphology
from matplotlib.colors import LinearSegmentedColormap
import time, sys
import numba
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib import cm
from io import BytesIO
import base64
import pyvista as pv

D2, D1, D0 = 0.0014037355196363848, -0.8514792665395621, 120.08049908837464
LH1, LH2, LH3, LH4 = 106626.10915626335, 8232.473083905528, -88.55157340056991, -0.5999325153044045
OMEGA = -1.0 # constant in curvature calculation (Howard and Knutson, 1984)
GAMMA = 2.5  # from Ikeda et al., 1981 and Howard and Knutson, 1984
K = 1.0 # constant in HK equation
ONE_YEAR = 365*24*60*60.0

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

#@numba.jit(nopython=True) 
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
        SIGMA_2 = np.hstack((np.array([0]),np.cumsum(dl[i-1::-1])))  # distance along centerline, backwards from current point 
        G = np.exp(-2.0 * K * Cf / (d[i] + 1) * SIGMA_2) # convolution vector
        r1[i] = OMEGA*r0[i] + GAMMA*np.sum(r0[i::-1]*G)/np.sum(G) # main equation
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

    def slope(self, ws = 2500):
        slope = np.gradient(self.z, self.x)
        NS = len(self.x)
        sl = np.zeros(NS)

        for i in range(0, NS):
            t = (self.x[i:] - self.x[i]) / ws
            G = np.exp(-t ** 2) 
            sl[i] = np.sum(slope[i:] * G)/np.sum(G)
        
        return sl

    def aggregate(self, density, kv, dt, aggr_factor):
        slope = self.slope()
        K = kv * density * 9.81 * dt
        self.z += K *(slope - aggr_factor*np.mean(slope))

    def incise(self, density, kv, dt):
        slope = self.slope()
        K = kv * density * 9.81 * dt
        self.z += K *slope 

class Channel:
    @classmethod
    def slope2width(cls, slope):
        W2, W1, W0 = 695.4154350661511, -45.231656699536124, 104.60941780103624
        return W2 * np.exp(- W1 * slope) + W0

    @classmethod
    def slope2depth(cls, slope):
        return slope * 100 / np.tan(-5 * np.pi / 180)

    @classmethod
    def depth2migration(cls, depth):
        T = 25
        K = T / (4 * np.log(2))
        return np.where(depth < T, np.exp(-(depth - T) ** 2 / K),1)

    """class for Channel objects"""
    def __init__(self, x, y, z = None, d = None, w = None):
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
        
    def refit(self, basin):
        slope = basin.fit_slope(self.x)

        self.z = basin.fit_elevation(self.x)
        
        self.w  = self.slope2width(slope)
        self.d  = self.slope2depth(slope)

    def resample(self, target_ds):
        _, _, _, _, s = self.derivatives()
        N = 1 + int(round(s[-1]/target_ds))

        tck, _ = scipy.interpolate.splprep([self.x, self.y], s=0)
        u = np.linspace(0,1,N)
        self.x, self.y = scipy.interpolate.splev(u,tck)

    def migrate(self, Cf, kl, dt):
        curv = self.curvature()
        dx, dy, dz, ds, s = self.derivatives()
        sinuosity = s[-1]/(self.x[-1]-self.x[0])
        R0 = kl * self.w * curv
        R1 = compute_migration_rate(R0, Cf, self.d, ds, s[-1])
        RN = sinuosity**(-2/3.0) * R1 * self.depth2migration(self.d)
        
        #plt.plot(self.x, R1);plt.show()
        #plt.plot(self.x, RN);plt.show()
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

    def plot(self, axis = plt, color=sns.xkcd_rgb["ocean blue"], points = False):
        x = self.x
        y = self.y

        xo, yo = self.margin_offset()
        
        xm = np.hstack((x + xo, (x - xo)[::-1]))
        ym = np.hstack((y + yo, (y - yo)[::-1]))
        
        if points:
            axis.plot(x, y, 'o')
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
        return (self.result_width, self.result_height)

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
        z_map = self.create_z_map(basin)
        sl_map = self.create_sl_map(basin)

        hw_inside = cld_map + md_map
        hw_outside = cld_map - md_map

        hw_inside[np.array(np.logical_not(ch_map).astype(bool))] = 0.0
        hw_outside[np.array(ch_map.astype(bool))] = 0.0
        hw_map = hw_inside + hw_outside

        return (ch_map, cld_map, md_map, z_map, sl_map, hw_map)

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

def channel_surface_(ch_map, cld_map, md_map, z_map):
    """
    docstring
    """
    w2 = cld_map + md_map
    d = D2 * w2 ** 2 + D1 * w2 + D0

    w2o = cld_map - md_map
    #levee = np.where((w2o >= 62.5) & (w2o < 100), (w2o - 62.5) / 3.75 , 0) + np.where((w2o >= 100) & (w2o <= 125), (125 - w2o) / 2.5, 0)
    levee = np.where((w2o >= 62.5) & (w2o < 100), (w2o - 62.5) / 3.75, 0) + np.where((w2o >= 100) & (w2o < 125), (125 - w2o) / 2.5, 0)
    levee[np.array(ch_map.astype(bool))] = 0.0
    levee = ndimage.median_filter(levee, size=3)
    levee = levee - md_map / 125
    levee = np.where((levee < 0), 0, levee)

    h_map = d *(cld_map / w2) ** 2 - d 
    h_map[np.array(np.logical_not(ch_map).astype(bool))] = 0.0

    return h_map + levee + z_map

def surface_2(ch_map, cld_map, md_map, z_map, hw_map):
    d = D2 * hw_map ** 2 + D1 * hw_map + D0

    levee = np.where((hw_map >= 62.5) & (hw_map < 100), (hw_map - 62.5) / 3.75, 0) + np.where((hw_map >= 100) & (hw_map < 125), (125 - hw_map) / 2.5, 0)
    levee[np.array(ch_map.astype(bool))] = 0.0
    levee = ndimage.median_filter(levee, size=3)
    levee = levee - md_map / 125
    levee = np.where((levee < 0), 0, levee)

    h_map = d * (cld_map / hw_map) ** 2 - d 
    h_map[np.array(np.logical_not(ch_map).astype(bool))] = 0.0

    return h_map + levee + z_map

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

def erosional_surface(cld_map, z_map, hw_map):
    d = D2 * hw_map ** 2 + D1 * hw_map + D0
    return np.where(d > 0,  d * ((cld_map / hw_map) ** 2 - 1), 0) + z_map

def deposicional_height_map(hw_map):
    return np.clip(LH1 * hw_map ** 3 + LH2 * hw_map ** 2 + LH3 * hw_map + LH4, 0, None)

def gausian_surface(S, cld_map, hw_map):
    return stats.norm.pdf(cld_map / hw_map, scale = S)

def deposional_surface(H, dh_map, cld_map, hw_map):
    return H * dh_map * np.exp(- (cld_map / (4 * hw_map))** 2)

class ChannelEvent:
    '''
        mode: 'LATERAL_MIGRATION' | 'INCISION' | 'AGGRADATION' 
    '''
    def __init__(self, mode = 'LATERAL_MIGRATION', nit = 100, dt = 0.1 * ONE_YEAR, saved_ts = 10, cr_dist = 25, Cf = 0.02, kl = 60.0/ONE_YEAR, kv = 0.01/ONE_YEAR, dens = 1000, aggr_factor = 2):
        self.mode = mode
        self.nit = nit
        self.dt = dt
        self.saved_ts = saved_ts
        self.cr_dist = cr_dist
        self.Cf = Cf
        self.kl = kl
        self.kv = kv
        self.dens = dens
        self.aggr_factor = aggr_factor

class ChannelBelt:
    def __init__(self, channel, basin):
        """
            Times in years.
        """
        self.channels = [channel]
        self.basins = [basin]
        self.times = [0.0]
        self.events = []


        channel.refit(basin)
        _, _, _, ds, _ = channel.derivatives()
        self.ds = np.mean(ds)

    def simulate(self, event):
        channel = self.channels[-1].copy()
        basin = self.basins[-1].copy()

        if len(self.events) == 0:
            self.events.append(event)

        for itn in range(event.nit):
            update_progress(itn/event.nit)

            channel.migrate(event.Cf, event.kl, event.dt)
            channel.cut_cutoffs(event.cr_dist, self.ds)
            channel.resample(self.ds)
            channel.refit(basin)
            
            if event.mode == 'INCISION':
                basin.incise(event.dens, event.kv, event.dt)
            if event.mode == 'AGGREGATION':
                basin.aggregate(event.dens, event.kv, event.dt, event.aggr_factor)

            if itn % event.saved_ts == 0:
                #plt.plot(basin.x, basin.z);plt.show()
                self.times.append((itn+1) * event.dt / ONE_YEAR)
                self.channels.append(channel.copy())
                self.basins.append(basin.copy())
                self.events.append(event)

    def plot(self, start_time=0, end_time = 0):
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
            self.channels[i].plot(axis, color)

        return fig

    def build_3d_model(self, dx):
        
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

        mapper = ChannelMapper(xmin, xmax, ymin * 5, ymax * 5, dx, dx)

        channel = self.channels[0]
        basin = self.basins[0]
        ch_map, cld_map, md_map, z_map, sl_map, hw_map = mapper.create_maps(channel, basin)

        surface = z_map
        last_z_map = z_map

        N = len(self.channels)
        L = 3 + 1

        topo = np.zeros((mapper.rheight, mapper.rwidth, N*L))

        for i in range(0, N):
            update_progress(i/N)
            event = self.events[i]
            ch_map, cld_map, md_map, z_map, sl_map, hw_map = mapper.create_maps(self.channels[i], self.basins[i])

            delta_surface = z_map - last_z_map
            last_z_map = z_map

            inc_map = np.where(delta_surface < 0, delta_surface, 0)
            aggr_map = np.where(delta_surface > 0, delta_surface, 0)

            plt.plot(self.basins[i].x, self.basins[i].z);plt.show()
            #mapper.plot_map(delta_surface)
            #mapper.plot_map(inc_map)
            #mapper.plot_map(aggr_map)
            # DT NORMALIZATION
            dt = self.times[i+1] - self.times[i] if i < N - 1 else self.times[i] - self.times[i-1]
            dh_map = deposicional_height_map(sl_map) * (dt / 2)

            channel_surface = erosional_surface(cld_map, z_map, hw_map) - inc_map
            
            levee1_surface = 1.0 * dh_map * gausian_surface(2, cld_map, hw_map) 
            levee1_surface += 0.3 * aggr_map

            levee2_surface = 2.0 * dh_map * gausian_surface(0.50, cld_map, hw_map)
            levee2_surface += 0.5 * aggr_map
            
            levee3_surface = 0.5 * dh_map * gausian_surface(0.25, cld_map, hw_map)
            levee3_surface += 0.2 * aggr_map

            # CUTTING CHANNEL
            surface = scipy.ndimage.gaussian_filter(np.minimum(surface, channel_surface), sigma = 10 / dx)

            topo[:,:,i*L + 0] = surface

            # DEPOSITING SEDIMENT
            surface = surface + levee3_surface
            topo[:,:,i*L + 1] = surface
            surface = surface + levee2_surface
            topo[:,:,i*L + 2] = surface
            surface = surface + levee1_surface
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

    def plot_xsection(self, xsec, ve = 5, substrat = True):
        strat = self.strat
        sy, sx, sz = np.shape(strat)
        xsec = int(xsec * sx)

        fig1 = plt.figure(figsize=(20,5))
        ax1 = fig1.add_subplot(111)

        Xv = np.linspace(self.ymin, self.ymin + sy * self.dy, sy)
        X1 = np.concatenate((Xv, Xv[::-1]))
        
        if substrat:
            Yb = np.ones(sy) * self.zmin
            ax1.fill(X1, np.concatenate((Yb, strat[::-1,xsec,0])), facecolor=[192/255, 192/255, 192/255])
        
        for i in range(0, sz, 4):
            Y1 = np.concatenate((strat[:,xsec,i],   strat[::-1,xsec,i+1])) 
            Y2 = np.concatenate((strat[:,xsec,i+1], strat[::-1,xsec,i+2]))
            Y3 = np.concatenate((strat[:,xsec,i+2], strat[::-1,xsec,i+3]))

            ax1.fill(X1, Y1, facecolor=[255/255, 102/255, 0/255],linewidth=0.1, edgecolor='r') # oxbow mud
            ax1.fill(X1, Y2, facecolor=[255/255, 204/255, 0/255]) # levee mud
            ax1.fill(X1, Y3, facecolor=[51/255, 51/255, 0]) # levee mud
        
        ax1.set_xlim(self.ymin, self.ymin + sy * self.dy)
        ax1.set_aspect(ve, adjustable='datalim')

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