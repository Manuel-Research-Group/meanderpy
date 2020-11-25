import bisect
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.interpolate
from scipy.spatial import distance
from scipy import ndimage
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
import pyvista as pv

D2, D1, D0 = 0.0013265249206961816, -0.8297607727804712, 118.89179709531096


OMEGA = -1.0 # constant in curvature calculation (Howard and Knutson, 1984)
GAMMA = 2.5  # from Ikeda et al., 1981 and Howard and Knutson, 1984
K = 1.0 # constant in HK equation

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
        G = np.exp(-2.0 * K * Cf / max(d[i], 1) * SIGMA_2) # convolution vector
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

class Channel:
    @classmethod
    def slope2width(cls, slope):
        W2, W1, W0 = 695.4154350661511, -45.231656699536124, 104.60941780103624
        return np.clip(W2 * np.exp(- W1 * slope) + W0, 100, 800)

    @classmethod
    def slope2depth(cls, slope):
        D2, D1, D0 = 35.903468691717954, 15.06078709431349, -35.96222916210254
        return np.clip(D2 * np.exp(- D1 * slope) + D0, 100, -5)

    @classmethod
    def slope2migration(cls, slope):
        ARCTG_2 = -0.035
        K = ARCTG_2 ** 2 / (4 * np.log(2))
        return np.where(slope > ARCTG_2, np.exp(-(slope - ARCTG_2) ** 2 / K),1)

    """class for Channel objects"""
    def __init__(self,x,y,z,w,d):
        """initialize Channel object
        x, y, z  - coordinates of centerline
        W - channel width
        D - channel depth"""
        self.x = x
        self.y = y
        self.z = z
        self.w = w
        self.d = d

    def copy(self):
        return Channel(self.x.copy(), self.y.copy(), self.z.copy(), self.w.copy(), self.d.copy())

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
        ds = np.sqrt(dx**2+dy**2+dz**2)
        s = np.hstack((0,np.cumsum(ds[1:])))

        return dx, dy, dz, ds, s

    def curvature(self):
        dx = np.gradient(self.x) 
        dy = np.gradient(self.y)      
        
        ddx = np.gradient(dx)
        ddy = np.gradient(dy) 

        return (dx*ddy-dy*ddx)/((dx**2+dy**2)**1.5)

    def refit(self, target_ds):
        _, _, dz, ds, _ = self.derivatives()
        
        slope = savgol_filter(dz / ds, 51, 3)
        plt.plot(self.x, slope)
        self.w  = self.slope2width(slope)
        self.d  = self.slope2depth(slope)


    def resample(self, target_ds):
        _, _, _, _, s = self.derivatives()
        
        tck, _ = scipy.interpolate.splprep([self.x,self.y,self.z,self.w,self.d],s=0) 
        u = np.linspace(0,1,1+int(round(s[-1]/target_ds)))
        self.x, self.y, self.z, self.w, self.d = scipy.interpolate.splev(u,tck) 

    def migrate(self,Cf,kl,dt):
        curv = self.curvature()
        dx, dy, dz, ds, s = self.derivatives()
        slope = dz / ds
        sinuosity = s[-1]/(self.x[-1]-self.x[0])
        R0 = kl * self.w * curv
        R1 = compute_migration_rate(R0, Cf, self.d, ds, s[-1])
        RN = sinuosity**(-2/3.0) * R1 * self.slope2migration(slope)
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

    def plot(self, axis = plt, color=sns.xkcd_rgb["ocean blue"]):
        x = self.x
        y = self.y

        xo, yo = self.margin_offset()
        
        xm = np.hstack((x + xo, (x - xo)[::-1]))
        ym = np.hstack((y + yo, (y - yo)[::-1]))
        axis.fill(xm, ym, color=color, edgecolor='k', linewidth=0.25)
        #axis.plot(x, y)

class ChannelMapper:
    def __init__(self, xmin, xmax, ymin, ymax, xsize, ysize):
        self.xmin = xmin
        self.ymin = ymin
        
        #grid size
        self.xsize = xsize
        self.ysize = ysize

        self.dx = xmax - xmin
        self.dy = ymax - ymin

        self.width = int(self.dx / self.xsize)
        self.height = int(self.dy / self.ysize)

    def __repr__(self):
        return 'GRID-SIZE: ({};{})\nIMAGE-SIZE: ({};{})\n PIXELS: {}'.format(self.xsize, self.ysize, self.width, self.height, self.width * self.height)

    def create_maps(self, channel):
        ch_map = self.create_ch_map(channel)
        
        cld_map = self.create_cld_map(channel)
        md_map = self.create_md_map(channel)
        z_map = self.create_z_map(channel)

        hw_inside = cld_map + md_map
        hw_outside = cld_map - md_map

        hw_inside[np.array(np.logical_not(ch_map).astype(bool))] = 0.0
        hw_outside[np.array(ch_map.astype(bool))] = 0.0
        hw_map = hw_inside + hw_outside

        return (ch_map, cld_map, md_map, z_map, hw_map)

    def create_md_map(self, channel):
        xo, yo = channel.margin_offset()

        upper_pixels = self.to_pixels(channel.x + xo, channel.y + yo)
        lower_pixels = self.to_pixels(channel.x - xo, channel.y - yo)

        img = Image.new("1", (self.width, self.height), 1)
        draw = ImageDraw.Draw(img)
        draw.line(upper_pixels, fill=0) 
        draw.line(lower_pixels, fill=0)

        md_map = ndimage.distance_transform_edt(np.array(img), sampling=[self.xsize, self.ysize])

        return md_map

    def create_cld_map(self, channel):
        pixels = self.to_pixels(channel.x, channel.y)
        img = Image.new("1", (self.width, self.height), 1)
        draw = ImageDraw.Draw(img)
        draw.line(pixels, fill=0)
    
        cld_map = ndimage.distance_transform_edt(np.array(img), sampling=[self.xsize, self.ysize])

        return cld_map

    def create_ch_map(self, channel):
        x, y = channel.x, channel.y
        xo, yo = channel.margin_offset()

        xm = np.hstack((x + xo, (x - xo)[::-1]))
        ym = np.hstack((y + yo, (y - yo)[::-1]))

        xy = self.to_pixels(xm, ym)

        img = Image.new("1", (self.width, self.height), 0)
        draw = ImageDraw.Draw(img)
        draw.polygon(xy, fill=1)

        return np.array(img)

    def create_z_map(self, channel):
        x_p = ((channel.x - self.xmin) / self.dx) * self.width

        tck, _ = scipy.interpolate.splprep([x_p, channel.z], s = 0)
        u = np.linspace(0,1,self.width)
        _, z_level = scipy.interpolate.splev(u, tck)

        return np.tile(z_level, (self.height, 1))

    def to_pixels(self, x, y):
        x_p = ((x - self.xmin) / self.dx) * self.width
        y_p = ((y - self.ymin) / self.dy) * self.height

        xy = np.vstack((x_p, y_p)).astype(int).T
        return tuple(map(tuple, xy))

def channel_surface(ch_map, cld_map, md_map, z_map):
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

def surface(ch_map, cld_map, md_map, z_map, hw_map):
    d = D2 * hw_map ** 2 + D1 * hw_map + D0

    levee = np.where((hw_map >= 62.5) & (hw_map < 100), (hw_map - 62.5) / 3.75, 0) + np.where((hw_map >= 100) & (hw_map < 125), (125 - hw_map) / 2.5, 0)
    levee[np.array(ch_map.astype(bool))] = 0.0
    levee = ndimage.median_filter(levee, size=3)
    levee = levee - md_map / 125
    levee = np.where((levee < 0), 0, levee)

    h_map = d * (cld_map / hw_map) ** 2 - d 
    h_map[np.array(np.logical_not(ch_map).astype(bool))] = 0.0

    return h_map + levee + z_map

class ChannelBelt:
    """class for ChannelBelt objects"""
    def __init__(self, channel):
        """initialize ChannelBelt object
        channels - list of Channel objects
        cutoffs - list of Cutoff objects
        cl_times - list of ages of Channel objects
        cutoff_times - list of ages of Cutoff objects"""
        self.channels = [channel]
        self.times = [0.0]

    def migrate(self,nit,saved_ts,ds,pad,crdist,Cf,kl,kv,dt,dens,t1,t2,t3,aggr_factor,*D):
        channel = self.channels[-1].copy()
        last_cl_time = 0

        for itn in range(nit):
            update_progress(itn/nit)
            #print(channel.w)
            #channel.refit(ds)
            
            channel.migrate(Cf,kl,dt)
            channel.cut_cutoffs(25,ds)
            channel.resample(ds)
            
            if itn % saved_ts == 0:
                #channel.refit(ds)
                self.times.append(last_cl_time+(itn+1)*dt/(365*24*60*60.0))
                self.channels.append(channel.copy())

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

    def image_test(self, dx, downscale = 1):
        channel = self.channels[0]
        xmax, xmin = max(channel.x), min(channel.x)
        ymax, ymin = max(channel.y), min(channel.y)

        mapper = ChannelMapper(xmin, xmax, ymin * 2, ymax * 2, dx / downscale, dx / downscale)

        print(mapper, int(mapper.width/downscale) * int(mapper.height/downscale))

        ch_map, cld_map, md_map, z_map, hw_map = mapper.create_maps(channel)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(hw_map, interpolation='nearest')
        fig.colorbar(cax)
        plt.show()

        surf = surface(ch_map, cld_map, md_map, z_map, hw_map)
        surf = np.array(Image.fromarray(surf).resize((int(mapper.width/downscale), int(mapper.height/downscale)), Image.BILINEAR))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(surf, interpolation='nearest')
        fig.colorbar(cax)
        plt.show()

        x = np.linspace(xmin, xmax, surf.shape[1])
        y = np.linspace(ymin, ymax, surf.shape[0])

        xx, yy = np.meshgrid(x, y)

        grid = pv.StructuredGrid(xx, yy, 5*surf)

        grid.plot()