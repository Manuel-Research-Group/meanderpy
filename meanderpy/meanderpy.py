from cmath import nan
from zipfile import ZipFile
import tempfile
from os import path, walk
from shutil import copyfile
import trimesh
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
from matplotlib.lines import Line2D
import time, sys
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import math
import struct
import os

OMEGA = -1.0 # constant in curvature calculation (Howard and Knutson, 1984)
GAMMA = 2.5  # from Ikeda et al., 1981 and Howard and Knutson, 1984
K = 4.0 # constant in HK equation
YEAR = 365*24*60*60.0

# NUMBER_OF_LAYERS_PER_EVENT: number of materials plus 1. This extra 1 comes from the eroded surface before aggradation.
# We then deposit (aggradate) a certain number of materials. For instance, when using gravel, sand and silt, the number of materials == 3.
# When using gravel, gross sand, medium sand, fine sand, and silt, then number of materials == 5.
# When using gravel, very gross sand, gross sand, medium sand, fine sand, very fine sand, and silt, then number of materials == 7.
NUMBER_OF_LAYERS_PER_EVENT = 9 # there are 7 layers + previous surface + separator
BIG_NUMBER = 10**9

def update_progress(progress):
    """
    Displays or updates a console progress bar.
    Adapted from https://stackoverflow.com/questions/3160699/python-progress-bar

    :param progress: (float) value 0 represents halt and 1 represents 100%.    
    """

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
    text = "\rPercent: [{}] {:.2f}% {}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

def compute_migration_rate(r0, Cf, d, dl, L):
    """
    Compute migration rate as weighted sum of upstream curvatures.

    pad - padding (number of nodepoints along centerline)
    ns - number of points in centerline
    ds - distances between points in centerline
    omega - constant in HK model
    gamma - constant in HK model

    :param r0: nominal migration rate (dimensionless curvature * migration rate constant)
    :param Cf: TODO
    :param d: TODO
    :param dl: TODO
    :param L: TODO
    :return: TODO
    """

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
    """
    Function for identifying locations of cutoffs along a centerline
    and the indices of the segments that will become part of the oxbows [Sylvester]

    :param x: coordinate x of centerline
    :param y: coordinate y of centerline
    :param crdist: critical cutoff distance
    :param diag: TODO   

    :return: TODO
    """

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
    """
    TODO

    :param R: TODO
    :param W: TODO
    :param T: TODO
    :return: TODO
    """

    indexes = np.where(np.abs(R) > T)[0][-1:]

    if len(indexes) == 0:
        return -1, -1

    ind1, ind2 = indexes[0] - W, indexes[0] + W
    
    for i in indexes:
        if i > ind1:
            ind1 = i - W
        
    return max(ind1, 0), min(ind2, len(R) -1)

def zipFilesInDir(dirName, zipFileName, filter):
    """
    Compress all the files in a given directory to a zip file.

    :param dirName: (string) input name of the directory to be zipped.
    :param zipFileName: (string) output name of the zip generated.
    :param filter: TODO    
    """

    with ZipFile(zipFileName, 'w') as zipObj:
        for (folderName, _, filenames) in walk(dirName):
            for filename in filenames:
                if filter(filename):
                    filePath = path.join(folderName, filename)
                    zipObj.write(filePath, filename)


class Basin:   
    """
    TODO
    """

    def __init__(self, x, z):
        """
        Inits Basin with x and z.

        :param x: (float) x coordinate.
        :param z: (float) z coordinate.
        """

        self.x = x
        self.z = z

    def copy(self):
        """
        Copies the Basin object.

        :return: (Basin) copy of Basin.
        """

        return Basin(self.x.copy(), self.z.copy())

    def fit_elevation(self, x):
        """
        TODO

        :param x: (float) TODO
        :return: TODO
        """

        return scipy.interpolate.interp1d(self.x, self.z, kind='cubic', fill_value='extrapolate')(x)

    def fit_slope(self, x, ws = 2500):
        """
        TODO

        :param x: TODO
        :param ws: (int) TODO
        :return: TODO
        """

        return scipy.interpolate.interp1d(self.x, self.slope(ws), kind='cubic', fill_value='extrapolate')(x)

    def slope(self, ws = 2500, degrees = True):
        """
        TODO

        :param ws: TODO
        :param degrees: (bool) TODO
        :return: TODO
        """

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
        """
        TODO

        :param density: water density in kg/m³ = 1000 kg/m³
        :param kv: aggradation/incision modulation (m/years).
        :param dt: time of each iteration (in years).
        :param aggr_factor: aggradation factor. According to Sylvester (meanderpy.py, line 201) it should be larger than 1 so that this leads to overall aggradation
        """

        slope = self.slope(degrees=False)
        K = kv * density * 9.81 * dt
        self.z += K *(slope - aggr_factor*np.mean(slope)) # Sylvester's Method
        #self.z += K *(slope - aggr_factor*np.min(slope)) # NEW METHOD

        #print('-----------------')        
        #print('aggr_factor: ', aggr_factor)
        #print('mean(slope): ', np.mean(slope))
        #print('slope: ', slope)
        #print('self.z: ', self.z)
        #print('-----------------')

        #self.z += -K* slope # BEUREN'S SUGGESTION

        #self.z += -K* (slope + aggr_factor*np.mean(slope)) # NEW EXPERIMENTAL METHOD

    def incise(self, density, kv, dt):
        """
        TODO

        :param density: water density in kg/m³ = 1000 kg/m³
        :param kv: aggradation/incision modulation (m/years).
        :param dt: time of each iteration (in years).        
        """

        slope = self.slope(degrees=False)
        K = kv * density * 9.81 * dt
        self.z += K *slope  # z decresce

    def plot(self, axis = plt, color=sns.xkcd_rgb["ocean blue"], points = False):
        """
        TODO

        :param axis: TODO
        :param color: TODO [unused]
        :param points: TODO [unused]     
        """

        axis.plot(self.x, self.z)

class Channel:
    """
    Class for Channel objects.
    """

    def __init__(self, x, y, z = [], d = [], w = []):
        """
        Initialize channel object.

        :param x: x-coordinate of centerline.
        :param y: y-coordinate of centerline.
        :param z: z-coordinate of centerline.
        :param d: channel depth.
        :param w: channel width.
        """

        self.x = x
        self.y = y
        self.z = z
        self.d = d
        self.w = w

    def copy(self):
        """
        Copies the Channel object.

        :return: (Channel) copy of Channel.
        """

        return Channel(self.x.copy(), self.y.copy(), self.z.copy(), self.d.copy(), self.w.copy())

    def margin_offset(self):
        """
        TODO

        :return: TODO
        """

        d = self.w / 2

        dx = np.gradient(self.x)
        dy = np.gradient(self.y)

        n = np.stack((dy, -dx))
        l = np.sqrt(np.sum(np.conj(n) * n, axis = 0))
            
        xo = d * (  dy / l )
        yo = d * ( -dx / l )

        return xo, yo

    def derivatives(self):
        """
        TODO

        :return: TODO
        """

        dx = np.gradient(self.x)
        dy = np.gradient(self.y)
        dz = np.gradient(self.z)
        ds = np.sqrt(dx**2 + dy**2 + dz**2)
        s = np.hstack((0,np.cumsum(ds[1:])))

        return dx, dy, dz, ds, s

    def curvature(self):
        """
        TODO

        :return: TODO
        """

        dx = np.gradient(self.x) 
        dy = np.gradient(self.y)      
        
        ddx = np.gradient(dx)
        ddy = np.gradient(dy) 

        return (dx*ddy-dy*ddx)/((dx**2+dy**2)**1.5)
        
    def refit(self, basin, ch_width, ch_depth):
        """
        TODO

        :param basin:
        :param ch_width: 
        :param ch_depth: channel depth in meters
        :return: TODO
        """

        slope = basin.fit_slope(self.x)

        self.z = basin.fit_elevation(self.x)
        self.w  = ch_width(slope)
        self.d  = ch_depth(slope)

    def resample(self, target_ds):
        """
        TODO

        :param target_ds: TODO
        """

        _, _, _, _, s = self.derivatives()
        N = 1 + int(round(s[-1]/target_ds))

        tck, _ = scipy.interpolate.splprep([self.x, self.y], s=0)
        u = np.linspace(0,1,N)
        self.x, self.y = scipy.interpolate.splev(u,tck)

    def migrate(self, Cf, kl, dt):
        """
        Method for computing migration rates along channel centerlines and moving the centerlines accordingly. [Sylvester]

        :param Cf: array of dimensionless Chezy friction factors (can vary across iterations) [Sylvester]
        :param kl: migration rate constant (m/s)
        :param dt: time step (s) [Sylvester] (?)
        """

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
        """
        TODO

        :param crdist: TODO
        :param ds: TODO  
        :return: TODO              
        """

        #print('inside cut_cutoffs')   
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
        """
        TODO

        :param cut_window: TODO
        :param ds: TODO
        """

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
        """
        Method  for plotting ChannelBelt object. [Sylvester]

        :param axis: TODO
        :param color: TODO
        :param points: TODO
        """

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

# xmin, xmax: min e max do grid
# xsize, ysize: dimensões
# mapas 2D que serão transformados em 3D (usado depois para as gaussian surfaces)
# downscale para geração dos mapas
class ChannelMapper:
    """
    Transforms 2D maps to 3D to be used for the gaussian surfaces

    :param crdist: TODO
    :param ds: TODO                
    """

    def __init__(self, xmin, xmax, ymin, ymax, xsize, ysize, downscale = 4, sigma = 2):
        """
        Initialize channel mapper.

        :param xmin: TODO
        :param xmax: TODO
        :param ymin: TODO
        :param ymax: TODO
        :param xsize: grid size in x axis.
        :param ysize: grid size in y axis.
        :param downscale: TODO
        :param sigma: TODO
        """

        self.xmin = xmin
        self.ymin = ymin
        
        self.downscale = downscale
        self.sigma = sigma
        
        self.xsize = int(xsize / downscale)
        self.ysize = int(ysize / downscale)

        self.dx = xmax - xmin
        self.dy = ymax - ymin

        self.width = int(self.dx / self.xsize)
        self.height = int(self.dy / self.ysize)

        self.rwidth = int(self.width/self.downscale)
        self.rheight = int(self.height/self.downscale)

    def __repr__(self):
        """
        TODO

        :return: (string) Information regarding grid size, image size and number of pixels.
        """

        return 'GRID-SIZE: ({};{})\nIMAGE-SIZE: ({};{})\n PIXELS: {}'.format(self.xsize, self.ysize, self.width, self.height, self.width * self.height)

    def map_size(self):
        """
        [DEBUG] TODO

        :return: TODO
        """

        return (self.xsize, self.ysize)

    def post_processing(self, _map):
        """
        TODO

        :param _map: TODO
        :return: TODO
        """

        return self.downsize(self.filter(_map))

    def filter(self, _map):
        """
        TODO

        :param _map: TODO
        :return: TODO
        """

        return scipy.ndimage.gaussian_filter(_map, sigma = self.sigma)

    def downsize(self, _map):
        """
        TODO

        :param _map: TODO
        :return: TODO
        """

        return np.array(Image.fromarray(_map).resize((self.rwidth, self.rheight), Image.BILINEAR))

    def create_maps(self, channel, basin):
        """
        TODO

        :param channel
        :param basin
        :return: TODO
        """

        ch_map = self.create_ch_map(channel)
        # MANUEL's comments on these channel maps
        cld_map = self.create_cld_map(channel)    # cld: centerline distance - distance from a point to the channel's centerline.
        md_map = self.create_md_map(channel)      # md: margin distance - distance from a point to the channel's closest margin.
        cz_map = self.create_z_map(channel)
        bz_map = self.create_z_map(basin)
        sl_map = self.create_sl_map(basin)

        hw_inside = cld_map + md_map
        hw_outside = cld_map - md_map

        hw_inside[np.array(np.logical_not(ch_map).astype(bool))] = 0.0
        hw_outside[np.array(ch_map.astype(bool))] = 0.0
        hw_map = hw_inside + hw_outside            # hw: half width - half width of the channel (stored in all pixels, but it seems to stored 
                                                   # values considering a cross section of the channel considering the closest point 
                                                   # where the margin is perpendicular to the vector formed by that point and the pixel 
                                                   # that stores the corresponding value)

        return (ch_map, cld_map, md_map, cz_map, bz_map, sl_map, hw_map)

    def create_md_map(self, channel):
        """
        Create margin distance

        :param channel: TODO        
        :return: TODO
        """

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
        """
        Create centerline distance map.

        :param channel: TODO        
        :return: TODO
        """

        pixels = self.to_pixels(channel.x, channel.y)
        img = Image.new("1", (self.width, self.height), 1)
        draw = ImageDraw.Draw(img)
        draw.line(pixels, fill=0)
    
        cld_map = ndimage.distance_transform_edt(np.array(img), sampling=[self.xsize, self.ysize])

        return self.post_processing(cld_map)

    def create_ch_map(self, channel):
        """
        TODO

        :param channel: TODO        
        :return: TODO
        """

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
        """
        TODO

        :param basin: TODO      
        :return: TODO
        """

        x_p = ((basin.x - self.xmin) / self.dx) * self.width

        tck, _ = scipy.interpolate.splprep([x_p, basin.z], s = 0)
        u = np.linspace(0,1,self.width)
        _, z_level = scipy.interpolate.splev(u, tck)

        return self.post_processing(np.tile(z_level, (self.height, 1)))

    def create_sl_map(self, basin):
        """
        TODO

        :param basin: TODO
        :return: TODO
        """

        x_p = ((basin.x - self.xmin) / self.dx) * self.width
        
        tck, _ = scipy.interpolate.splprep([x_p, basin.slope()], s = 0)
        u = np.linspace(0,1,self.width)
        _, z_level = scipy.interpolate.splev(u, tck)

        return self.post_processing(np.tile(z_level, (self.height, 1)))

    def plot_map(self, _map):
        """
        [DEBUG] Debug method to plot the map as a colorbar.

        :param _map: TODO
        """

        plt.matshow(_map)
        plt.colorbar()
        plt.show()

    def to_pixels(self, x, y):
        """
        Method to plot the map as a colorbar.

        :param x: TODO
        :param y: TODO
        :return: TODO
        """

        x_p = ((x - self.xmin) / self.dx) * self.width
        y_p = ((y - self.ymin) / self.dy) * self.height

        xy = np.vstack((x_p, y_p)).astype(int).T
        return tuple(map(tuple, xy))

def topostrat(topo, N = None):
    """
    Function for converting a stack of geomorphic surfaces into stratigraphic surfaces.

    :param topo: 3D numpy array of geomorphic surfaces
    :param N: TODO
    :return: strat - 3D numpy array of stratigraphic surfaces
    """    

    r,c,ts = np.shape(topo)    

    T = N if N is not None else ts # added to solve the incision problem    

    strat = np.copy(topo)    
    #print('(D) topo: ', topo[:,:,-2])    
    for i in (range(0,T)): # the layer 0 is the bottom one
        strat[:,:,i] = np.amin(topo[:,:,i:], axis=2)        
    #print('(E) strat: ', strat)
    return strat # matriz com todos os pontos (armazenado valor do z mínimo)

def plot2D(x, y, title, ylabel, fileName, save=True):
    """
    Plots in a Matplotlib graph the x and y array values.
    
    :param x: vector containing the x elements.
    :param y: vector containing the y elements.
    :param title: title of the plot.
    :param ylabel: label of the y axis.
    """
    
    plt.plot(x, y)        
    plt.title(title)
    plt.xlabel('Length (m)')
    plt.ylabel(ylabel)
    plt.ylim(-50, 500)
    if (save):
        plt.savefig(fileName)
        plt.clf()
    else:
        plt.show()

def topostrat_evolution(topo):
    """
    Function for converting a stack of geomorphic surfaces into stratigraphic surfaces.

    :param topo: 3D numpy array of geomorphic surfaces
    :return: strat: 3D numpy array of stratigraphic surfaces
    """
    N = 4
    r,c,ts = np.shape(topo)
    strat = np.zeros((r,c,int(ts/N)))
    for i in (range(0,ts, N)):
        strat[:,:,int((i+1)/N)] = np.amin(topo[:,:,i:i+N], axis=2)
    return strat

def plot3D(Z, grid_size = 1):
    """
    TODO

    :param Z: TODO
    :param grid_size: TODO
    :return: TODO
    """

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
    """
    TODO

    :param cld_map: centerline distance map
    :param z_map: represents the current z level of the basin
    :param hw_map: map that relates each pixel to its closer channel width that is called the half-width
    :param cd_map: the channel depth at each location which multiplies the terms that creates the parabolic formation
    :return: TODO
    """

    return cd_map * ((cld_map / hw_map) ** 2 - 1) + z_map

def gaussian_surface(sigma_map, cld_map, hw_map):
    """
    Calculates the gaussian 

    :param sigma_map: TODO
    :param cld_map: centerline distance map
    :param hw_map: map that relates each pixel to its closer channel width that is called the half-width
    :return: TODO
    """

    return np.exp(- 1 / 2 * ((cld_map / hw_map) / sigma_map) ** 2)

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
        
        self.dens = dens
        self.aggr_factor = aggr_factor
        self.start_time = -1
    
    
    # DEBUG methods
    '''
    def plot_ch_depth(self, slope = np.linspace(-5, 0, 20), axis = None):
        """
        [DEBUG] Plots the channel depth for debugging.

        :param slope: TODO
        :param axis: TODO
        :return: figure which can be shown by the matplotlib.
        """

        if axis is None:
            fig, axis = plt.subplots(1, 1)
        else:
            fig = None

        axis.set_xlabel('Slope(°)')
        axis.set_ylabel('Channel Depth')
        axis.plot(slope, self.ch_depth(slope))
        return fig

    def plot_ch_width(self, slope = np.linspace(-5, 0, 20), axis = None):
        """
        [DEBUG] Plots the channel width for debugging.

        :param slope: TODO
        :param axis: TODO
        :return: figure which can be shown by the matplotlib.
        """

        if axis is None:
            fig, axis = plt.subplots(1, 1)
        else:
            fig = None
            
        axis.set_xlabel('Slope(°)')
        axis.set_ylabel('Channel Width(m)')
        axis.plot(slope, self.ch_width(slope))
        return fig

    def plot_dep_height(self, slope = np.linspace(-5, 0, 20), axis = None):
        """
        [DEBUG] Plots the channel width for debugging.

        :param slope: TODO
        :param axis: TODO
        :return: figure which can be shown by the matplotlib.
        """

        if axis is None:
            fig, axis = plt.subplots(1, 1)
        else:
            fig = None
            
        axis.set_xlabel('Slope(°)')
        axis.set_ylabel('Deposition Height(m)')
        axis.plot(slope, self.dep_height(slope))
        return fig

    def plot_dep_props(self, slope = np.linspace(-5, 0, 20), axis = None):
        """
        [DEBUG] Plots the deposition proportions of the material filling the deposition depth.

        :param slope: TODO
        :param axis: TODO
        :return: figure which can be shown by the matplotlib.
        """

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
        """
        [DEBUG] Plots the deposition sigmas for debugging.

        :param slope: TODO
        :param axis: TODO
        :return: figure which can be shown by the matplotlib.
        """

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
        """
        [DEBUG] Plots the aggradation proportions for debugging.

        :param slope: TODO
        :param axis: TODO
        :return: figure which can be shown by the matplotlib.
        """

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
        """
        [DEBUG] Plots the aggradation sigmas for debugging.

        :param slope: TODO
        :param axis: TODO
        :return: figure which can be shown by the matplotlib.
        """

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
        """
        [DEBUG] Plots all the previous plot methods together.
        
        :return: figure which can be shown by the matplotlib.
        """

        fig, axes = plt.subplots(4, 2)

        self.plot_ch_depth(axis = axes[0][0])
        self.plot_ch_width(axis = axes[0][1])
        self.plot_dep_height(axis = axes[1][0])
        self.plot_dep_props(axis = axes[2][0])
        self.plot_dep_sigmas(axis = axes[2][1])
        self.plot_aggr_props(axis = axes[3][0])
        self.plot_aggr_sigmas(axis = axes[3][1])

        return fig
    '''

class ChannelBelt:
    """
    Class for ChannelBelt objects.
    """

    def __init__(self, channel, basin):
        """
        Initializes the ChannelBelt object. Times in years.

        :param channel: list of Channel objects [Sylvester]
        :param basin: TODO        
        """

        self.channels = [channel.copy()]        
        self.basins = [basin.copy()]
        self.times = [0.0]
        self.events = []

    # 2 progress bars: meandering + modeling
    # essa parte é simulação 2D
    def simulate(self, event, eventOrder): # parte 2D
        """
        TODO

        :param event: event list. 
        :param eventOrder: order of the event in the event list. Use to allow saving all intermediate channel and basin profiles.            
        """        
        
        last_time = self.times[-1]
        event.start_time = last_time + event.dt        

        if len(self.events) == 0:
            channel = self.channels[0]
            basin = self.basins[0]
            self.events.append(event) #create a "base event" which is a copy of the first one (need to be confirmed if necessary)
            channel.refit(basin, event.ch_width, event.ch_depth)
            _, _, _, ds, _ = channel.derivatives()
            self.ds = np.mean(ds)
            event.start_time = 0

        channel = self.channels[-1].copy()
        basin = self.basins[-1].copy()
        last_time = self.times[-1]
        
        for itn in range(1, event.nit+1):            
            update_progress(itn/event.nit)    

            #plot2D(channel.x, channel.y, 'Channel Preview', 'Width (m)') # need to update the call
            #plot2D(basin.x, basin.z, 'Basin Preview', 'Elevation (m)') # need to update the call

            channel.migrate(event.Cf, event.kl / YEAR, event.dt * YEAR)            
            channel.cut_cutoffs(event.cr_dist, self.ds)            
            channel.cut_cutoffs_R(event.cr_wind, self.ds)            
            channel.resample(self.ds) # deixar curva smooth e deixa dl (comprimento de largura da curva) constante              
            channel.refit(basin, event.ch_width, event.ch_depth) # avaliação da elevação (z) do canal a cada ponto com base na bacia              
            
            if event.mode == 'INCISION':
                print('INCISION!!!')
                basin.incise(event.dens, event.kv / YEAR, event.dt * YEAR)
            if event.mode == 'AGGRADATION':
                print('AGGRADATION!!!')
                basin.aggradate(event.dens, event.kv / YEAR, event.dt * YEAR, event.aggr_factor)
            # TODO Dennis: SEPARATE must be included as a function from basin            
            if event.mode == 'SEPARATOR':
                print('SIMULATE/SEPARATOR')    

            # número de canais = time stamp
            if (itn % event.saved_ts == 0 and (event.mode == 'INCISION' or event.mode == 'AGGRADATION')):
                self.times.append(last_time + (itn+1) * event.dt)
                self.channels.append(channel.copy())
                self.basins.append(basin.copy())
                self.events.append(event)
                # Used to create the gif files containing the basin animation
                plot2D(basin.x, basin.z, 'Basin Preview', 'Elevation (m)', 'basin_' + str(eventOrder) +'-' + str(itn) + '.png', save=True) # vista lateral
                #plot2D(channel.x, channel.y, 'Channel Preview', 'Elevation (m)', 'channel_' + str(eventOrder) + '-' + str(itn) + '.png', save=True) # vista superior
            
        # DENNIS: saves a single layer of separator
        if event.mode == 'SEPARATOR':
            self.times.append(last_time + event.dt)
            self.channels.append(channel.copy())
            self.basins.append(basin.copy())
            self.events.append(event)   

        # dgb: Save the final mesh
        '''
        print("\nFinal mesh: ", itn, " <space>.")
        self.times.append(last_time + (itn+1) * event.dt)
        self.channels.append(channel.copy())
        self.basins.append(basin.copy())
        self.events.append(event)
        '''        
        
        #print("\n#times: ", len(self.times), " <space>.")

    def plot_basin(self, evolution = True):
        """
        TODO

        :param evolution: TODO
        :return: TODO
        """

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
        """
        TODO

        :param start_time: TODO
        :param end_time: TODO
        :param points: TODO
        :return: TODO
        """

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

    def build_3d_model(self, dx, margin = 500): # recebe lista de bacias e lista de canais
        """
        TODO

        :param dx: grid generated from dxdy variable (coming from JSON "Map Scale"). See initialization of grid variable in runner.py
        :param margin: margin informed as "Channel Padding" in the "Channel Configuration" interface
        :return: TODO
        """

        #print('LEN: ', len(self.events))
        #print('MODE: ', self.events[0].mode)
        #print('DEP PROPS: ', self.events[0].dep_props)
        #print('MODE: ', self.events[1].mode)
        #print('DEP PROPS: ', self.events[1].dep_props)

        xmax, xmin, ymax, ymin = [], [], [], []
        for channel in self.channels: # um canal para cada snapshot. Cada passo gera 4 malhas
            xmax.append(max(channel.x))
            xmin.append(min(channel.x))
            ymax.append(max(channel.y))
            ymin.append(min(channel.y))

        xmax = max(xmax)
        xmin = min(xmin)
        ymax = max(ymax)
        ymin = min(ymin)

        # cria mapas
        mapper = ChannelMapper(xmin + margin, xmax - margin, ymin - margin, ymax + margin, dx, dx)

        channel = self.channels[0] #canais 2D vista superior
        basin = self.basins[0] # bacia 2D vista lateral
        ch_map, cld_map, md_map, cz_map, bz_map, sl_map, hw_map = mapper.create_maps(channel, basin)

        # surface: resultado atual do processo de corte e deposição (cut and fill)
        surface = bz_map # bz_map: altura do centro do canal explodido lateralmente

        N = len(self.channels) 
        L = NUMBER_OF_LAYERS_PER_EVENT # atualizar

        topo = np.zeros((mapper.rheight, mapper.rwidth, N*L))

        #print("numero canais (N): ", N)

        for i in range(0, N):
            update_progress(i/N)

            event = self.events[i]            
            # Last iteration 
            # aggr_map: qual parte do terreno está sofrendo aggradation
            # surface: parte mais superior computada
            aggr_map = bz_map - surface 
            aggr_map[aggr_map < 0] = 0            

            # channel, centerline distance, channel z, basin z, slope, half width
            # ch_map:
            ch_map, cld_map, md_map, cz_map, bz_map, sl_map, hw_map = mapper.create_maps(self.channels[i], self.basins[i])            

            # channel depth
            dh_map = event.dep_height(sl_map)
            cd_map = event.ch_depth(sl_map)

            channel_surface = erosional_surface(cld_map, cz_map, hw_map, cd_map)

            # NEED TO ADJUST THIS TO HANDLE MORE THAN THREE LAYERS (GRAVEL, SAND, SILT)
                        
            # gr_p: gravel proportions. gr_s: gravel sigma.
            # sa_p: sand proportions. sa_s: sand sigma.
            # si_p: silt proportions. si_s: silt sigma.
            # t_p: soma das proporções total (não precisa dar 100%) = gr_p + sa_p + si_p

            # We then deposit (aggradate) a certain number of materials. For instance, when using gravel, sand and silt, the number of materials == 3.
            # When using gravel, gross sand, medium sand, fine sand, and silt, then number of materials == 5.
            # When using gravel, very gross sand, gross sand, medium sand, fine sand, very fine sand, and silt, then number of materials == 7.
            
            # atualizar: retornar 8 variáveis em vez das 3 (algumas terão zeros, para garantir que sejam 7 camadas + separator)
            # gravel, sand and silt variables:        
            # gr_p: gravel proportions. gr_s: gravel sigma.
            # vcsa_p: very coarse sand proportions. vcsa_s: very coarse sand sigma.
            # csa_p: coarse sand proportions. csa_s: coarse sand sigma.
            # sa_p: sand proportions. sa_s: sand sigma.
            # fsa_p: fine sand proportions. fsa_s: fine sand sigma.
            # vfsa_s: very fine sand proportions. vfsa_s: very fine sand sigma.        
            # si_p: silt proportions. si_s: silt sigma.  
            # sep_p: separator proportions. sep_s: separator sigma.

            print('event mode:', event.mode)
            #print('event dep props:', event.dep_props(sl_map))
            #print('event dep props (len):', len(event.dep_props(sl_map)))
            print('----------')
            
            gr_p, vcsa_p, csa_p, sa_p, fsa_p, vfsa_p, si_p, sep_p = event.dep_props(sl_map)
            gr_s, vcsa_s, csa_s, sa_s, fsa_s, vfsa_s, si_s, sep_s = event.dep_sigmas(sl_map)
            t_p = gr_p + vcsa_s + csa_p + sa_p + fsa_p + vfsa_p + si_p + sep_p

            #print('TP: ', t_p)
            
            # t_p equals zero in case of a SEPARATOR event - NEED TO CHECK HERE
            if ((type(t_p) == np.ndarray and t_p.all() == 0) or (type(t_p) == int and t_p == 0)):
                t_p = 1

            #print('event.sep_thickness = ', event.sep_thickness)

            gravel_surface = (gr_p / t_p) * dh_map * gaussian_surface(gr_s, cld_map, hw_map)
            very_coarse_sand_surface = (vcsa_p / t_p) * dh_map * gaussian_surface(vcsa_s, cld_map, hw_map)
            coarse_sand_surface = (csa_p / t_p) * dh_map * gaussian_surface(csa_s, cld_map, hw_map)
            sand_surface = (sa_p / t_p) * dh_map * gaussian_surface(sa_s, cld_map, hw_map)
            fine_sand_surface = (fsa_p / t_p) * dh_map * gaussian_surface(fsa_s, cld_map, hw_map)
            very_fine_sand_surface = (vfsa_p / t_p) * dh_map * gaussian_surface(vfsa_s, cld_map, hw_map)
            silt_surface = (si_p / t_p) * dh_map * gaussian_surface(si_s, cld_map, hw_map)
            separator_surface = (sep_p / t_p) * dh_map * event.sep_thickness

            # reusing deposition variables for aggradation purposes
            gr_p, vcsa_p, csa_p, sa_p, fsa_p, vfsa_p, si_p, sep_p = event.aggr_props(sl_map)
            gr_s, vcsa_s, csa_s, sa_s, fsa_s, vfsa_s, si_s, sep_s = event.aggr_sigmas(sl_map)
            t_p = gr_p + vcsa_s + csa_p + sa_p + fsa_p + vfsa_p + si_p + sep_p
            
            # MANUEL: modulate the aggradation mapps in the case of gravel and sand by Gaussians with standard 
            #         deviations defined experimentally to avoid gravel and sand moving up walls of the channel.
            #         This actually works as a way of implementing a smooth cutoff for these material depositions.
            #         The function gaussian_surface defines a Gaussian inside the channel, thus returning zero
            #         only at the channels boarders. To force a quicker fall off (although only reaching zero at the channel's
            #         boarder) we used these experimentally defined standard deviations when accumulating the results of aggradation. 
            #    
            # DENNIS: corrected the value of t_p to avoid division by zero. t_p can be either an array or an integer           

            if isinstance(t_p, int) == True and t_p == 0: # check here (if isinstance is removed it does not work)
                t_p = 0.001

            STD_FOR_GRAVEL_FALL_OFF = 0.1   # EXPERIMENTALLY_DEFINED_STD_FOR_GRAVEL_FALL_OFF
            STD_FOR_SAND_FALL_OFF   = 0.6   # EXPERIMENTALLY_DEFINED_STD_FOR_SAND_FALL_OFF
            # atualizar              
            gravel_surface += (gr_p / t_p) * aggr_map * gaussian_surface(STD_FOR_GRAVEL_FALL_OFF, cld_map, hw_map)  # MANUEL
            very_coarse_sand_surface   += (vcsa_p / t_p) * aggr_map * gaussian_surface(STD_FOR_SAND_FALL_OFF, cld_map, hw_map)    # MANUEL
            coarse_sand_surface   += (csa_p / t_p) * aggr_map * gaussian_surface(STD_FOR_SAND_FALL_OFF, cld_map, hw_map)    # MANUEL
            sand_surface   += (sa_p / t_p) * aggr_map * gaussian_surface(STD_FOR_SAND_FALL_OFF, cld_map, hw_map)    # MANUEL
            fine_sand_surface   += (fsa_p / t_p) * aggr_map * gaussian_surface(STD_FOR_SAND_FALL_OFF, cld_map, hw_map)    # MANUEL
            very_fine_sand_surface   += (vfsa_p / t_p) * aggr_map * gaussian_surface(STD_FOR_SAND_FALL_OFF, cld_map, hw_map)    # MANUEL
            silt_surface   += (si_p / t_p) * aggr_map # CHECK WHETHER SILT NEEDS TO BE MODULATED BY A GAUSSIAN
            separator_surface   += (sep_p / t_p) * event.sep_thickness              
                
            # ADDED by MANUEL to smooth the aggradation maps due to their low resolutions
            gravel_surface = scipy.ndimage.gaussian_filter(gravel_surface, sigma = 10 / dx)
            very_coarse_sand_surface   = scipy.ndimage.gaussian_filter(very_coarse_sand_surface, sigma = 10 / dx)
            coarse_sand_surface   = scipy.ndimage.gaussian_filter(coarse_sand_surface, sigma = 10 / dx)
            sand_surface   = scipy.ndimage.gaussian_filter(sand_surface, sigma = 10 / dx)
            fine_sand_surface   = scipy.ndimage.gaussian_filter(fine_sand_surface, sigma = 10 / dx)
            very_fine_sand_surface   = scipy.ndimage.gaussian_filter(very_fine_sand_surface, sigma = 10 / dx)
            silt_surface   = scipy.ndimage.gaussian_filter(silt_surface, sigma = 10 / dx)
                              

            '''
            gravel_surface += (gr_p / t_p) * aggr_map
            sand_surface += (sa_p / t_p) * aggr_map
            silt_surface += (si_p / t_p) * aggr_map
            '''

            # CUTTING CHANNEL
            surface = scipy.ndimage.gaussian_filter(np.minimum(surface, channel_surface), sigma = 10 / dx)

            topo[:,:,i*L + 0] = surface

            #print('gravel_surface: ', gravel_surface)
            #print('very_coarse_sand_surface: ', very_coarse_sand_surface)

            # DEPOSITING SEDIMENT - superfície acumula gravel (1) + sand (5) + silt (1) + separator (1)
            # Colocar isso com 8 camadas com condicionais (se proporção for zero não soma)
            # TODO
            # atualizar
            surface += gravel_surface
            topo[:,:,i*L + 1] = surface
            surface += very_coarse_sand_surface
            topo[:,:,i*L + 2] = surface
            surface += coarse_sand_surface
            topo[:,:,i*L + 3] = surface
            surface += sand_surface
            topo[:,:,i*L + 4] = surface
            surface += fine_sand_surface
            topo[:,:,i*L + 5] = surface
            surface += very_fine_sand_surface
            topo[:,:,i*L + 6] = surface
            surface += silt_surface
            topo[:,:,i*L + 7] = surface
            surface += separator_surface
            topo[:,:,i*L + 8] = surface        
            
        return ChannelBelt3D(topo, xmin, ymin, dx, dx) # correto

class ChannelBelt3D():
    """
    Class for 3D models of channel belts. (Sylvester)
    """

    def __init__(self, topo, xmin, ymin, dx, dy):
        """
        Initializes the ChannelBelt3D object.

        :param topo: set of topographic surfaces (3D numpy array) (Sylvester)
        :param xmin: TODO
        :param ymin: TODO
        :param dx: gridcell size (m) (Sylvester)
        :param dy: TODO
        :return: TODO
        """

        #self.raw_plot_xsection(0.1, topo)
        #print('Entering topostrat from ChannelBelt3D INIT')
        #print('self.topo[:,:,-1]: (before topostrat) ', topo[:,:,-1])
        self.strat = topostrat(topo)      
        #self.raw_plot_xsection(0.1, self.strat)
        self.topo = topo
        #self.raw_plot_xsection(0.1, self.topo)

        self.xmin = xmin
        self.ymin = ymin        

        zmin, zmax = np.amin(self.strat[:,:,0]), np.amax(self.strat[:,:,-1])

        #print('self.strat[:,:,0]: ', self.strat[:,:,0])
        #print('self.topo[:,:,0]: (after topostrat)', self.topo[:,:,0])

        #print('zmin (1): ', zmin)
        #print('zmax (1): ', zmax)

        dz = zmax - zmin
        
        self.zmin = zmin - dz * 0.1
        self.zmax = zmax + dz * 0.1   

        #print('self.zmin INIT: ', self.zmin)
        #print('self.zmax INIT: ', self.zmax)
    
        self.dx = dx
        self.dy = dy

    def raw_plot_xsection(self, xsec, matrix):
        """
        TODO

        :param xsec: TODO
        :param matrix: TODO        
        """

        sy, sx, sz = np.shape(matrix)
        xindex = int(xsec * sx)
        Xv = np.linspace(0, sy, sy)
        for i in range(0, sz, 4):            
            Y1 = matrix[:,xindex,i]
            plt.plot(Xv, Y1)

        plt.show()

    def plot_xsection(self, xsec, ve = 1, substrat = True, title = '',
                    silt_color = [51/255, 51/255, 0], very_coarse_sand_color = [255/255, 153/255, 0], coarse_sand_color = [255/255, 204/255, 0],
                    sand_color = [255/255, 255/255, 0], fine_sand_color = [255/255, 255/255, 153/255], very_fine_sand_color = [255/255, 204/255, 153/255],
                    gravel_color = [255/255, 102/255, 0], separator_color = [0/255, 0/255, 255/255]):
        """
        Method for plotting a cross section through a 3D model; also plots map of 
        basal erosional surface and map of final geomorphic surface. [Sylvester]

        :param xsec: location of cross section along the x-axis (in pixel/ voxel coordinates)
        :param ve: vertical exaggeration
        :param substrat: (bool) TODO
        :param title: (string) TODO
        :param silt_color: (list)
        :param sand_color: (list)
        :param gravel_color: (list)
        """
        
        # DENNIS: modificado aqui
        #strat = self.topo
        strat = self.strat # aqui apenas strat final
        sy, sx, sz = np.shape(strat)
        if title != '': 
            title += '\n'
        
        xindex = int(xsec * sx)

        # gera as legendas para o Matplotlib
        # atualizar: testa numero de camadas e cria legenda com 3, 5 ou 7 (não criar legenda com "fantasmas")
        legend_elements = [
            Line2D([0], [0], color=silt_color, lw=NUMBER_OF_LAYERS_PER_EVENT, label='Silt'),
            Line2D([0], [0], color=very_fine_sand_color, lw=NUMBER_OF_LAYERS_PER_EVENT, label='Very Fine Sand'),
            Line2D([0], [0], color=fine_sand_color, lw=NUMBER_OF_LAYERS_PER_EVENT, label='Fine Sand'),            
            Line2D([0], [0], color=sand_color, lw=NUMBER_OF_LAYERS_PER_EVENT, label='Sand'),
            Line2D([0], [0], color=coarse_sand_color, lw=NUMBER_OF_LAYERS_PER_EVENT, label='Coarse Sand'),
            Line2D([0], [0], color=very_coarse_sand_color, lw=NUMBER_OF_LAYERS_PER_EVENT, label='Very Coarse Sand'),
            Line2D([0], [0], color=gravel_color, lw=NUMBER_OF_LAYERS_PER_EVENT, label='Gravel'),
            Line2D([0], [0], color=separator_color, lw=NUMBER_OF_LAYERS_PER_EVENT, label='Condensed Section'),
        ]

        # Matplotlib
        fig1 = plt.figure(figsize=(20,5))
        ax1 = fig1.add_subplot(111)
        ax1.set_title('{}Cross section at ({:.3f}) - {:.3f} km'.format(title, xsec, xindex * self.dx + self.xmin))

        Xv = np.linspace(self.ymin, self.ymin + sy * self.dy, sy)
        X1 = np.concatenate((Xv, Xv[::-1])) # faz array inverso
        
        if substrat:
            substract_color = [192/255, 192/255, 192/255]
            Yb = np.ones(sy) * self.zmin
            ax1.fill(X1, np.concatenate((Yb, strat[::-1,xindex,0])), facecolor=substract_color)
            legend_elements.append(
                Line2D([0], [0], color=substract_color, lw=4, label='Substract') 
            )
        
        # atualizar: Y1...Y7
        for i in range(0, sz, NUMBER_OF_LAYERS_PER_EVENT):            
            Y1 = np.concatenate((strat[:,xindex,i],   strat[::-1,xindex,i+1])) 
            Y2 = np.concatenate((strat[:,xindex,i+1], strat[::-1,xindex,i+2]))
            Y3 = np.concatenate((strat[:,xindex,i+2], strat[::-1,xindex,i+3]))
            Y4 = np.concatenate((strat[:,xindex,i+3], strat[::-1,xindex,i+4]))
            Y5 = np.concatenate((strat[:,xindex,i+4], strat[::-1,xindex,i+5]))
            Y6 = np.concatenate((strat[:,xindex,i+5], strat[::-1,xindex,i+6]))
            Y7 = np.concatenate((strat[:,xindex,i+6], strat[::-1,xindex,i+7]))
            Y8 = np.concatenate((strat[:,xindex,i+7], strat[::-1,xindex,i+8]))
            
            ax1.fill(X1, Y1, facecolor=gravel_color)
            ax1.fill(X1, Y2, facecolor=very_coarse_sand_color) 
            ax1.fill(X1, Y3, facecolor=coarse_sand_color)
            ax1.fill(X1, Y4, facecolor=sand_color)
            ax1.fill(X1, Y5, facecolor=fine_sand_color)
            ax1.fill(X1, Y6, facecolor=very_fine_sand_color)
            ax1.fill(X1, Y7, facecolor=silt_color)        
            ax1.fill(X1, Y8, facecolor=separator_color)        
        
        if ve != 1: 
            ax1.set_aspect(ve, adjustable='datalim')
        ax1.set_xlim(self.ymin, self.ymin + sy * self.dy)

        # DEBUG
        #print('self.zmin: ', self.zmin)
        #print('self.zmax: ', self.zmax)

        ax1.set_ylim(self.zmin, self.zmax)
        ax1.set_xlabel('Width (m)')
        ax1.set_ylabel('Elevation (m)')
        
        ax1.legend(handles=legend_elements, loc='upper right')
        return fig1

    def plot(self, ve = 1, curvature = False, save = False):
        """
        TODO

        :param ve: TODO 
        :param curvature: (bool) TODO        
        :param save: (bool) TODO 
        """

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

    def render(self, ve = 3):
        """
        TODO

        :param ve: TODO 
        """

        sy, sx, sz = np.shape(self.strat)
        x = np.linspace(self.xmin, self.xmin + sx * self.dx, sx)
        y = np.linspace(self.ymin, self.ymin + sy * self.dy, sy)
        
        xx, yy = np.meshgrid(x, y)

        zz = self.topo[:,:,0] * ve

        grid = pv.StructuredGrid(xx, yy, zz)

        plotter = pv.Plotter()
        plotter.add_mesh(grid)

        plotter.show()  

    # Generates a new PLY file containing the x,y,z coordinates in float instead of double
    def reducePlySize(self, inFileName, outFileName):
        """
        TODO

        :param inFileName: TODO 
        :param outFileName: TODO 
        """

        try:        
            # First part: read (and write) the ply header as text file
            with open(inFileName, "rt", encoding="Latin-1") as inFile:
                with open(outFileName, "wt") as outFile:
                    line = ''
                    while line != 'end_header\n':
                        line = inFile.readline()
                        if line == 'property double x\n':
                            outFile.write('property float x\n')
                        elif line == 'property double y\n':
                            outFile.write('property float y\n')
                        elif line == 'property double z\n':
                            outFile.write('property float z\n')
                            
                        else:                        
                            if line[0:15] == 'element vertex ':
                                nVertices = int(line[15:-1]) #gets que number of vertices
                            #if line[0:13] == 'element face ':
                            #    nFaces = int(line[13:-1]) #gets que number of faces                            

                            outFile.write(line)
                    
                    currentPos = inFile.tell()
            
            # Second part: read (and write) the ply vertices colors and faces as text file
            with open(inFileName, "rb") as inFile:                        
                with open(outFileName, "ab") as outFile:
                    inFile.seek(currentPos)                

                    # Read and trasnform all the vertex values to float, maintaining their colors
                    # Part 1: convert x, y, z from double to float... RGB keep the same as ubyte
                    for i in range(nVertices):
                        x = np.float32(struct.unpack('d',inFile.read(8))) # Read the x double value
                        y = np.float32(struct.unpack('d',inFile.read(8)))
                        z = np.float32(struct.unpack('d',inFile.read(8)))                    
                        x = struct.pack('f',x[0])                    
                        y = struct.pack('f',y[0])
                        z = struct.pack('f',z[0])

                        red = np.ubyte(struct.unpack('B',inFile.read(1)))                    
                        green = np.ubyte(struct.unpack('B',inFile.read(1)))
                        blue = np.ubyte(struct.unpack('B',inFile.read(1)))                      
                        red = struct.pack('B',red[0])                                        
                        green = struct.pack('B',green[0])
                        blue = struct.pack('B',blue[0])

                        outFile.write(x)
                        outFile.write(y)
                        outFile.write(z)
                        outFile.write(red)
                        outFile.write(green)
                        outFile.write(blue)    

                    # Read and trasnform all the vertex values to float, maintaining their colors
                    buffer = inFile.read(1)
                    while buffer != b"":                                     
                        outFile.write(buffer)
                        buffer = inFile.read(1)                

        except IOError:
            print("Error. Could not read files ", inFileName, " and ", outFileName)    

    def generateTriangleMesh(self, vertices, faces, colors, fileNameOut='out.ply', coloredMesh=True):
        """
        Generate and export a 3D model in PLY file format. The file has reduced size with float32 data instead of double.

        :param vertices: TODO 
        :param faces: TODO 
        :param faces: TODO 
        :param colors: TODO 
        :param fileNameOut: (string) TODO 
        :param coloredMesh: (bool) TODO 
        """

        mesh = o3d.geometry.TriangleMesh()
        
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        if coloredMesh:
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        
        o3d.io.write_triangle_mesh(fileNameOut, mesh, write_vertex_colors=coloredMesh, compressed=True)
        
        # New code for saving/overwriting a new PLY file with float32 instead of double (64)        
        self.reducePlySize(fileNameOut, fileNameOut[:-4]+'_'+'.ply')
        if os.path.isfile(fileNameOut):
            os.remove(fileNameOut)
        if os.path.isfile(fileNameOut[:-4]+'_'+'.ply'):
            os.rename(fileNameOut[:-4]+'_'+'.ply', fileNameOut)

    def export_top_layer(self, structure, structure_colors, event_top_layer, number_layers_per_event, grid, top, filename, plant_view, \
                        reduction = None, colored_mesh = True):
        """
        TODO

        :param structure: TODO 
        :param structure_colors: TODO 
        :param event_top_layer: TODO 
        :param number_layers_per_event: TODO 
        :param grid: TODO 
        :param top: TODO
        :param filename: (string) TODO
        :param plant_view: TODO
        :param reduction: TODO
        :param colored_mesh: (bool) TODO
        """

        FLOAT_TO_INT_FACTOR = 1

        sy, sx, sz = np.shape(structure) # sz contains the number of layers from strat
        # Array containing the possible colors from strat        
        colorCpInt = np.zeros([sy,sx])    
            
        # Stores information of the points and their colors of the first layer (top)
        colorCpInt = np.zeros((sy,sx, 3))
        for xIndex in range(0, sx):
            for yIndex in range(0, sy):
                #for zIndex in range(sz-1, 1, -1): # 0 is the bottom layer
                for zIndex in range(event_top_layer, 0, -1): # 0 is the bottom layer
                    if structure[yIndex,xIndex,zIndex] == 1:
                        # Computes in colorIndex the correct layer color according to the following formula
                        colorIndex = abs((zIndex % number_layers_per_event) - (number_layers_per_event-1))
                        colorCpInt[yIndex,xIndex] = structure_colors[colorIndex]
                        break

        # Stores the data structure from points, colors and plant view
        surfacePointList = []
        colorPointList = []
        cols, rows, channel = np.shape(colorCpInt)
        topCp = top.copy()
        topCp = np.reshape(topCp, (rows,cols,channel))                                    
        for col in range(cols):
            for row in range(rows):    
                surfacePointList.append([topCp[row][col][0], topCp[row][col][1], topCp[row][col][2]])
                colorPointList.append([colorCpInt[col][row][0], colorCpInt[col][row][1], colorCpInt[col][row][2]])
                plant_view[col,row] = np.uint8(colorCpInt[col,row] * 255)
        
        im = Image.fromarray(plant_view)
        im.save(filename + ".png")
        
        # Produce an PLY file with it's box already triangulated
        bottom = grid.points.copy()                
        bottom[:,-1] = self.zmin
        grid.points = np.vstack((top, bottom))
        grid.dimensions = [*grid.dimensions[0:2], 2]
        plotter = pv.Plotter()
        plotter.add_mesh(grid)#, scalars="colors", rgb=True) # add to scene                
        plotter.export_obj(filename) # export two independent meshes: top and bottom            
        data = trimesh.load(filename + '.obj', force='mesh') # load two triangle meshes: top and bottom                
        vertices = data.vertices
        faces = np.ones((data.faces.shape[0], data.faces.shape[1]+1)) * data.faces.shape[1]                
        faces[:,1:] = data.faces
        faces = np.hstack(faces).astype(int)
        mesh = pv.PolyData(vertices, faces)            
        if reduction is not None:
            mesh.decimate(reduction)
        mesh.save(filename + '.ply')

        # Map the surface points (and their respective surface color points) to the block points, which have doubled the amount of points
        # with a projection on XY axis. The idea is to associate each color from the surface with the new ordered block points. Due to the
        # performance, instead of comparing both lists we converted the surface points (each point as [x,y,z]) to a dictionary composed by
        # their math.ceil values. This may lead to some precision errors.
        surface_points = np.asarray(surfacePointList)
        surface_colors = np.asarray(colorPointList)

        mesh = o3d.io.read_triangle_mesh(filename + ".ply")
        block_vertices = np.asarray(mesh.vertices)
        block_triangles = np.asarray(mesh.triangles)

        # Code to export the obj and ply surface mesh as a block with ground and walls with color
        if colored_mesh:
            # Create hash in surfaceDict to compare more efficiently "two lists" of elements
            block_colors = np.zeros([len(block_vertices),3])
            surfaceDict = {}
            for event_top_layer in range(len(surface_points)):
                x_str = str(math.ceil((surface_points[event_top_layer][0]*FLOAT_TO_INT_FACTOR)//1))
                y_str = str(math.ceil((surface_points[event_top_layer][1]*FLOAT_TO_INT_FACTOR)//1))
                z_str = str(math.ceil((surface_points[event_top_layer][2]*FLOAT_TO_INT_FACTOR)//1))
                hash_aux = x_str + ',' + y_str + ',' + z_str
                surfaceDict[hash_aux] = surface_colors[event_top_layer]

            cont = 0
            for v in block_vertices:                    
                x_str = str(math.ceil((v[0]*FLOAT_TO_INT_FACTOR)//1))
                y_str = str(math.ceil((v[1]*FLOAT_TO_INT_FACTOR)//1))
                z_str = str(math.ceil((v[2]*FLOAT_TO_INT_FACTOR)//1))
                hash_aux = x_str + ',' + y_str + ',' + z_str
                if hash_aux in surfaceDict:
                    block_colors[cont] = surfaceDict[hash_aux]
                cont = cont + 1                
            
            self.generateTriangleMesh(block_vertices, block_triangles, block_colors, filename + '.ply', coloredMesh=True)

        # Code to export the obj and ply surface mesh as a block with ground and walls (Beuren's original mesh)
        else:              
            block_colors = np.zeros([len(block_vertices),3])
            self.generateTriangleMesh(block_vertices, block_triangles, block_colors, filename + '.ply', coloredMesh=False)
  
    def export_objs(self, top_event_layers_zipname = 'event_layers.zip', reduction = None, ve = 3):
        """
        Function to export the 3D meshes for each layer. Meshes are compressed as PLY files and compacted into a ZIP.
        Outputs a ZIP file containing the models for each layer (names as model1.ply, model2.ply, etc).

        :param top_event_layers_zipname: TODO 
        :param reduction: TODO 
        :param ve: vertical exaggeration TODO        
        """

        # Constants        
        LAYER_THICKNESS_THRESHOLD = 1e-1#0.9 #1e-2   

        # atualizar
        
        GRAVEL_COLOR = [255/255, 102/255, 0]
        VERY_COARSE_SAND_COLOR = [255/255, 153/255, 0]
        COARSE_SAND_COLOR = [255/255, 204/255, 0]
        SAND_COLOR = [255/255, 255/255, 0]
        FINE_SAND_COLOR = [255/255, 255/255, 153/255]
        VERY_FINE_SAND_COLOR = [255/255, 204/255, 153/255]
        SILT_COLOR = [51/255, 51/255, 0]
        SEPARATOR_COLOR = [0/255, 0/255, 255/255]
        SUBSTRACT_COLOR = [192/255, 192/255, 192/255]        
        
        # Set the strat material colors to an array
        strat_colors = np.array([SEPARATOR_COLOR, SILT_COLOR, VERY_FINE_SAND_COLOR, FINE_SAND_COLOR, SAND_COLOR, COARSE_SAND_COLOR,
                                VERY_COARSE_SAND_COLOR, GRAVEL_COLOR, SUBSTRACT_COLOR])
        
        # dir contains the path of the temp directory, used to store the intermediate meshes
        dir = tempfile.mkdtemp()
        
        # strat: 3D numpy array of statigraphic surfaces (previously was named zz)
        #strat = topostrat(self.topo)
        strat = self.strat

        sy, sx, sz = np.shape(strat) # sz contains the number of layers from strat

        # Produce a list of 'sx' and 'sy' interpolated values from a given range
        x = np.linspace(self.xmin, self.xmin + sx * self.dx, sx)
        y = np.linspace(self.ymin, self.ymin + sy * self.dy, sy)

        # xx and yy form a grid (2D plane) according to x and y values
        xx, yy = np.meshgrid(x, y)
        
        # For each adjacent strat layer compares their thickness, setting to 0 if they are very close and 1 otherwise
        stratCp = strat.copy()
        for xIndex in range(0, sx):
            for yIndex in range(0, sy):
                for zIndex in range(sz-1, 1, -1):
                    if (abs(stratCp[yIndex,xIndex,zIndex-1] - stratCp[yIndex,xIndex,zIndex-2]) < LAYER_THICKNESS_THRESHOLD):
                        stratCp[yIndex,xIndex,zIndex-1] = 0
                    else:
                        stratCp[yIndex,xIndex,zIndex-1] = 1

                    #break
                stratCp[yIndex,xIndex,sz-1] = 0
        
        # Initializes the plant view of the channel for each of the layers.
        plant_view = np.uint8(np.zeros((sy,sx,3)))
        mesh_iterator = 0
        # Main loop to generate a mesh for each layer. The meshes are available in a zip file names i.ply.
        # For now, we have 4 layers in the following order: silt, sand, gravel and substract
        # Layer 0 corresponds to the initialized layer in the constructor of the channel belt        
        for event_top_layer in range(0, sz, NUMBER_OF_LAYERS_PER_EVENT):
            update_progress(event_top_layer/sz)            
            #filename = 'model{}'.format(int(i/3) + 1) # local folder
            filename = path.join(dir, '{}'.format((int)(event_top_layer/NUMBER_OF_LAYERS_PER_EVENT) + 1)) # temp folder, all models

            print('Entering topostrat from export_objs')
            strat = topostrat(self.topo, event_top_layer)
            
            # Produces a grid for the current z layer containing the points in grid.points
            grid = pv.StructuredGrid(xx, yy, strat[:,:,event_top_layer] * ve)

            # top contains all the surface points for each layer
            top = grid.points.copy()

            # Export one mesh
            self.export_top_layer(stratCp, strat_colors, event_top_layer, NUMBER_OF_LAYERS_PER_EVENT, grid, top, filename, plant_view, \
                        reduction, colored_mesh=True)            

            mesh_iterator = mesh_iterator + 1

        # Compact in a zip file all the ply files in filename folder
        zipfile = path.join(dir, top_event_layers_zipname)
        zipFilesInDir(dir, zipfile, lambda fn: path.splitext(fn)[1] == '.ply')
        copyfile(zipfile, top_event_layers_zipname)

        # Compact in a zip file all the ply files in filename folder
        zipfile = path.join(dir, 'plant_views.zip')
        zipFilesInDir(dir, zipfile, lambda fn: path.splitext(fn)[1] == '.png')
        copyfile(zipfile, 'plant_views.zip')

        # EXPORT ALL THE LAYERS OF THE FINAL MESH
        dir = tempfile.mkdtemp()
        mesh_iterator = 0
        for event_top_layer in range(sz-NUMBER_OF_LAYERS_PER_EVENT, sz):
            update_progress(event_top_layer/sz)            
            #filename = 'model{}'.format(int(i/3) + 1) # local folder
            filename = path.join(dir, '{}'.format((int)(mesh_iterator) + 1)) # temp folder, all models
            
            # Produces a grid for the current z layer containing the points in grid.points
            grid = pv.StructuredGrid(xx, yy, strat[:,:,event_top_layer] * ve)

            # top contains all the surface points for each layer
            top = grid.points.copy()

            # Export one mesh
            self.export_top_layer(stratCp, strat_colors, event_top_layer, NUMBER_OF_LAYERS_PER_EVENT, grid, top, filename, plant_view, \
                        reduction, colored_mesh = True)

            mesh_iterator = mesh_iterator + 1
        
        # Compact in a zip file all the ply files in filename folder
        zipfile = path.join(dir, 'final_layers.zip')
        zipFilesInDir(dir, zipfile, lambda fn: path.splitext(fn)[1] == '.ply')
        copyfile(zipfile, 'final_layers.zip')        

    def export(self, ve = 3):
        """
        TODO

        :param ve: TODO   
        """
        #np.savetxt('shape.txt',[sy, sx, sz],fmt='%.4f') # DEBUG
        #zz = topostrat_evolution(self.topo)
        print("entering topostrat from export")
        zz = topostrat(self.topo)
        np.save("terrain.npy", zz)