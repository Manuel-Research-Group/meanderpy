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
        G = np.exp(-2.0 * K * Cf / d[i] * SIGMA_2) # convolution vector
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

    def curve_offset(self):
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

    def resample(self, ds):
        _, _, _, _, s = self.derivatives()
        tck, _ = scipy.interpolate.splprep([self.x,self.y,self.z,self.w,self.d],s=0) 
        u = np.linspace(0,1,1+int(round(s[-1]/ds)))
        self.x, self.y, self.z, self.w, self.d = scipy.interpolate.splev(u,tck) 

    def migrate(self,Cf,kl,dt):
        curv = self.curvature()
        dx, dy, _, ds, s = self.derivatives()
        sinuosity = s[-1]/(self.x[-1]-self.x[0])
        R0 = kl * self.w * curv
        R1 = compute_migration_rate(R0, Cf, self.d, ds, s[-1])
        RN = sinuosity**(-2/3.0) * R1 
        self.x += RN * (dy/ds) * dt  
        self.y -= RN * (dx/ds) * dt 

    def cut_cutoffs(self, crdist, ds):
        cuts = []
        
        diag_blank_width = int((crdist+20*ds)/ds)
        # UPPER MARGIN
        xo, yo = self.curve_offset()
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

            xo, yo = self.curve_offset()
            ind1, ind2 = find_cutoffs(self.x+xo, self.y+yo, crdist, diag_blank_width)

        # LOWER MARGIN
        xo, yo = self.curve_offset()
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

            xo, yo = self.curve_offset()
            ind1, ind2 = find_cutoffs(self.x-xo, self.y-yo, crdist, diag_blank_width)

        return cuts

    def plot(self, axis = plt, color=sns.xkcd_rgb["ocean blue"]):
        x = self.x
        y = self.y

        xo, yo = self.curve_offset()
        
        xm = np.hstack((x + xo, (x - xo)[::-1]))
        ym = np.hstack((y + yo, (y - yo)[::-1]))
        axis.fill(xm, ym, color=color, edgecolor='k', linewidth=0.25)

class Cutoff:
    """class for Cutoff objects"""
    def __init__(self,x,y,z,W,D):
        """initialize Cutoff object
        x, y, z  - coordinates of centerline
        W - channel width
        D - channel depth"""
        self.x = x
        self.y = y
        self.z = z
        self.W = W
        self.D = D
class ChannelBelt:
    """class for ChannelBelt objects"""
    def __init__(self, channels, cutoffs, cl_times, cutoff_times):
        """initialize ChannelBelt object
        channels - list of Channel objects
        cutoffs - list of Cutoff objects
        cl_times - list of ages of Channel objects
        cutoff_times - list of ages of Cutoff objects"""
        self.channels = channels
        self.cutoffs = cutoffs
        self.cl_times = cl_times
        self.cutoff_times = cutoff_times

    def migrate(self,nit,saved_ts,ds,pad,crdist,Cf,kl,kv,dt,dens,t1,t2,t3,aggr_factor,*D):
        channel = self.channels[-1].copy()
        last_cl_time = 0

        for itn in range(nit):
            update_progress(itn/nit)
            channel.migrate(Cf,kl,dt)
            channel.cut_cutoffs(25,ds)
            channel.resample(ds)

            if itn % saved_ts == 0:
                self.cl_times.append(last_cl_time+(itn+1)*dt/(365*24*60*60.0))
                self.channels.append(channel.copy())

    def plot(self, end_time = 0):
        cot = np.array(self.cutoff_times)
        sclt = np.array(self.cl_times)

        if end_time > 0:
            cot = cot[cot<=end_time]
            sclt = sclt[sclt<=end_time]
        times = np.sort(np.hstack((cot,sclt)))
        times = np.unique(times).tolist()

        fig, axis = plt.subplots(1, 1)

        for i, t in enumerate(times):
            color = sns.xkcd_rgb["ocean blue"] if i == len(times) - 1 else sns.xkcd_rgb["sand yellow"]
            if t in sclt:
                index = np.where(sclt==t)[0][0]
                self.channels[index].plot(axis, color)

            if t in cot:
                index = np.where(cot==times[i])[0][0]
                for cutoff in self.cutoffs[index]:
                    cutoff.plot(axis, color)
        return fig
