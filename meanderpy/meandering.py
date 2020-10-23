import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.interpolate
from scipy.spatial import distance
from scipy import ndimage
from scipy.signal import savgol_filter
from scipy.interpolate import splev, splrep
from PIL import Image, ImageDraw
from skimage import measure
from skimage import morphology
from matplotlib.colors import LinearSegmentedColormap
import time, sys
import numba
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib import cm

# FITTING PARAMETERS
W2, W1, W0 = 695.4154350661511, -45.231656699536124, 104.60941780103624
D1, D0 = -1038.6756354535573, -4.136666533884889

OMEGA = -1.0 # constant in curvature calculation (Howard and Knutson, 1984)
GAMMA = 2.5  # from Ikeda et al., 1981 and Howard and Knutson, 1984
K = 1.0 # constant in HK equation

#USING CONSTANT VALUES FOR NOW
D = 20
W = 400

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

def width_ds(ds):
    return W2 * np.exp(- W1 * ds) + W0

def depth_ds(ds):
    return D1 * ds + D0

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

    def curvature(self):
        x = self.x
        y = self.y

        dx = np.gradient(x)
        dy = np.gradient(y)      
        
        ddx = np.gradient(dx)
        ddy = np.gradient(dy) 
        
        return np.abs(dx * ddy - dy*ddx)/((dx**2 + dy**2)**1.5)

    def curve_offset(self):
        x = self.x
        y = self.y
        d = self.w / 2

        dx = np.gradient(x)
        dy = np.gradient(y)

        n = np.stack((dy, -dx))
        l = np.sqrt(np.sum(np.conj(n) * n, axis = 0))
            
        xo = d * (  dy / l )
        yo = d * ( -dx / l )

        return xo, yo

    def plot(self, axis = plt, color=sns.xkcd_rgb["ocean blue"]):
        x = self.x
        y = self.y

        xo, yo = self.curve_offset()
        
        xm = np.hstack((x + xo, (x - xo)[::-1]))
        ym = np.hstack((y + yo, (y - yo)[::-1]))
        axis.fill(xm, ym, color=color, edgecolor='k', linewidth=0.25)
        
        #axis.plot(x + xo, y + yo, color=color)
        #axis.plot(x - xo, y - yo, color=color)

        #axis.plot(x, y)

        
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

    def migrate(self,nit,saved_ts,deltas, pad, crdist,Cf,kl,kv,dt,dens,t1,t2,t3,aggr_factor):
        channel = self.channels[-1] # first channel is the same as last channel of input
        x = channel.x; y = channel.y; z = channel.z
        w = channel.w; d = d = channel.d;
        plt.plot(x, y)
        plt.show()
        k = 1.0 # constant in HK equation
        xc = [] # initialize cutoff coordinates
        # determine age of last channel:
        if len(self.cl_times)>0:
            last_cl_time = self.cl_times[-1]
        else:
            last_cl_time = 0
        dx, dy, dz, ds, s = compute_derivatives(x,y,z)
        slope = np.gradient(z)/ds
        # padding at the beginning can be shorter than padding at the downstream end:
        pad1 = int(pad/10.0)
        omega = -1.0 # constant in curvature calculation (Howard and Knutson, 1984)
        gamma = 2.5 # from Ikeda et al., 1981 and Howard and Knutson, 1984
    
        
        for itn in range(nit): # main loop
            update_progress(itn/nit)
            x, y = migrate_one_step(x,y,z,w,d,kl,dt,Cf)
            
            f.write('\n')
            np.savetxt(f, x, fmt='%4.3f')
            f.write('\n')
            np.savetxt(f, y, fmt='%4.3f')
            f.write('\n\n')
            
            # x, y = migrate_one_step_w_bias(x,y,z,W,kl,dt,k,Cf,D,pad,pad1,omega,gamma)
            x,y,z,xc,yc,zc = cut_off_cutoffs(x,y,z,s,crdist,deltas) # find and execute cutoffs
            x,y,z,dx,dy,dz,ds,s = resample_centerline(x,y,z,deltas) # resample centerline
            slope = np.gradient(z)/ds
            # for itn<t1, z is unchanged
            if (itn>t1) & (itn<=t2): # incision
                if np.min(np.abs(slope))!=0: # if slope is not zero
                    z = z + kv*dens*9.81*D*slope*dt
                else:
                    z = z - kv*dens*9.81*D*dt*0.05 # if slope is zero
            if (itn>t2) & (itn<=t3): # lateral migration
                if np.min(np.abs(slope))!=0: # if slope is not zero
                    z = z + kv*dens*9.81*D*slope*dt - kv*dens*9.81*D*np.median(slope)*dt
                else:
                    z = z # no change in z
            if (itn>t3): # aggradation
                if np.min(np.abs(slope))!=0: # if slope is not zero
                    z = z + kv*dens*9.81*D*slope*dt - aggr_factor*kv*dens*9.81*D*np.mean(slope)*dt 
                else:
                    z = z + aggr_factor*dt
            if len(xc)>0: # save cutoff data
                self.cutoff_times.append(last_cl_time+(itn+1)*dt/(365*24*60*60.0))
                cutoff = Cutoff(xc,yc,zc,0,0) # create cutoff object
                self.cutoffs.append(cutoff)
            # saving centerlines:
            if np.mod(itn,saved_ts)==0:

                self.cl_times.append(last_cl_time+(itn+1)*dt/(365*24*60*60.0))
                channel = Channel(x,y,z,w[0] * np.ones(len(x)), d[0] * np.ones(len(x))) # create channel object
                self.channels.append(channel)

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

      
def resample_centerline(x,y,z,deltas):
    dx, dy, dz, ds, s = compute_derivatives(x,y,z) # compute derivatives
    # resample centerline so that 'deltas' is roughly constant
    # [parametric spline representation of curve; note that there is *no* smoothing]
    tck, u = scipy.interpolate.splprep([x,y,z],s=0) 
    unew = np.linspace(0,1,1+int(round(s[-1]/deltas))) # vector for resampling
    out = scipy.interpolate.splev(unew,tck) # resampling
    x, y, z = out[0], out[1], out[2] # assign new coordinate values
    dx, dy, dz, ds, s = compute_derivatives(x,y,z) # recompute derivatives
    return x,y,z,dx,dy,dz,ds,s

def migrate_one_step(x,y,z,w,d,kl,dt,Cf):
    ns = len(x)
    curv = compute_curvature(x,y)
    dx, dy, dz, ds, s = compute_derivatives(x,y,z)
    sinuosity = s[-1] / (x[-1] - x[0])
    R0 = kl * W * curv # simple linear relationship between curvature and nominal migration rate
    alpha = K * 2 * Cf / D # exponent for convolution function G
    R1 = compute_migration_rate(ns,ds,alpha,R0) * sinuosity**(-2/3.0) 
    # calculate new centerline coordinates:
    dy_ds = dy/ds
    dx_ds = dx/ds
    # adjust x and y coordinates (this *is* the migration):
    x = x + R1 * dy_ds * dt
    y = y - R1 * dx_ds * dt
    return x,y

@numba.jit(nopython=True) # use Numba to speed up the heaviest computation
def compute_migration_rate(ns,ds,alpha,R0):
    """compute migration rate as weighted sum of upstream curvatures
    pad - padding (number of nodepoints along centerline)
    ns - number of points in centerline
    ds - distances between points in centerline
    omega - constant in HK model
    gamma - constant in HK model
    R0 - nominal migration rate (dimensionless curvature * migration rate constant)"""
    R1 = np.zeros(ns)                                             # preallocate adjusted channel migration rate
    for i in range(0,ns):
        si2 = np.hstack((np.array([0]),np.cumsum(ds[i-1::-1])))   # distance along centerline, backwards from current point 
        G = np.exp(-alpha*si2)                                    # convolution vector
        R1[i] = OMEGA*R0[i] + GAMMA*np.sum(R0[i::-1]*G)/np.sum(G) # main equation
    return R1

def compute_derivatives(x,y,z):
    """function for computing first derivatives of a curve (centerline)
    x,y are cartesian coodinates of the curve
    outputs:
    dx - first derivative of x coordinate
    dy - first derivative of y coordinate
    ds - distances between consecutive points along the curve
    s - cumulative distance along the curve"""

    dx = np.gradient(x)
    dy = np.gradient(y)   
    dz = np.gradient(z)   

    ds = np.vstack((np.power(dx, 2), np.power(dy, 2), np.power(dz, 2)))
    ds = np.sqrt(np.sum(ds, axis = 0))

    s = np.hstack((0,np.cumsum(ds[1:])))

    return dx, dy, dz, ds, s

def compute_curvature(x,y):
    """function for computing first derivatives and curvature of a curve (centerline)
    x,y are cartesian coodinates of the curve
    outputs:
    dx - first derivative of x coordinate
    dy - first derivative of y coordinate
    ds - distances between consecutive points along the curve
    s - cumulative distance along the curve
    curvature - curvature of the curve (in 1/units of x and y)"""
    
    dx = np.gradient(x)
    dy = np.gradient(y)      
    
    ddx = np.gradient(dx)
    ddy = np.gradient(dy) 
    
    curvature = np.abs(dx*ddy-dy*ddx)/((dx**2+dy**2)**1.5)

    return curvature

def kth_diag_indices(a,k):
    """function for finding diagonal indices with k offset
    [from https://stackoverflow.com/questions/10925671/numpy-k-th-diagonal-indices]"""
    rows, cols = np.diag_indices_from(a)
    if k<0:
        return rows[:k], cols[-k:]
    elif k>0:
        return rows[k:], cols[:-k]
    else:
        return rows, cols

def find_cutoffs(x, y, crdist, deltas):
    """function for identifying locations of cutoffs along a centerline
    and the indices of the segments that will become part of the oxbows
    x,y - coordinates of centerline
    crdist - critical cutoff distance
    deltas - distance between neighboring points along the centerline"""
    
    diag_blank_width = int((crdist+20*deltas)/deltas)
    # distance matrix for centerline points:
    XY = np.array([x,y]).T

    dist = distance.cdist(XY, XY)

    dist[dist>crdist] = np.NaN # set all values that are larger than the cutoff threshold to NaN
    # set matrix to NaN along the diagonal zone:
    
    for k in range(-diag_blank_width, diag_blank_width + 1):
        rows, cols = kth_diag_indices(dist,k)
        dist[rows,cols] = np.NaN
    
    i1, i2 = np.where(~np.isnan(dist))
    
    ind1 = i1[np.where(i1<i2)[0]] # get rid of unnecessary indices
    ind2 = i2[np.where(i1<i2)[0]] # get rid of unnecessary indices
    
    return ind1, ind2 # return indices of cutoff points and cutoff coordinates

def cut_off_cutoffs(x,y,z,s,crdist,deltas):
    """function for executing cutoffs - removing oxbows from centerline and storing cutoff coordinates
    x,y - coordinates of centerline
    crdist - critical cutoff distance
    deltas - distance between neighboring points along the centerline
    outputs:
    x,y,z - updated coordinates of centerline
    xc, yc, zc - lists with coordinates of cutoff segments"""
    xc = []
    yc = []
    zc = []
    ind1, ind2 = find_cutoffs(x,y,crdist,deltas) # initial check for cutoffs
    while len(ind1)>0:
        xc.append(x[ind1[0]:ind2[0]+1]) # x coordinates of cutoff
        yc.append(y[ind1[0]:ind2[0]+1]) # y coordinates of cutoff
        zc.append(z[ind1[0]:ind2[0]+1]) # z coordinates of cutoff
        x = np.hstack((x[:ind1[0]+1],x[ind2[0]:])) # x coordinates after cutoff
        y = np.hstack((y[:ind1[0]+1],y[ind2[0]:])) # y coordinates after cutoff
        z = np.hstack((z[:ind1[0]+1],z[ind2[0]:])) # z coordinates after cutoff
        ind1, ind2 = find_cutoffs(x,y,crdist,deltas)       
    return x,y,z,xc,yc,zc



