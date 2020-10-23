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

class ChannelBelt3D:
    """class for 3D models of channel belts"""
    def __init__(self, model_type, topo, strat, facies, facies_code, dx, channels):
        """model_type - can be either 'fluvial' or 'submarine'
        topo - set of topographic surfaces (3D numpy array)
        strat - set of stratigraphic surfaces (3D numpy array)
        facies - facies volume (3D numpy array)
        facies_code - dictionary of facies codes, e.g. {0:'oxbow', 1:'point bar', 2:'levee'}
        dx - gridcell size (m)
        channels - list of channel objects that form 3D model"""
        self.model_type = model_type
        self.topo = topo
        self.strat = strat
        self.facies = facies
        self.facies_code = facies_code
        self.dx = dx
        self.channels = channels

    def plot_xsection(self, xsec, colors, ve):
        """method for plotting a cross section through a 3D model; also plots map of 
        basal erosional surface and map of final geomorphic surface
        xsec - location of cross section along the x-axis (in pixel/ voxel coordinates) 
        colors - list of RGB values that define the colors for different facies
        ve - vertical exaggeration"""
        strat = self.strat
        dx = self.dx
        fig1 = plt.figure(figsize=(20,5))
        ax1 = fig1.add_subplot(111)
        r,c,ts = np.shape(strat)
        Xv = dx * np.arange(0,r)
        for xloc in range(xsec,xsec+1,1):
            for i in range(0,ts-1,3):
                X1 = np.concatenate((Xv, Xv[::-1]))  
                Y1 = np.concatenate((strat[:,xloc,i], strat[::-1,xloc,i+1])) 
                Y2 = np.concatenate((strat[:,xloc,i+1], strat[::-1,xloc,i+2]))
                Y3 = np.concatenate((strat[:,xloc,i+2], strat[::-1,xloc,i+3]))
                if self.model_type == 'submarine':
                    ax1.fill(X1, Y1, facecolor=colors[2], linewidth=0.5, edgecolor=[0,0,0]) # oxbow mud
                    ax1.fill(X1, Y2, facecolor=colors[0], linewidth=0.5, edgecolor=[0,0,0]) # point bar sand
                    ax1.fill(X1, Y3, facecolor=colors[1], linewidth=0.5) # levee mud
                if self.model_type == 'fluvial':
                    ax1.fill(X1, Y1, facecolor=colors[0], linewidth=0.5, edgecolor=[0,0,0]) # levee mud
                    ax1.fill(X1, Y2, facecolor=colors[1], linewidth=0.5, edgecolor=[0,0,0]) # oxbow mud
                    ax1.fill(X1, Y3, facecolor=colors[2], linewidth=0.5) # channel sand
            ax1.set_xlim(0,dx*(r-1))
            ax1.set_aspect(ve, adjustable='datalim')
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.contourf(strat[:,:,ts-1],100,cmap='viridis')
        ax2.contour(strat[:,:,ts-1],100,colors='k',linestyles='solid',linewidths=0.1,alpha=0.4)
        ax2.plot([xloc, xloc],[0,r],'k',linewidth=2)
        ax2.axis([0,c,0,r])
        ax2.set_aspect('equal', adjustable='box')        
        ax2.set_title('final geomorphic surface')
        ax2.tick_params(bottom=False,top=False,left=False,right=False,labelbottom=False,labelleft=False)
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        ax3.contourf(strat[:,:,0],100,cmap='viridis')
        ax3.contour(strat[:,:,0],100,colors='k',linestyles='solid',linewidths=0.1,alpha=0.4)
        ax3.plot([xloc, xloc],[0,r],'k',linewidth=2)
        ax3.axis([0,c,0,r])
        ax3.set_aspect('equal', adjustable='box')
        ax3.set_title('basal erosional surface')
        ax3.tick_params(bottom=False,top=False,left=False,right=False,labelbottom=False,labelleft=False)
        return fig1,fig2,fig3

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

    def migrate(self,nit,saved_ts,deltas,pad,crdist,Cf,kl,kv,dt,dens,t1,t2,t3,aggr_factor,*D):
        """function for computing migration rates along channel centerlines and moving the centerlines accordingly
        inputs:
        nit - number of iterations
        saved_ts - which time steps will be saved
        deltas - distance between nodes on centerline
        pad - padding (number of nodepoints along centerline)
        crdist - threshold distance at which cutoffs occur
        Cf - dimensionless Chezy friction factor
        kl - migration rate constant (m/s)
        kv - vertical slope-dependent erosion rate constant (m/s)
        dt - time step (s)
        dens - density of fluid (kg/m3)
        t1 - time step when incision starts
        t2 - time step when lateral migration starts
        t3 - time step when aggradation starts
        aggr_factor - aggradation factor
        D - channel depth (m)"""
        channel = self.channels[-1] # first channel is the same as last channel of input
        x = channel.x; y = channel.y; z = channel.z
        W = channel.w[0];
        if len(D)==0: 
            D = channel.d[0]
        else:
            D = D[0]
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
            x, y = migrate_one_step(x,y,z,W,kl,dt,k,Cf,D,pad,pad1,omega,gamma)
        
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
                cutoff = Cutoff(xc,yc,zc,W,D) # create cutoff object
                self.cutoffs.append(cutoff)
            # saving centerlines:
            if np.mod(itn,saved_ts)==0:
                self.cl_times.append(last_cl_time+(itn+1)*dt/(365*24*60*60.0))
                channel = Channel(x,y,z,W,D) # create channel object
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

        return fig

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

def migrate_one_step(x,y,z,W,kl,dt,k,Cf,D,pad,pad1,omega,gamma):
    ns=len(x)
    curv = compute_curvature(x,y)
    dx, dy, dz, ds, s = compute_derivatives(x,y,z)
    sinuosity = s[-1]/(x[-1]-x[0])
    curv = W*curv # dimensionless curvature
    R0 = kl*curv # simple linear relationship between curvature and nominal migration rate
    alpha = k*2*Cf/D # exponent for convolution function G
    R1 = compute_migration_rate(pad,ns,ds,alpha,omega,gamma,R0)
    R1 = sinuosity**(-2/3.0)*R1
    # calculate new centerline coordinates:
    dy_ds = dy[pad1:ns-pad+1]/ds[pad1:ns-pad+1]
    dx_ds = dx[pad1:ns-pad+1]/ds[pad1:ns-pad+1]
    # adjust x and y coordinates (this *is* the migration):
    x[pad1:ns-pad+1] = x[pad1:ns-pad+1] + R1[pad1:ns-pad+1]*dy_ds*dt  
    y[pad1:ns-pad+1] = y[pad1:ns-pad+1] - R1[pad1:ns-pad+1]*dx_ds*dt 
    return x,y

#@numba.jit(nopython=True) # use Numba to speed up the heaviest computation
def compute_migration_rate(pad,ns,ds,alpha,omega,gamma,R0):
    """compute migration rate as weighted sum of upstream curvatures
    pad - padding (number of nodepoints along centerline)
    ns - number of points in centerline
    ds - distances between points in centerline
    omega - constant in HK model
    gamma - constant in HK model
    R0 - nominal migration rate (dimensionless curvature * migration rate constant)"""
    R1 = np.zeros(ns) # preallocate adjusted channel migration rate
    pad1 = int(pad/10.0) # padding at upstream end can be shorter than padding on downstream end
    for i in range(pad1,ns-pad):
        si2 = np.hstack((np.array([0]),np.cumsum(ds[i-1::-1])))  # distance along centerline, backwards from current point 
        G = np.exp(-alpha*si2) # convolution vector
        R1[i] = omega*R0[i] + gamma*np.sum(R0[i::-1]*G)/np.sum(G) # main equation
    return R1

def compute_derivatives(x,y,z):
    """function for computing first derivatives of a curve (centerline)
    x,y are cartesian coodinates of the curve
    outputs:
    dx - first derivative of x coordinate
    dy - first derivative of y coordinate
    ds - distances between consecutive points along the curve
    s - cumulative distance along the curve"""
    dx = np.gradient(x) # first derivatives
    dy = np.gradient(y)   
    dz = np.gradient(z)   
    ds = np.sqrt(dx**2+dy**2+dz**2)
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
    dx = np.gradient(x) # first derivatives
    dy = np.gradient(y)      
    ddx = np.gradient(dx) # second derivatives 
    ddy = np.gradient(dy) 
    curvature = (dx*ddy-dy*ddx)/((dx**2+dy**2)**1.5)
    return curvature

def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    [from: https://stackoverflow.com/questions/16834861/create-own-colormap-using-matplotlib-and-plot-color-scale]
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

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
    
def find_cutoffs(x,y,crdist,deltas):
    """function for identifying locations of cutoffs along a centerline
    and the indices of the segments that will become part of the oxbows
    x,y - coordinates of centerline
    crdist - critical cutoff distance
    deltas - distance between neighboring points along the centerline"""
    diag_blank_width = int((crdist+20*deltas)/deltas)
    # distance matrix for centerline points:
    dist = distance.cdist(np.array([x,y]).T,np.array([x,y]).T)
    dist[dist>crdist] = np.NaN # set all values that are larger than the cutoff threshold to NaN
    # set matrix to NaN along the diagonal zone:
    for k in range(-diag_blank_width,diag_blank_width+1):
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

def dist_map(x,y,z,xmin,xmax,ymin,ymax,dx,delta_s):
    """function for centerline rasterization and distance map calculation
    inputs:
    x,y,z - coordinates of centerline
    xmin, xmax, ymin, ymax - x and y coordinates that define the area of interest
    dx - gridcell size (m)
    delta_s - distance between points along centerline (m)
    returns:
    cl_dist - distance map (distance from centerline)
    x_pix, y_pix, z_pix - x,y, and z pixel coordinates of the centerline
    s_pix - along-channel distance in pixels
    z_map - map of reference channel thalweg elevation (elevation of closest point along centerline)
    x, y, z - x,y,z centerline coordinates clipped to the 3D model domain"""
    y = y[(x>xmin) & (x<xmax)]
    z = z[(x>xmin) & (x<xmax)]
    x = x[(x>xmin) & (x<xmax)] 
    dummy,dy,dz,ds,s = compute_derivatives(x,y,z)
    if len(np.where(ds>2*delta_s)[0])>0:
        inds = np.where(ds>2*delta_s)[0]
        inds = np.hstack((0,inds,len(x)))
        lengths = np.diff(inds)
        long_segment = np.where(lengths==max(lengths))[0][0]
        start_ind = inds[long_segment]+1
        end_ind = inds[long_segment+1]
        if end_ind<len(x):
            x = x[start_ind:end_ind]
            y = y[start_ind:end_ind]
            z = z[start_ind:end_ind] 
        else:
            x = x[start_ind:]
            y = y[start_ind:]
            z = z[start_ind:]
    xdist = xmax - xmin
    ydist = ymax - ymin
    iwidth = int((xmax-xmin)/dx)
    iheight = int((ymax-ymin)/dx)
    xratio = iwidth/xdist
    # create list with pixel coordinates:
    pixels = []
    for i in range(0,len(x)):
        px = int(iwidth - (xmax - x[i]) * xratio)
        py = int(iheight - (ymax - y[i]) * xratio)
        pixels.append((px,py))
    # create image and numpy array:
    img = Image.new("RGB", (iwidth, iheight), "white")
    draw = ImageDraw.Draw(img)
    draw.line(pixels, fill="rgb(0, 0, 0)") # draw centerline as black line
    pix = np.array(img)
    cl = pix[:,:,0]
    cl[cl==255] = 1 # set background to 1 (centerline is 0)
    y_pix,x_pix = np.where(cl==0) 
    x_pix,y_pix = order_cl_pixels(x_pix,y_pix)
    # This next block of code is kind of a hack. Looking for, and eliminating, 'bad' pixels.
    img = np.array(img)
    img = img[:,:,0]
    img[img==255] = 1 
    img1 = morphology.binary_dilation(img, morphology.square(2)).astype(np.uint8)
    if len(np.where(img1==0)[0])>0:
        x_pix, y_pix = eliminate_bad_pixels(img,img1)
        x_pix,y_pix = order_cl_pixels(x_pix,y_pix) 
    img1 = morphology.binary_dilation(img, np.array([[1,0,1],[1,1,1]],dtype=np.uint8)).astype(np.uint8)
    if len(np.where(img1==0)[0])>0:
        x_pix, y_pix = eliminate_bad_pixels(img,img1)
        x_pix,y_pix = order_cl_pixels(x_pix,y_pix)
    img1 = morphology.binary_dilation(img, np.array([[1,0,1],[0,1,0],[1,0,1]],dtype=np.uint8)).astype(np.uint8)
    if len(np.where(img1==0)[0])>0:
        x_pix, y_pix = eliminate_bad_pixels(img,img1)
        x_pix,y_pix = order_cl_pixels(x_pix,y_pix)
    #redo the distance calculation (because x_pix and y_pix do not always contain all the points in cl):
    cl[cl==0] = 1
    cl[y_pix,x_pix] = 0
    cl_dist, inds = ndimage.distance_transform_edt(cl, return_indices=True)
    dx,dy,dz,ds,s = compute_derivatives(x,y,z)
    dx_pix = np.diff(x_pix)
    dy_pix = np.diff(y_pix)
    ds_pix = np.sqrt(dx_pix**2+dy_pix**2)
    s_pix = np.hstack((0,np.cumsum(ds_pix)))
    f = scipy.interpolate.interp1d(s,z)
    snew = s_pix*s[-1]/s_pix[-1]
    if snew[-1]>s[-1]:
        snew[-1]=s[-1]
    snew[snew<s[0]]=s[0]
    z_pix = f(snew)
    # create z_map:
    z_map = np.zeros(np.shape(cl_dist)) 
    z_map[y_pix,x_pix]=z_pix
    xinds=inds[1,:,:]
    yinds=inds[0,:,:]
    for i in range(0,len(x_pix)):
        z_map[(xinds==x_pix[i]) & (yinds==y_pix[i])] = z_pix[i]
    return cl_dist, x_pix, y_pix, z_pix, s_pix, z_map, x, y, z

def erosion_surface(h,w,cl_dist,z):
    """function for creating a parabolic erosional surface
    inputs:
    h - geomorphic channel depth (m)
    w - geomorphic channel width (in pixels, as cl_dist is also given in pixels)
    cl_dist - distance map (distance from centerline)
    z - reference elevation (m)
    returns:
    surf - map of the quadratic erosional surface (m)
    """
    surf = z + (4*h/w**2)*(cl_dist+w*0.5)*(cl_dist-w*0.5)
    return surf

def point_bar_surface(cl_dist,z,h,w):
    """function for creating a Gaussian-based point bar surface
    used in 3D fluvial model
    inputs:
    cl_dist - distance map (distance from centerline)
    z - reference elevation (m)
    h - channel depth (m)
    w - channel width, in pixels, as cl_dist is also given in pixels
    returns:
    pb - map of the Gaussian surface that can be used to from a point bar deposit (m)"""
    pb = z-h*np.exp(-(cl_dist**2)/(2*(w*0.33)**2))
    return pb

def sand_surface(surf,bth,dcr,z_map,h):
    """function for creating the top horizontal surface sand-rich deposit in the bottom of the channel
    used in 3D submarine channel models
    inputs:
    surf - current geomorphic surface
    bth - thickness of sand deposit in axis of channel (m)
    dcr - critical channel depth, above which there is no sand deposition (m)
    z_map - map of reference channel thalweg elevation (elevation of closest point along centerline)
    h - channel depth (m)
    returns:
    th - thickness map of sand deposit (m)
    relief - map of channel relief (m)"""
    relief = abs(surf-z_map+h)
    relief = abs(relief-np.amin(relief))
    th = bth * (1 - relief/dcr) # bed thickness inversely related to relief
    th[th<0] = 0.0 # set negative th values to zero
    return th, relief

def mud_surface(h_mud,levee_width,cl_dist,w,z_map,topo):
    """function for creating a map of overbank deposit thickness
    inputs:
    h_mud - maximum thickness of overbank deposit (m)
    levee_width - half-width of overbank deposit (m)
    cl_dist - distance map (distance from centerline)
    w - channel width (in pixels, as cl_dist is also given in pixels)
    z_map - map of reference channel thalweg elevation (elevation of closest point along centerline)
    topo - current geomorphic surface
    returns:
    surf - map of overbank deposit thickness (m)"""
    # create a surface that thins linearly away from the channel centerline:
    surf1 = (-2*h_mud/levee_width)*cl_dist+h_mud;
    surf2 = (2*h_mud/levee_width)*cl_dist+h_mud;
    surf = np.minimum(surf1,surf2)
    # surface for 'eroding' the central part of the mud layer:
    surf3 = h_mud + (4*1.5*h_mud/w**2)*(cl_dist+w*0.5)*(cl_dist-w*0.5) 
    surf = np.minimum(surf,surf3)
    surf[surf<0] = 0; # eliminate negative thicknesses
    return surf

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

def cl_dist_map(x,y,z,xmin,xmax,ymin,ymax,dx):
    """function for centerline rasterization and distance map calculation (does not return zmap)
    used for cutoffs only 
    inputs:
    x,y,z - coordinates of centerline
    xmin, xmax, ymin, ymax - x and y coordinates that define the area of interest
    dx - gridcell size (m)
    returns:
    cl_dist - distance map (distance from centerline)
    x_pix, y_pix, - x and y pixel coordinates of the centerline
    """
    y = y[(x>xmin) & (x<xmax)]
    z = z[(x>xmin) & (x<xmax)]
    x = x[(x>xmin) & (x<xmax)]    
    xdist = xmax - xmin
    ydist = ymax - ymin
    iwidth = int((xmax-xmin)/dx)
    iheight = int((ymax-ymin)/dx)
    xratio = iwidth/xdist
    # create list with pixel coordinates:
    pixels = []
    for i in range(0,len(x)):
        px = int(iwidth - (xmax - x[i]) * xratio)
        py = int(iheight - (ymax - y[i]) * xratio)
        pixels.append((px,py))
    # create image and numpy array:
    img = Image.new("RGB", (iwidth, iheight), "white")
    draw = ImageDraw.Draw(img)
    draw.line(pixels, fill="rgb(0, 0, 0)") # draw centerline as black line
    pix = np.array(img)
    cl = pix[:,:,0]
    cl[cl==255] = 1 # set background to 1 (centerline is 0)
    # calculate Euclidean distance map:
    cl_dist, inds = ndimage.distance_transform_edt(cl, return_indices=True)
    y_pix,x_pix = np.where(cl==0)
    return cl_dist, x_pix, y_pix

def eliminate_bad_pixels(img,img1):
    x_ind = np.where(img1==0)[1][0]
    y_ind = np.where(img1==0)[0][0]
    img[y_ind:y_ind+2,x_ind:x_ind+2] = np.ones(1,).astype(np.uint8)
    all_labels = measure.label(img,background=1,connectivity=2)
    cl=all_labels.copy()
    cl[cl==2]=0
    cl[cl>0]=1
    y_pix,x_pix = np.where(cl==1)
    return x_pix, y_pix

def order_cl_pixels(x_pix,y_pix):
    dist = distance.cdist(np.array([x_pix,y_pix]).T,np.array([x_pix,y_pix]).T)
    dist[np.diag_indices_from(dist)]=100.0
    ind = np.argmin(x_pix) # select starting point on left side of image
    clinds = [ind]
    count = 0
    while count<len(x_pix):
        t = dist[ind,:].copy()
        if len(clinds)>2:
            t[clinds[-2]]=t[clinds[-2]]+100.0
            t[clinds[-3]]=t[clinds[-3]]+100.0
        ind = np.argmin(t)
        clinds.append(ind)
        count=count+1
    x_pix = x_pix[clinds]
    y_pix = y_pix[clinds]
    return x_pix,y_pix
