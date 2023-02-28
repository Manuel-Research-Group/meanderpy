import numpy as np
import scipy.interpolate
from scipy.spatial import distance
import matplotlib.pyplot as plt
import seaborn as sns

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

        self.OMEGA = -1.0 # constant in curvature calculation (Howard and Knutson, 1984)
        self.GAMMA = 2.5  # from Ikeda et al., 1981 and Howard and Knutson, 1984
        self.K = 4.0 # constant in HK equation

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

    def compute_migration_rate(self, r0, Cf, d, dl, L):
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
                G = np.exp(-2.0 * self.K * Cf / d[i] * SIGMA_2) # convolution vector
                r1[i] = self.OMEGA*r0[i] + self.GAMMA*np.sum(r0[i::-1]*G)/np.sum(G) # main equation
            else:
                r1[i] = r0[i]
        return r1

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
        R1 = self.compute_migration_rate(R0, Cf, self.d, ds, s[-1])

        RN = sinuosity**(-2/3.0) * R1 * (area / np.max(area))
        #plt.plot(self.x, R0, self.x, R1, self.x, RN);plt.show()
        #plt.plot(self.x, (area / np.max(area)));plt.show()
        
        self.x += RN * (dy/ds) * dt 
        self.y -= RN * (dx/ds) * dt

    def find_cutoffs(self, x, y, crdist, diag):
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

    def find_cutoffs_R(self, R, W = 5, T = 1):
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
 
    def cut_cutoffs(self, crdist, ds):
        """
        TODO

        :param crdist: TODO
        :param ds: TODO  
        :return: TODO              
        """
        
        cuts = []   
        
        diag_blank_width = int((crdist+20*ds)/ds)
        # UPPER MARGIN
        xo, yo = self.margin_offset()
        ind1, ind2 = self.find_cutoffs(self.x+xo, self.y+yo, crdist, diag_blank_width)
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
            ind1, ind2 = self.find_cutoffs(self.x+xo, self.y+yo, crdist, diag_blank_width)

        # LOWER MARGIN
        xo, yo = self.margin_offset()
        ind1, ind2 = self.find_cutoffs(self.x-xo, self.y-yo, crdist, diag_blank_width)
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
            ind1, ind2 = self.find_cutoffs(self.x-xo, self.y-yo, crdist, diag_blank_width)

        return cuts

    def cut_cutoffs_R(self, cut_window, ds):
        """
        TODO

        :param cut_window: TODO
        :param ds: TODO
        """

        D = int(cut_window / (2 * ds))
        ind1, ind2 = self.find_cutoffs_R(self.w / 2 * self.curvature(), D)
        if ind1 != -1:
            self.x = np.hstack((self.x[:ind1+1],self.x[ind2:])) # x coordinates after cutoff
            self.y = np.hstack((self.y[:ind1+1],self.y[ind2:])) # y coordinates after cutoff
            self.z = np.hstack((self.z[:ind1+1],self.z[ind2:])) # z coordinates after cutoff
            self.w = np.hstack((self.w[:ind1+1],self.w[ind2:])) # z coordinates after cutoff
            self.d = np.hstack((self.d[:ind1+1],self.d[ind2:])) # z coordinates after cutoff
            ind1, ind2 = self.find_cutoffs_R(self.w * self.curvature(), D)

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