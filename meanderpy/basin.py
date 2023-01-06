import matplotlib.pyplot as plt
import seaborn as sns
import scipy.interpolate
import numpy as np

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