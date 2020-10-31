import meanderpy as mp
import meanderpyp as mpn
import matplotlib.pyplot as plt
import numpy as np


W2, W1, W0 = 695.4154350661511, -45.231656699536124, 104.60941780103624
D2, D1 = 9.914006824924071, 26.588599799775164

L = 20000                     # channel length (m)
W = 200.0                     # constant channel width (m)
D = 12.0                      # constant channel depth (m)

pad = 5                    # padding (number of nodepoints along centerline)
deltas = 100.0                # sampling distance along centerline
nit = 1500                   # number of iterations
Cf = 0.02                    # dimensionless Chezy friction factor
crdist = 1.5*W               # threshold distance at which cutoffs occur
kl = 60.0/(365*24*60*60.0)   # migration rate constant (m/s)
kv =  1.0E-11               # vertical slope-dependent erosion rate constant (m/s)
dt = 4 * 0.05*365*24*60*60.0     # time step (s)
dens = 1000                  # density of water (kg/m3)
saved_ts = int(nit/25) + 1                # which time steps will be saved
n_bends = 15                 # approximate number of bends you want to model
Sl = 0.01                     # initial slope (matters more for submarine channels than rivers)
t1 = 1500                    # time step when incision starts
t2 = 1700                    # time step when lateral migration starts
t3 = 11000                    # time step when aggradation starts
aggr_factor = 4.0          # aggradation factor (it kicks in after t3)


ds = deltas
x = np.linspace(0, L, int(L/ds))
z = np.tan(5.0 * np.pi / 180) * (L/2  + x * ( x / (2*L) - 1))
dz = np.gradient(z) / ds

w = W2 * np.exp(- W1 * dz) + W0
d = D2 * np.exp(- D1 * dz)

y = w * np.sin( x / 875 )
#plt.plot(x,y)
#plt.plot(x, d)
plt.show()

chp = mpn.Channel(x, y, z, w, d)
chbp = mpn.ChannelBelt(channels=[chp],cutoffs=[],cl_times=[0.0],cutoff_times=[]) 

ch = mp.Channel(x, y, z, 200, 12)
chb = mp.ChannelBelt(channels=[ch],cutoffs=[],cl_times=[0.0],cutoff_times=[]) # create channel belt object

chbp.migrate(nit,saved_ts,deltas,pad,crdist,Cf,kl,kv,dt,dens,t1,t2,t3,aggr_factor)
chb.migrate(nit,saved_ts,deltas,pad,crdist,Cf,kl,kv,dt,dens,t1,t2,t3,aggr_factor)
chb.plot('strat',20,60, 0, 0)
chbp.plot()
plt.show()
