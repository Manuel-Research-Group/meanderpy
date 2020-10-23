import meanderpy as mp
import meanderpyp as mpp
import matplotlib.pyplot as plt
import numpy as np


L = 20000                     # channel length (m)
W = 200.0                     # constant channel width (m)
D = 12.0                      # constant channel depth (m)

pad = 0                       # padding (number of nodepoints along centerline)
deltas = 1000.0                # sampling distance along centerline
nit = 150                    # number of iterations
Cf = 0.02                     # dimensionless Chezy friction factor
crdist = 1.5*W                # threshold distance at which cutoffs occur
kl = 60.0/(365*24*60*60.0)    # migration rate constant (m/s)
kv =  1.0E-11                 # vertical slope-dependent erosion rate constant (m/s)
dt = 10 * 2*0.05*365*24*60*60.0    # time step (s)
dens = 1000                   # density of water (kg/m3)
saved_ts = 10                 # which time steps will be saved
n_bends = 15                  # approximate number of bends you want to model
Sl = 0.01                     # initial slope (matters more for submarine channels than rivers)
t1 = 150                      # time step when incision starts
t2 = 170                      # time step when lateral migration starts
t3 = 1100                     # time step when aggradation starts
aggr_factor = 4.0             # aggradation factor (it kicks in after t3)

ds = deltas
x = np.linspace(0, L, int(L/ds))
y = 10 * W * np.exp(- (1 / L) * x) * np.sin( x * 10 / L )
#z = np.tan(5.0 * np.pi / 180) * (L - x)
z = np.zeros(len(x))

plt.plot(x, y)
plt.plot(x, z)
plt.show()

ch = mp.Channel(x, y, z, W, D)
chb = mp.ChannelBelt(channels=[ch],cutoffs=[],cl_times=[0.0],cutoff_times=[]) 

chb.migrate(nit,saved_ts,deltas,pad,crdist,Cf,kl,kv, dt,dens,t1,t2,t3,aggr_factor)

chn = mpp.Channel(x, y, z, W * np.ones(len(x)), D * np.ones(len(x)))
chbn = mpp.ChannelBelt(channels=[chn],cutoffs=[],cl_times=[0.0],cutoff_times=[])

chbn.migrate(nit,saved_ts,deltas,pad,crdist,Cf,kl,kv, dt,dens,t1,t2,t3,aggr_factor)

chb.plot('strat',20,60, 0, 0)
chbn.plot()
plt.show()
