import meanderpy as mp
import numpy as np
import matplotlib.pyplot as plt
import cases

ONE_YEAR = 365*24*60*60.0

L = 20000
ds = 100

W = 200.0                    # channel width (m)
D = 12.0                     # channel depth (m)
pad = 50                    # padding (number of nodepoints along centerline)
deltas = 100.0                # sampling distance along centerline
nit = 500                   # number of iterations
Cf = 0.02                    # dimensionless Chezy friction factor
crdist = 1.5*W               # threshold distance at which cutoffs occur
kl = 60.0/ONE_YEAR   # migration rate constant (m/s)
kv =  1.0E-11               # vertical slope-dependent erosion rate constant (m/s)
dt = 2*0.05*ONE_YEAR     # time step (s)
dens = 1000                  # density of water (kg/m3)
saved_ts = 20                # which time steps will be saved
n_bends = 50                 # approximate number of bends you want to model
Sl = 0.01                     # initial slope (matters more for submarine channels than rivers)
t1 = 500                    # time step when incision starts
t2 = 700                    # time step when lateral migration starts
t3 = 1000                    # time step when aggradation starts
aggr_factor = 4.0          # aggradation factor (it kicks in after t3)

x = np.linspace(0, L, int(L/ds) + 1)
y = 250 * np.exp(( 1.0 / L) * x) * np.cos((x / L) * 16 * np.pi) / (np.exp((x - 0.75 * L) / (0.025 * L)) + 1)
z = np.tan(5.0 * np.pi / 180) / (2 * L) * (x ** 2 + L * ( L - 2 * x ) )

ch = mp.Channel(x, y, z, W, D)
chb = mp.ChannelBelt(channels=[ch],cutoffs=[],cl_times=[0.0],cutoff_times=[]) # create channel belt object

chb.migrate(nit,saved_ts,deltas,pad,crdist,Cf,kl,kv,dt,dens,t1,t2,t3,aggr_factor) # channel migration
plt.show()
fig = chb.plot('strat',20,60) # plotting
plt.show()