import pyvista as pv
import meanderpy as mp
import numpy as np

N = 10
L = 2000
W = 100
D = 12

pad = 0                    # padding (number of nodepoints along centerline)
deltas = L/N                # sampling distance along centerline
nit = 50                   # number of iterations
Cf = 0.02                    # dimensionless Chezy friction factor
crdist = 1.5*W               # threshold distance at which cutoffs occur
kl = 60.0/(365*24*60*60.0)   # migration rate constant (m/s)
kv =  1.0E-11               # vertical slope-dependent erosion rate constant (m/s)
dt = 2 * 0.05*365*24*60*60.0     # time step (s)
dens = 1000                  # density of water (kg/m3)
saved_ts = 10                # which time steps will be saved
n_bends = 15                 # approximate number of bends you want to model
Sl = 0.01                     # initial slope (matters more for submarine channels than rivers)
t1 = 1500                    # time step when incision starts
t2 = 1700                    # time step when lateral migration starts
t3 = 11000                    # time step when aggradation starts
aggr_factor = 4.0          # aggradation factor (it kicks in after t3)



x = np.linspace(0, L, N)
y = W * np.sin(x / L * np.pi * 2)
z = np.zeros(N)

ch = mp.Channel(x, y, z, W, D)
chb = mp.ChannelBelt([ch], [], [0.0], [])
chb.migrate(nit,saved_ts,deltas,pad,crdist,Cf,kl,kv,dt,dens,t1,t2,t3,aggr_factor)

chb.plot('strat',20,60, 0, 0)

h_mud = 3.0*np.ones((len(chb.cl_times[0:]),))
dx = 50 # gridcell size in meters

model, xmin, xmax, ymin, ymax = chb.build_3d_model('submarine',h_mud=h_mud,levee_width=4000.0,h=12.0,w=W,bth=0.0,
                            dcr=10.0,dx=dx,delta_s=deltas,starttime=chb.cl_times[0],endtime=chb.cl_times[-1],
                            xmin=-1,xmax=L,ymin=-2*W,ymax=2*W)

print(model.topo.shape, model.strat.shape)

x = np.linspace(xmin, xmax, model.strat.shape[1])
y = np.linspace(ymin, ymax, model.strat.shape[0])

xx, yy, zz = np.meshgrid(x, y, np.linspace(ymin, ymax, model.strat.shape[2]))

grid = pv.StructuredGrid(xx, yy, model.topo[:,:,0:19])

grid.plot()