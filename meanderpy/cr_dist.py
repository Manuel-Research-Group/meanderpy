import cases
import meanderpy as mp
import meanderpyp as mpn
import matplotlib.pyplot as plt
import numpy as np

PARAMS = cases.ChannelScatteredSineRampSlope()

scale = np.exp((PARAMS.x - 0.75 * PARAMS.L) / (0.025 * PARAMS.L)) + 1
chp = mpn.Channel(PARAMS.x, PARAMS.y/scale, PARAMS.z, PARAMS.w, PARAMS.d)
chbp = mpn.ChannelBelt(chp) 


#chb.migrate(1,PARAMS.saved_ts,PARAMS.ds,PARAMS.pad,PARAMS.crdist,PARAMS.Cf,PARAMS.kl,PARAMS.kv,PARAMS.dt,PARAMS.density,PARAMS.t1,PARAMS.t2,PARAMS.t3,PARAMS.aggr_factor)
#chb.plot('strat', 0, 0, 0, 0)
#plt.show()
#h_mud = 3.0*np.ones((len(chb.cl_times[0:]),))
#chbp.plot()
#plt.plot(chp.x, chp.z)
#plt.show()
dx = 25

chbp.migrate(100, PARAMS.dt, 10, PARAMS.crdist)
chbp.plot()
plt.show()


model = chbp.build_3d_model(dx)

model.plot(curvature = True, ve = 3)

#chb.build_3d_model('submarine',h_mud=h_mud,levee_width=5000.0,h=12.0,w=PARAMS.W,bth=6.0, dcr=7.0,dx=dx,delta_s=PARAMS.ds,starttime=chb.cl_times[0],endtime=chb.cl_times[-1], xmin=1,xmax=20000,ymin=-2500,ymax=2500)