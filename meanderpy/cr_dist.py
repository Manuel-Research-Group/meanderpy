import cases
import meanderpy as mp
import meanderpyp as mpn
import matplotlib.pyplot as plt
import numpy as np

PARAMS = cases.ChannelScatteredSineRampSlope()


chp = mpn.Channel(PARAMS.x, PARAMS.y, PARAMS.z, PARAMS.w, PARAMS.d)
chbp = mpn.ChannelBelt(channels=[chp],cutoffs=[],cl_times=[0.0],cutoff_times=[]) 

ch = mp.Channel(PARAMS.x, PARAMS.y, PARAMS.z, PARAMS.W, PARAMS.D)
chb = mp.ChannelBelt(channels=[ch],cutoffs=[],cl_times=[0.0],cutoff_times=[]) # create channel belt object

#chb.migrate(1,PARAMS.saved_ts,PARAMS.ds,PARAMS.pad,PARAMS.crdist,PARAMS.Cf,PARAMS.kl,PARAMS.kv,PARAMS.dt,PARAMS.density,PARAMS.t1,PARAMS.t2,PARAMS.t3,PARAMS.aggr_factor)
#chb.plot('strat', 0, 0, 0, 0)
#plt.show()
#h_mud = 3.0*np.ones((len(chb.cl_times[0:]),))
#chbp.plot()
plt.plot(chp.x, chp.z)
plt.show()
dx = 15

chbp.migrate(1,PARAMS.saved_ts,PARAMS.ds,PARAMS.pad,PARAMS.crdist,PARAMS.Cf,PARAMS.kl,PARAMS.kv,PARAMS.dt,PARAMS.density,PARAMS.t1,PARAMS.t2,PARAMS.t3,PARAMS.aggr_factor)
chbp.image_test(dx)
#chb.build_3d_model('submarine',h_mud=h_mud,levee_width=5000.0,h=12.0,w=PARAMS.W,bth=6.0, dcr=7.0,dx=dx,delta_s=PARAMS.ds,starttime=chb.cl_times[0],endtime=chb.cl_times[-1], xmin=1,xmax=20000,ymin=-2500,ymax=2500)