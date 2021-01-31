import cases
import meanderpyp as mp
import matplotlib.pyplot as plt
import numpy as np

PARAMS = cases.ChannelScatteredSineRampSlope()
FOLDER = 'figures'

scale = np.exp((PARAMS.x - 0.75 * PARAMS.L) / (0.025 * PARAMS.L)) + 1
ch = mp.Channel(PARAMS.x, PARAMS.y/scale, PARAMS.z, PARAMS.w, PARAMS.d)
chb = mp.ChannelBelt(ch) 

dx = 25

chb.migrate(720,15,PARAMS.ds,PARAMS.pad,PARAMS.crdist,PARAMS.Cf,PARAMS.kl,PARAMS.kv, PARAMS.dt,PARAMS.density,PARAMS.t1,PARAMS.t2,PARAMS.t3,PARAMS.aggr_factor)

model = chb.build_3d_model(dx)
model.plot(ve = 3, curvature = True)

fig = chb.plot()
fig.savefig('{}/channel.png'.format(FOLDER), transparent=True, bbox_inches='tight', pad_inches = 0, dpi = 300)
for xsec in [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]:
    for ve in [1,2,5]:
        fig = model.plot_xsection(xsec, ve)
        fig.savefig('{}/section-{:d}m_ve-{}.png'.format(FOLDER, int(xsec * PARAMS.L), ve), transparent=True, bbox_inches='tight', pad_inches = 0, dpi = 300)
        plt.close(fig)
        fig = model.plot_xsection(xsec, ve, substrat=False)
        fig.savefig('{}/section-{:d}m_ve-{}_no-sub.png'.format(FOLDER, int(xsec * PARAMS.L), ve), transparent=True, bbox_inches='tight', pad_inches = 0, dpi = 300)
        plt.close(fig)
    
