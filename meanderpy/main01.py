import meanderpy as mp
import numpy as np
import matplotlib.pyplot as plt
import cases

PARAMS = cases.ChannelScatteredSineRampSlope()

L = 20000
ds = 100

x = np.linspace(0, L, int(L/ds) + 1)
y = 100 * np.exp(( 1.0 / L) * x) * np.sin((x / L) * 16 * np.pi) / (np.exp((x - 0.75 * L) / (0.025 * L)) + 1)
z = np.tan(5.0 * np.pi / 180) / (2 * L) * (x ** 2 + L * ( L - 2 * x ) )

event1 = mp.ChannelEvent(nit = 150, saved_ts = 15, dt = 0.15 * 365*24*60*60.0)
event2 = mp.ChannelEvent(nit = 200, saved_ts = 20, mode='INCISION')
event3 = mp.ChannelEvent(nit = 200, saved_ts = 20, mode='AGGRADATION')

scale = np.exp((PARAMS.x - 0.75 * PARAMS.L) / (0.025 * PARAMS.L)) + 1
#ch = mp.Channel(x, y, z)
ch = mp.Channel(PARAMS.x, PARAMS.y/scale, PARAMS.z)
ch.refit(ds)

channels = mp.ChannelBelt(ch)
channels.simulate(event1)
#channels.simulate(event3)
channels.plot()
plt.show()


model = channels.build_3d_model(25)

def plots():
    for xsec in [0.1, 0.30, 0.60, 0.90]:
        model.plot_xsection(xsec, 3)
        plt.show()

#model.plot(curvature = True, ve = 3, save = True)
model.render()
