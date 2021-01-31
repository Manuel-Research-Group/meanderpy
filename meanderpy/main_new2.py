import meanderpyp2 as mp
import numpy as np
import matplotlib.pyplot as plt
import cases

L = 20000
ds = 100

x = np.linspace(0, L, int(L/ds) + 1)
y = 200 * np.exp(( 1.0 / L) * x) * np.sin((x / L) * 16 * np.pi) / (np.exp((x - 0.75 * L) / (0.025 * L)) + 1)

z = np.tan(5.0 * np.pi / 180) / (2 * L) * (x ** 2 + L * ( L - 2 * x ) )

event1 = mp.ChannelEvent(nit = 250, saved_ts = 25, mode='AGGREGATION', aggr_factor=4)
event2 = mp.ChannelEvent(nit = 250, saved_ts = 25, mode='INCISION')

channel = mp.Channel(x, y)
basin = mp.Basin(x, z)

belt = mp.ChannelBelt(channel, basin)
belt.simulate(event1)
belt.simulate(event2)
#belt.plot();plt.show()
model = belt.build_3d_model(25)


def plots():
    for xsec in [0.1, 0.30, 0.50, 0.75, 0.9]:
        model.plot_xsection(xsec, 3)
        plt.show()

plots()
#model.plot()
model.render()