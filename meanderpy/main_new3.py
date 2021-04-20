import meanderpyp2 as mp
import numpy as np
import matplotlib.pyplot as plt

ONE_YEAR = 365*24*60*60.0

L = 20000
ds = 100

x = np.linspace(0, L, int(L/ds) + 1)
y = 500 * np.exp(( 1.0 / L) * x) * np.cos((x / L) * 16 * np.pi) / (np.exp((x - 0.75 * L) / (0.025 * L)) + 1)

z = np.tan(5.0 * np.pi / 180) / (2 * L) * (x ** 2 + L * ( L - 2 * x ) )

def dep_props(slope):
    p = slope / -5 # 1 - > 0
    return (0.3, (2 - p) * 0.5, p * 0.2)

def aggr_props(slope):
    p = slope / -5 # 1 - > 0
    return (0.1, (2 - p) * 0.2, (p)* 0.7)

events = [
    mp.ChannelEvent(nit = 200, saved_ts = 25, mode='INCISION', kv = 0.005, dep_props = dep_props, aggr_props=aggr_props),
    mp.ChannelEvent(nit = 200, saved_ts = 25, mode='AGGRADATION', aggr_factor=2, kv = 0.01, dep_props = dep_props, aggr_props=aggr_props),
]

events[0].plot_all_relations();plt.show()

channel = mp.Channel(x, y)
basin = mp.Basin(x, z)

belt = mp.ChannelBelt(channel, basin)
for event in events:
    belt.simulate(event)
belt.plot_basin();plt.show()
belt.plot();plt.show()

model = belt.build_3d_model(25)

def plots():
    for xsec in [0.1, 0.30, 0.50, 0.75, 0.9]:
        model.plot_xsection(xsec, 3)
        plt.show()

plots()
#model.plot()
#model.render()
model.export(ve = 3)