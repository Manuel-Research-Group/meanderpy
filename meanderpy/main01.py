import meanderpy as mp
import numpy as np
import matplotlib.pyplot as plt

ONE_YEAR = 365*24*60*60.0

L = 20000
ds = 100

x = np.linspace(0, L, int(L/ds) + 1)
y = 500 * np.exp(( 1.0 / L) * x) * np.cos((x / L) * 16 * np.pi) / (np.exp((x - 0.75 * L) / (0.025 * L)) + 1)

z = np.tan(5.0 * np.pi / 180) / (2 * L) * (x ** 2 + L * ( L - 2 * x ) )

events = [
    mp.ChannelEvent(nit = 100, saved_ts = 25, mode='INCISION', kv = 0.0033 / ONE_YEAR),
    mp.ChannelEvent(nit = 100, saved_ts = 25, mode='AGGREGATION', aggr_factor=2, kv = 0.002 / ONE_YEAR),
    mp.ChannelEvent(nit = 100, saved_ts = 25, mode='INCISION', kv = 0.0033 / ONE_YEAR),
    mp.ChannelEvent(nit = 100, saved_ts = 25, mode='AGGREGATION', aggr_factor=2, kv = 0.002 / ONE_YEAR),
    mp.ChannelEvent(nit = 100, saved_ts = 25, mode='INCISION', kv = 0.002 / ONE_YEAR),
    mp.ChannelEvent(nit = 100, saved_ts = 25, mode='AGGREGATION', aggr_factor=2, kv = 0.0033 / ONE_YEAR),
    mp.ChannelEvent(nit = 100, saved_ts = 25, mode='INCISION', kv = 0.002 / ONE_YEAR),
    mp.ChannelEvent(nit = 100, saved_ts = 25, mode='AGGREGATION', aggr_factor=2, kv = 0.0033 / ONE_YEAR)
]

channel = mp.Channel(x, y)
basin = mp.Basin(x, z)

belt = mp.ChannelBelt(channel, basin)
for event in events:
    belt.simulate(event)
belt.plot();plt.show()

model = belt.build_3d_model(25)

def plots():
    for xsec in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        model.plot_xsection(xsec, 3)
        plt.show()

#plots()
#model.plot()
#model.render()
model.export_objs(ve = 3)