import meanderpy as mp
import numpy as np
import matplotlib.pyplot as plt
import cases

L = 5000
ds = 100

x = np.linspace(0, L, int(L/ds) + 1)
y = 250 * np.exp(( 1.0 / L) * x) * np.cos((x / L) * 16 * np.pi) / (np.exp((x - 0.75 * L) / (0.025 * L)) + 1) / (np.exp((x - 0.75 * L) / (0.025 * L)) + 1)
z = np.tan(2.0 * np.pi / 180) * (L - x)

events = [
  mp.ChannelEvent(
    nit = 100, saved_ts = 25, Cf = 0.02, mode='INCISION', kv = 0.0033 * 10, 
    dep_props= lambda x: (0.9, 0, 0.1), aggr_props= lambda x: (0, 1, 0)),
  mp.ChannelEvent(
    nit = 150, saved_ts = 25, Cf = 0.02, mode='AGGRADATION', kv = 0, 
    dep_props= lambda x: (0.9, 0, 0.1), aggr_props= lambda x: (0, 1, 0)),
  mp.ChannelEvent(
    nit = 150, saved_ts = 15, Cf = 0.02, mode='AGGRADATION', aggr_factor=2, 
    kv = 0.0033 * 5, dep_props= lambda x: (0.9, 0, 0.1), aggr_props= lambda x: (0, 1, 0)),
  mp.ChannelEvent(
    nit = 150, saved_ts = 15, Cf = 0.02, mode='AGGRADATION', kv = 0, 
    dep_props= lambda x: (0.9, 0, 0.1), aggr_props= lambda x: (0, 1, 0)),
]

channel = mp.Channel(x, y)
basin = mp.Basin(x, z)

belt = mp.ChannelBelt(channel, basin)

for evt in events:
  belt.simulate(evt)

belt.plot();plt.show()
belt.plot_basin();plt.show()

model = belt.build_3d_model(25)

model.plot_xsection(0.230, ve = 3); plt.show()
