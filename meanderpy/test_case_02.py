import meanderpyp2 as mp
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import cases

L = 30000
ds = 100

x_ = [0, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000, 22500, 25000, 27500, 30000]
y_ = [0, 300,  0,    -600,   600,   0,     500,  -350,  -500, -100,      100,   0,     0]
z_ = [1135, 920, 750, 570, 395, 315, 215, 129, 86, 43, 0, 0, 0]

x = np.linspace(0, L, int(L/ds) + 1)

def dep_props(slope):
    p = slope / -5 # 1 - > 0
    return (0.3, (2 - p) * 0.5, p * 0.2)

def aggr_props(slope):
    p = slope / -5 # 1 - > 0
    return (0.1, (2 - p) * 0.2, (p)* 0.7)

events = [
  mp.ChannelEvent(nit = 100, saved_ts = 25, Cf = 0.02, mode='INCISION', kv = 0.0033, dep_props = dep_props, aggr_props=aggr_props),
  mp.ChannelEvent(nit = 100, saved_ts = 25, Cf = 0.02, mode='AGGRADATION', aggr_factor=2, kv = 0.002, dep_props = dep_props, aggr_props=aggr_props),
  mp.ChannelEvent(nit = 100, saved_ts = 25, Cf = 0.02, mode='INCISION', kv = 0.0033, dep_props = dep_props, aggr_props=aggr_props),
  mp.ChannelEvent(nit = 100, saved_ts = 25, Cf = 0.02, mode='AGGRADATION', aggr_factor=2, kv = 0.002, dep_props = dep_props, aggr_props=aggr_props),
  mp.ChannelEvent(nit = 100, saved_ts = 25, Cf = 0.02, mode='INCISION', kv = 0.002, dep_props = dep_props, aggr_props=aggr_props),
  mp.ChannelEvent(nit = 100, saved_ts = 25, Cf = 0.02, mode='AGGRADATION', aggr_factor=2, kv = 0.0033, dep_props = dep_props, aggr_props=aggr_props),
  mp.ChannelEvent(nit = 100, saved_ts = 25, Cf = 0.02, mode='INCISION', kv = 0.002, dep_props = dep_props, aggr_props=aggr_props),
  mp.ChannelEvent(nit = 100, saved_ts = 25, Cf = 0.02, mode='AGGRADATION', aggr_factor=2, kv = 0.0033, dep_props = dep_props, aggr_props=aggr_props)
]

channel = mp.Channel(x, scipy.interpolate.interp1d(x_, y_, kind = 'cubic')(x))
channel.plot();plt.show()
basin = mp.Basin(x, scipy.interpolate.interp1d(x_, z_)(x))
basin.plot();plt.show()
belt = mp.ChannelBelt(channel, basin)

for evt in events:
  belt.simulate(evt)

belt.plot();plt.show()
belt.plot_basin();plt.show()


model = belt.build_3d_model(25)

def plots():
  for xsec in [0.1, 0.30, 0.50, 0.75, 0.8, 0.85, 0.9]:
    model.plot_xsection(xsec, 3)
    plt.show()

plots()
#model.plot()
#model.render()
model.export(ve = 3)