import meanderpyp2 as mp
import numpy as np
import matplotlib.pyplot as plt
import cases

L = 20000
ds = 100

x = np.linspace(0, L, int(L/ds) + 1)
y = 250 * np.exp(( 1.0 / L) * x) * np.cos((x / L) * 16 * np.pi) / (np.exp((x - 0.75 * L) / (0.025 * L)) + 1) / (np.exp((x - 0.75 * L) / (0.025 * L)) + 1)
z = np.tan(5.0 * np.pi / 180) / (2 * L) * (x ** 2 + L * ( L - 2 * x ) )

events_a = [
  mp.ChannelEvent(
    nit = 150, saved_ts = 25, Cf = 0.02, mode='INCISION', kv = 0.0033 * 12, 
    dep_props=lambda x: (0, 0.85, 0.15)),
  mp.ChannelEvent(
    nit = 50, saved_ts = 10, Cf = 0.02, mode='AGGRADATION', aggr_factor=1, kv = 0.0033 * 15, 
    aggr_props=lambda x: (0.4, 0.6, 0.0), dep_props= lambda x: (0.6, 0.3, 0.1)),
  mp.ChannelEvent(
    nit = 50, saved_ts = 10, Cf = 0.02, mode='AGGRADATION', aggr_factor=1, kv = 0.0033 * 15, 
    aggr_props=lambda x: (0, 0.9, 0.1), dep_props= lambda x: (0.8, 0.0, 0.2)),
  mp.ChannelEvent(
    nit = 100, saved_ts = 15, Cf = 0.02, mode='AGGRADATION', aggr_factor=1, kv = 0.0033 * 15, 
    aggr_props=lambda x: (0, 0.75, 0.25), dep_props= lambda x: (0.8, 0.0, 0.2)),
]

events_b = [
  mp.ChannelEvent(
    nit = 150, saved_ts = 25, Cf = 0.02, mode='INCISION', kv = 0.0033 * 5, 
    aggr_props=lambda x: (0.8, 0.2, 0), dep_props= lambda x: (0.8, 0.2, 0)),
  mp.ChannelEvent(
    nit = 50, saved_ts = 10, Cf = 0.02, mode='AGGRADATION', aggr_factor=2, kv = 0.002 * 5, 
    aggr_props=lambda x: (0.1, 0.9, 0), dep_props= lambda x: (0.8, 0.2, 0)),
  mp.ChannelEvent(
    nit = 100, saved_ts = 25, Cf = 0.02, mode='AGGRADATION', aggr_factor=2, kv = 0.002 * 5, 
    aggr_props=lambda x: (0, 1, 0), dep_props= lambda x: (0.25, 0.75, 0.0)),
]

channel = mp.Channel(x, y)
basin = mp.Basin(x, z)

belt_a = mp.ChannelBelt(channel, basin)
belt_b = mp.ChannelBelt(channel, basin)

for evt in events_a:
  belt_a.simulate(evt)

for evt in events_b:
  belt_b.simulate(evt)

model_a = belt_a.build_3d_model(25)
model_b = belt_b.build_3d_model(25)

model_a.plot_xsection(0.65, ve = 10, silt_color = [63/255, 169/255, 75/255], gravel_color = [195/255, 147/255, 82/255]); plt.show()
model_b.plot_xsection(0.69, ve = 5, silt_color = [63/255, 169/255, 75/255], gravel_color = [195/255, 147/255, 82/255]); plt.show()
