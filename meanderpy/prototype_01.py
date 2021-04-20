import meanderpyp2 as mp
import numpy as np
import matplotlib.pyplot as plt
import cases

FOLDER = 'figures'

L = 20000
ds = 100

x = np.linspace(0, L, int(L/ds) + 1)
y = 250 * np.exp(( 1.0 / L) * x) * np.cos((x / L) * 16 * np.pi) / (np.exp((x - 0.75 * L) / (0.025 * L)) + 1) / (np.exp((x - 0.75 * L) / (0.025 * L)) + 1)

z = np.tan(5.0 * np.pi / 180) / (2 * L) * (x ** 2 + L * ( L - 2 * x ) )

def dep_props(slope):
  p = slope / -5 # 1 - > 0
  return (0.3, (2 - p) * 0.5, p * 0.2)

# ideia de fazer a agradação ter silt no final

def aggr_props_inc(slope):
  p = slope / -5 # 1 - > 0
  return (0.1, (2 - p) * 0.2, (p)* 0.7)

def aggr_props_aggr(slope):
  p = slope / -5 # 1 - > 0
  return (0.1, (2 - p) * 0.2, (p + 0.2)* 0.7)

events = [
  mp.ChannelEvent(nit = 100, saved_ts = 25, Cf = 0.02, mode='INCISION', kv = 0.0033, dep_props = dep_props, aggr_props=aggr_props_inc),
  mp.ChannelEvent(nit = 100, saved_ts = 25, Cf = 0.02, mode='AGGRADATION', aggr_factor=2, kv = 0.002, dep_props = dep_props, aggr_props=aggr_props_aggr),
  mp.ChannelEvent(nit = 100, saved_ts = 25, Cf = 0.02, mode='INCISION', kv = 0.0033, dep_props = dep_props, aggr_props=aggr_props_inc),
  mp.ChannelEvent(nit = 100, saved_ts = 25, Cf = 0.02, mode='AGGRADATION', aggr_factor=2, kv = 0.002, dep_props = dep_props, aggr_props=aggr_props_aggr),
  mp.ChannelEvent(nit = 100, saved_ts = 25, Cf = 0.02, mode='INCISION', kv = 0.002, dep_props = dep_props, aggr_props=aggr_props_inc),
  mp.ChannelEvent(nit = 100, saved_ts = 25, Cf = 0.02, mode='AGGRADATION', aggr_factor=2, kv = 0.0033, dep_props = dep_props, aggr_props=aggr_props_aggr),
  mp.ChannelEvent(nit = 100, saved_ts = 25, Cf = 0.02, mode='INCISION', kv = 0.002, dep_props = dep_props, aggr_props=aggr_props_inc),
  mp.ChannelEvent(nit = 100, saved_ts = 25, Cf = 0.02, mode='AGGRADATION', aggr_factor=2, kv = 0.0033, dep_props = dep_props, aggr_props=aggr_props_aggr)
]
channel = mp.Channel(x, y)
basin = mp.Basin(x, z)

belt = mp.ChannelBelt(channel, basin)
events[0].plot_all_relations();plt.show()

for evt in events:
  belt.simulate(evt)

fig = belt.plot()
fig.savefig('{}/2D.png'.format(FOLDER), transparent=True, bbox_inches='tight', pad_inches = 0, dpi = 300)
model = belt.build_3d_model(25)
for xsec in [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]:
    for ve in [3]:
        fig = model.plot_xsection(xsec, ve)
        fig.savefig('{}/section-{:d}m_ve-{}.png'.format(FOLDER, int(xsec * L), ve), transparent=True, bbox_inches='tight', pad_inches = 0, dpi = 300)
        plt.close(fig)
        fig = model.plot_xsection(xsec, ve, substrat=False)
        fig.savefig('{}/section-{:d}m_ve-{}_no-sub.png'.format(FOLDER, int(xsec * L), ve), transparent=True, bbox_inches='tight', pad_inches = 0, dpi = 300)
        plt.close(fig)
    
