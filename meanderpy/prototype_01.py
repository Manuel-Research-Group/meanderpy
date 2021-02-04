import cases
import meanderpyp2 as mp
import matplotlib.pyplot as plt
import numpy as np

FOLDER = 'figures'
dx = 25
ONE_YEAR = 365*24*60*60.0

L = 20000
ds = 100

x = np.linspace(0, L, int(L/ds) + 1)
y = 250 * np.exp(( 1.0 / L) * x) * np.cos((x / L) * 16 * np.pi) / (np.exp((x - 0.75 * L) / (0.025 * L)) + 1)

z = np.tan(5.0 * np.pi / 180) / (2 * L) * (x ** 2 + L * ( L - 2 * x ) )

event1 = mp.ChannelEvent(nit = 100, saved_ts = 25, Cf = 0.02, dep_offset=10, mode='INCISION', kv = 0.0033 / ONE_YEAR)
event2 = mp.ChannelEvent(nit = 100, saved_ts = 25, Cf = 0.02, dep_offset=10, mode='AGGREGATION', aggr_factor=2, kv = 0.002 / ONE_YEAR)
event3 = mp.ChannelEvent(nit = 100, saved_ts = 25, Cf = 0.02, dep_offset=10, mode='INCISION', kv = 0.0033 / ONE_YEAR)
event4 = mp.ChannelEvent(nit = 100, saved_ts = 25, Cf = 0.02, dep_offset=10, mode='AGGREGATION', aggr_factor=2, kv = 0.002 / ONE_YEAR)
event5 = mp.ChannelEvent(nit = 100, saved_ts = 25, Cf = 0.02, dep_offset=10, mode='INCISION', kv = 0.002 / ONE_YEAR)
event6 = mp.ChannelEvent(nit = 100, saved_ts = 25, Cf = 0.02, dep_offset=10, mode='AGGREGATION', aggr_factor=2, kv = 0.0033 / ONE_YEAR)
event7 = mp.ChannelEvent(nit = 100, saved_ts = 25, Cf = 0.02, dep_offset=10, mode='INCISION', kv = 0.002 / ONE_YEAR)
event8 = mp.ChannelEvent(nit = 100, saved_ts = 25, Cf = 0.02, dep_offset=10, mode='AGGREGATION', aggr_factor=2, kv = 0.0033 / ONE_YEAR)

channel = mp.Channel(x, y)
basin = mp.Basin(x, z)

fig, axis = plt.subplots(1, 1)
axis.set_autoscale_on(True) # enable autoscale
axis.autoscale_view(True,True,True)

line, = plt.plot([], []) # Plot blank data

belt = mp.ChannelBelt(channel, basin)
belt.simulate(event1)
belt.simulate(event2)
belt.simulate(event3)
belt.simulate(event4)
belt.simulate(event5)
belt.simulate(event6)
belt.simulate(event7)
belt.simulate(event8)

model = belt.build_3d_model(25)
for xsec in [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]:
    for ve in [3]:
        fig = model.plot_xsection(xsec, ve)
        fig.savefig('{}/section-{:d}m_ve-{}.png'.format(FOLDER, int(xsec * L), ve), transparent=True, bbox_inches='tight', pad_inches = 0, dpi = 300)
        plt.close(fig)
        fig = model.plot_xsection(xsec, ve, substrat=False)
        fig.savefig('{}/section-{:d}m_ve-{}_no-sub.png'.format(FOLDER, int(xsec * L), ve), transparent=True, bbox_inches='tight', pad_inches = 0, dpi = 300)
        plt.close(fig)
    
