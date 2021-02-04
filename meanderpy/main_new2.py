import meanderpyp2 as mp
import numpy as np
import matplotlib.pyplot as plt
import cases

ONE_YEAR = 365*24*60*60.0

L = 50000
ds = 100

x = np.linspace(0, L, int(L/ds) + 1)
y = 500 * np.exp(( 1.0 / L) * x) * np.cos((x / L) * 16 * np.pi) / (np.exp((x - 0.75 * L) / (0.025 * L)) + 1)

z = np.tan(5.0 * np.pi / 180) / (2 * L) * (x ** 2 + L * ( L - 2 * x ) )

event1 = mp.ChannelEvent(nit = 100, saved_ts = 25, mode='INCISION', kv = 0.0033 / ONE_YEAR)
event2 = mp.ChannelEvent(nit = 100, saved_ts = 25, mode='AGGREGATION', aggr_factor=2, kv = 0.002 / ONE_YEAR)
event3 = mp.ChannelEvent(nit = 100, saved_ts = 25, mode='INCISION', kv = 0.0033 / ONE_YEAR)
event4 = mp.ChannelEvent(nit = 100, saved_ts = 25, mode='AGGREGATION', aggr_factor=2, kv = 0.002 / ONE_YEAR)
event5 = mp.ChannelEvent(nit = 100, saved_ts = 25, mode='INCISION', kv = 0.002 / ONE_YEAR)
event6 = mp.ChannelEvent(nit = 100, saved_ts = 25, mode='AGGREGATION', aggr_factor=2, kv = 0.0033 / ONE_YEAR)
event7 = mp.ChannelEvent(nit = 100, saved_ts = 25, mode='INCISION', kv = 0.002 / ONE_YEAR)
event8 = mp.ChannelEvent(nit = 100, saved_ts = 25, mode='AGGREGATION', aggr_factor=2, kv = 0.0033 / ONE_YEAR)

channel = mp.Channel(x, y)
basin = mp.Basin(x, z)

fig, axis = plt.subplots(1, 1)
axis.set_autoscale_on(True) # enable autoscale
axis.autoscale_view(True,True,True)

line, = plt.plot([], []) # Plot blank data

belt = mp.ChannelBelt(channel, basin)
belt.simulate(event1, axis, line)
belt.simulate(event2, axis, line)
belt.simulate(event3, axis, line)
belt.simulate(event4, axis, line)
belt.simulate(event5, axis, line)
belt.simulate(event6, axis, line)
belt.simulate(event7, axis, line)
belt.simulate(event8, axis, line)
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