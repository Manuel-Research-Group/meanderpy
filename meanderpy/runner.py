from scipy.interpolate import interp1d, splrep, splev
import scipy.interpolate as si

import matplotlib.pyplot as plt
import meanderpy as mp
import numpy as np
import json
import sys

### FILES
CHANNELS_FILE = './channels.json'
EVENTS_FILE = './events.json'
CONFIG_FILE = './config.json'

### DEFAULTS

###  -CHANNELS
DEFAULT_SAMPLE_RATE = 50

DEFAULT_CHANNEL_WIDTH = 0
DEFAULT_CHANNEL_LENGTH = 0
DEFAULT_CHANNEL_ELEVATION = 0

DEFAULT_CHANNEL_SINUOSITY = []
DEFAULT_CHANNEL_SLOPE = []

###  -EVENTS 
DEFAULT_EVENT_NIT = 100       # Number of iterations
DEFAULT_EVENT_SAVED_TS = 25   # Saved 
DEFAULT_EVENT_DT = 0.1
DEFAULT_EVENT_MODE = 'AGGREGATION'
DEFAULT_EVENT_KV = 0.01
DEFAULT_EVENT_KL = 60.0
DEFAULT_EVENT_CR_DIST = 200
DEFAULT_EVENT_CR_WIND = 1500

DEFAULT_EVENT_CH_DEPTH = [
  {"slope": -5, "value": 100},
  {"slope": -4, "value": 80},
  {"slope": -3, "value": 60},
  {"slope": -2, "value": 40},
  {"slope": -1, "value": 20},
  {"slope": 00, "value": 0}
]
DEFAULT_EVENT_CH_WIDTH =  [
  {"slope": -5, "value": 100},
  {"slope": -4, "value": 125},
  {"slope": -3, "value": 150},
  {"slope": -2, "value": 250},
  {"slope": -1, "value": 400},
  {"slope": 00, "value": 800}
]

DEFAULT_EVENT_DEP_HEIGHT = [
  {"slope": -5, "value": 25},
  {"slope": -4, "value": 20},
  {"slope": -3, "value": 15},
  {"slope": -2, "value": 10},
  {"slope": -1, "value": 5},
  {"slope": 00, "value": 0}
]
DEFAULT_EVENT_DEP_PROPS = [
  {"slope": -5, "value": [0.3, 0.5, 0.2]},
  {"slope": -4, "value": [0.3, 0.5, 0.2]},
  {"slope": -3, "value": [0.3, 0.5, 0.2]},
  {"slope": -2, "value": [0.3, 0.5, 0.2]},
  {"slope": -1, "value": [0.3, 0.5, 0.2]},
  {"slope": 00, "value": [0.3, 0.5, 0.2]}
]
DEFAULT_EVENT_DEP_SIGMAS = [
  {"slope": -5, "value": [0.25, 0.5, 2]},
  {"slope": -4, "value": [0.25, 0.5, 2]},
  {"slope": -3, "value": [0.25, 0.5, 2]},
  {"slope": -2, "value": [0.25, 0.5, 2]},
  {"slope": -1, "value": [0.25, 0.5, 2]},
  {"slope": 0, "value": [0.25, 0.5, 2]}
]

DEFAULT_EVENT_AGGR_PROPS = [
  {"slope": -5, "value": [0.333, 0.333, 0.333]},
  {"slope": -4, "value": [0.333, 0.333, 0.333]},
  {"slope": -3, "value": [0.333, 0.333, 0.333]},
  {"slope": -2, "value": [0.333, 0.333, 0.333]},
  {"slope": -1, "value": [0.333, 0.333, 0.333]},
  {"slope": 0, "value": [0.333, 0.333, 0.333]}
]
DEFAULT_EVENT_AGGR_SIGMAS = [
  {"slope": -5, "value": [2, 5, 10]},
  {"slope": -4, "value": [2, 5, 10]},
  {"slope": -3, "value": [2, 5, 10]},
  {"slope": -2, "value": [2, 5, 10]},
  {"slope": -1, "value": [2, 5, 10]},
  {"slope": 0, "value": [2, 5, 10]}
]

### -CONFIGS

DEFAULT_CONFIG_VE = 3
DEFAULT_CONFIG_GRID = 25
DEFAULT_CONFIG_MARGIN = 500
DEFAULT_CONFIG_CROSS_SECTIONS = []
DEFAULT_CONFIG_TITLE = ''
DEFAULT_CONFIG_PREVIEW = False
DEFAULT_CONFIG_RENDER = False
DEFAULT_CONFIG_EXPORT = False

### AUXILIAR FUNCTIONS
def create_tabular_param(param):
  first_value = param[0]['value']
  slope = [p['slope'] for p in param]
  if isinstance(first_value, list):
    L = len(first_value)
    values = [p['value'] for p in param]
    value = []
    for i in range(L):
      value.append(
        [value[i] for value in values]
      )
    return si.interp1d(slope, value, fill_value="extrapolate")
  else:
    value = [p['value'] for p in param]
    return si.interp1d(slope, value, fill_value="extrapolate")

# Code adapted from Dennis:
# https://stackoverflow.com/questions/55808363/how-can-i-give-specific-x-values-to-scipy-interpolate-splev
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splrep.html#scipy.interpolate.splrep

def b_spline_eval(param, l, dx, degree=3):
  xCtrlPoints = [value[0] for value in param]
  yCtrlPoints = [value[1] for value in param]
  xp = xCtrlPoints
  yp = yCtrlPoints    

  length = len(xp)  

  minY = min(yp)  
  maxY = max(yp)      

  t = np.linspace(0., xp[-1], length - (degree-1), endpoint=True)
  t = np.append([0, 0, 0], t)
  t = np.append(t, [xp[-1], xp[-1], xp[-1]])    

  sx = si.BSpline(t, xp, degree)    
  sy = si.BSpline(t, yp, degree)

  xEval = []
  yEval = []    

  for i in range(0, l, int(dx)):        
    x0 = i
    u0 = si.PPoly.from_spline((sx.t, sx.c - x0, degree)).roots()        
    nRoot = len(sx(u0))

    xEval.append(sx(u0)[0])

    # Select the correct roots from sy(u0)
    for i in range(nRoot):
      if sy(u0)[i] >= minY and sy(u0)[i] <= maxY:
        yEval.append(sy(u0)[i])
  
  return xEval, yEval

def plot2D(x, y, title, ylabel):
  plt.plot(x, y)
  plt.title(title)
  plt.xlabel('Length (m)')
  plt.ylabel(ylabel)
  plt.show()

channels_file = open(CHANNELS_FILE, 'r')
events_file = open(EVENTS_FILE, 'r')
config_file = open(CONFIG_FILE, 'r')

channels_json = json.load(channels_file)
events_json = json.load(events_file)
config_json = json.load(config_file)

##### CHANNEL-BASIN #####

width = channels_json.get('width', DEFAULT_CHANNEL_WIDTH)
length = channels_json.get('length', DEFAULT_CHANNEL_LENGTH)
elevation = channels_json.get('elevation', DEFAULT_CHANNEL_ELEVATION)

sinuosity = channels_json.get('sinuosity', DEFAULT_CHANNEL_SINUOSITY)
slope = channels_json.get('slope', DEFAULT_CHANNEL_SLOPE)

channel_x, channel_y = b_spline_eval(sinuosity, length, DEFAULT_SAMPLE_RATE)
basin_x, basin_z = b_spline_eval(slope, length, DEFAULT_SAMPLE_RATE)

preview = config_json.get('preview', DEFAULT_CONFIG_PREVIEW)

if(preview):
  plot2D(channel_x, channel_y, 'Initial Channel Preview', 'Width (m)')
  plot2D(basin_x, basin_z, 'Initial Channel Preview', 'Elevation (m)')

channel = mp.Channel(channel_x, channel_y)
basin = mp.Basin(basin_x, basin_z)

### EVENTS
events = [] 
for evt in events_json:
  nit = evt.get('nit', DEFAULT_EVENT_NIT)
  saved_ts = evt.get('saved_ts', DEFAULT_EVENT_SAVED_TS)
  dt = evt.get('dt', DEFAULT_EVENT_DT)
  mode = evt.get('mode', DEFAULT_EVENT_MODE)
  kv = evt.get('kv', DEFAULT_EVENT_KV)
  kl = evt.get('kl', DEFAULT_EVENT_KL)
  cr_dist = evt.get('cr_dist', DEFAULT_EVENT_CR_DIST)
  cr_wind = evt.get('cr_wind', DEFAULT_EVENT_CR_WIND)

  ch_depth = evt.get('ch_depth', DEFAULT_EVENT_CH_DEPTH)
  ch_width = evt.get('ch_width', DEFAULT_EVENT_CH_WIDTH)

  dep_height = evt.get('dep_height', DEFAULT_EVENT_DEP_HEIGHT)

  dep_props = evt.get('dep_props', DEFAULT_EVENT_DEP_PROPS)
  dep_sigmas = evt.get('dep_sigmas', DEFAULT_EVENT_DEP_SIGMAS)
  
  aggr_props = evt.get('aggr_props', DEFAULT_EVENT_AGGR_PROPS)
  aggr_sigmas = evt.get('aggr_sigmas', DEFAULT_EVENT_AGGR_SIGMAS)
  
  event = mp.ChannelEvent(
    nit = nit,
    saved_ts = saved_ts,
    dt = dt,
    mode = mode,
    kv = kv,
    kl = kl,
    cr_dist = cr_dist,
    cr_wind = cr_wind,
    ch_depth = create_tabular_param(ch_depth),
    ch_width = create_tabular_param(ch_width),
    dep_height = create_tabular_param(dep_height),
    dep_props = create_tabular_param(dep_props),
    dep_sigmas = create_tabular_param(dep_sigmas),
    aggr_props = create_tabular_param(aggr_props),
    aggr_sigmas = create_tabular_param(aggr_sigmas),
  )
  events.append(event)

### RUN

belt = mp.ChannelBelt(channel, basin)
for i, event in enumerate(events):
  print('Simulating event {} of {}'.format(i, len(events)))
  belt.simulate(event)

### CONFIG
ve = config_json.get('ve', DEFAULT_CONFIG_VE)
grid = config_json.get('dxdy', DEFAULT_CONFIG_GRID)
margin = config_json.get('margin', DEFAULT_CONFIG_MARGIN)
cross_sections = config_json.get('cross_sections', DEFAULT_CONFIG_CROSS_SECTIONS)
title = config_json.get('title', DEFAULT_CONFIG_TITLE)
render = config_json.get('render', DEFAULT_CONFIG_RENDER)
export = config_json.get('export', DEFAULT_CONFIG_EXPORT)

print('Building 3D model using {} meters grid'.format(grid))
model = belt.build_3d_model(grid, margin)

if len(cross_sections) > 0:
  print('Rendering {} cross-section images'.format(len(cross_sections)))

for xsec in cross_sections:
  print('- Cross-section @ {}'.format(xsec))
  model.plot_xsection(
    xsec = xsec, 
    ve = ve, 
    title = title
  )
  plt.show()

if export:
  print('Exporting 3D model')
  model.export_objs(ve = ve)

if render:
  print('Rendering 3D model')
  model.render()