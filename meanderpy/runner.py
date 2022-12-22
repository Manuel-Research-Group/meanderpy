from scipy.interpolate import interp1d, splrep, splev
import scipy.interpolate as si

import matplotlib.pyplot as plt
import meanderpy as mp
import numpy as np
import json
import tempfile
from os import path, walk
from zipfile import ZipFile
from shutil import copyfile, rmtree
import sys
import time # for debugging purposes only

# DEBUG code:
# print('DEBUG1:', type(xCoordPoints[0]))
# time.sleep(10)

### FILE PATHS: contains the paths to the three files required to running the code
CHANNELS_FILE = './channels.json'
EVENTS_FILE = './events.json'
CONFIG_FILE = './config.json'

### DEFAULTS CHANNEL PROPERTIES: these values are used if no values are found in the input file "channels.json"
DEFAULT_SAMPLE_RATE = 50
DEFAULT_CHANNEL_WIDTH = 0
DEFAULT_CHANNEL_LENGTH = 0
DEFAULT_CHANNEL_ELEVATION = 0
DEFAULT_CHANNEL_SINUOSITY = []
DEFAULT_CHANNEL_SLOPE = []

### DEFAULT CHANNEL CONFIGS: these values are used if no values are found in the input file "config.json"
DEFAULT_CONFIG_VE = 3
DEFAULT_CONFIG_GRID = 25
DEFAULT_CONFIG_MARGIN = 500
DEFAULT_CONFIG_CROSS_SECTIONS = []
DEFAULT_CONFIG_SHOW_SECTIONS = False
DEFAULT_CONFIG_TITLE = ''
DEFAULT_CONFIG_PREVIEW = False
DEFAULT_CONFIG_RENDER = False
DEFAULT_CONFIG_EXPORT = False

### DEFAULT CHANNEL EVENTS: these values are used if no values are found in the input file "events.json"
DEFAULT_EVENT_NIT = 100               # Number of iterations: total number of iterations for the simulation
DEFAULT_EVENT_SAVED_TS = 25           # Saved simlation state every _ iterations
DEFAULT_EVENT_DT = 0.1                # Time of each iteration (in years)
DEFAULT_EVENT_MODE = 'AGGRADATION'    # Type of event (aggradation, incision or separator)
DEFAULT_EVENT_KV = 0.01               # Aggradation/incision modulation (m/years): coefficient used to modulate the aggradation/incision aggressiveness in the equilibrium profile
DEFAULT_EVENT_KL = 60.0               # Meandering modulation (m/years): coefficient used to modlate the channel migration aggressiveness
DEFAULT_EVENT_CR_DIST = 200           # ???
DEFAULT_EVENT_CR_WIND = 1500          # ???
DEFAULT_NUMBER_LAYERS = 3             # Number of layers for the current event
DEFAULT_EVENT_CH_DEPTH = [            # Depth of the channel centerline in meters, per slope
  {"slope": -5, "value": 100},
  {"slope": -4, "value": 80},
  {"slope": -3, "value": 60},
  {"slope": -2, "value": 40},
  {"slope": -1, "value": 20},
  {"slope": 00, "value": 0}
]
DEFAULT_EVENT_CH_WIDTH =  [           # Width of the channel in meters, per slope
  {"slope": -5, "value": 100},
  {"slope": -4, "value": 125},
  {"slope": -3, "value": 150},
  {"slope": -2, "value": 250},
  {"slope": -1, "value": 400},
  {"slope": 00, "value": 800}
]
DEFAULT_EVENT_DEP_HEIGHT = [          # Deposition height in meters for the cut and fill process, per slope
  {"slope": -5, "value": 25},
  {"slope": -4, "value": 20},
  {"slope": -3, "value": 15},
  {"slope": -2, "value": 10},
  {"slope": -1, "value": 5},
  {"slope": 00, "value": 0}
]
DEFAULT_EVENT_DEP_PROPS = [           # Layer deposition distribution (%) during the cut and fill process, per slope
  {"slope": -5, "value": [0.3, 0.5, 0.2]},
  {"slope": -4, "value": [0.3, 0.5, 0.2]},
  {"slope": -3, "value": [0.3, 0.5, 0.2]},
  {"slope": -2, "value": [0.3, 0.5, 0.2]},
  {"slope": -1, "value": [0.3, 0.5, 0.2]},
  {"slope": 00, "value": [0.3, 0.5, 0.2]}
]
DEFAULT_EVENT_DEP_SIGMAS = [          # Gaussian standard deviation of deposition process during the cut and fill process per layer, per slope
  {"slope": -5, "value": [0.25, 0.5, 2]},
  {"slope": -4, "value": [0.25, 0.5, 2]},
  {"slope": -3, "value": [0.25, 0.5, 2]},
  {"slope": -2, "value": [0.25, 0.5, 2]},
  {"slope": -1, "value": [0.25, 0.5, 2]},
  {"slope": 0, "value": [0.25, 0.5, 2]}
]
DEFAULT_EVENT_AGGR_PROPS = [          # Distribution of the layers (%) in the aggradation process, per slope
  {"slope": -5, "value": [0.333, 0.333, 0.333]},
  {"slope": -4, "value": [0.333, 0.333, 0.333]},
  {"slope": -3, "value": [0.333, 0.333, 0.333]},
  {"slope": -2, "value": [0.333, 0.333, 0.333]},
  {"slope": -1, "value": [0.333, 0.333, 0.333]},
  {"slope": 0, "value": [0.333, 0.333, 0.333]}
]
DEFAULT_EVENT_AGGR_SIGMAS = [         # Gaussian standard deviation of aggradation process per layer, per slope
  {"slope": -5, "value": [2, 5, 10]},
  {"slope": -4, "value": [2, 5, 10]},
  {"slope": -3, "value": [2, 5, 10]},
  {"slope": -2, "value": [2, 5, 10]},
  {"slope": -1, "value": [2, 5, 10]},
  {"slope": 0, "value": [2, 5, 10]}
]
DEFAULT_EVENT_SEP_THICKNESS = 0       # Separator layer thickness, in meters
DEFAULT_EVENT_SEP_TYPE = 'CONDENSED_SECTION'  # Material of the separator (CONDENSED_SECTION, BASAL_SURFACE or INVERSION)

### AUXILIARY FUNCTIONS

def create_tabular_param(par):
  """
    Converts the JSON format parameters to another one created by interp1d function.

    :param par: TODO
    :return: TODO
  """
  first_value = par[0]['value']
  slope = [p['slope'] for p in par]
  if isinstance(first_value, list):
    L = len(first_value)
    values = [p['value'] for p in par]
    value = []
    for i in range(L):
      value.append(
        [value[i] for value in values]
      )
    return si.interp1d(slope, value, fill_value="extrapolate")
  else:
    value = [p['value'] for p in par]
    return si.interp1d(slope, value, fill_value="extrapolate")    

def correctRoots(rootsX, rootsY, yMin, yMax):
  """
    Auxiliary function for b_spline_eval. Decides the correct root values from sy(u0) given by si.Poly.from_spline function

    :param rootsX: TODO
    :param rootsY: TODO
    :param yMin: TODO
    :param yMax: TODO
    :return: TODO
  """

  if(len(rootsX) != len(rootsY) or len(rootsX[0]) != len(rootsY[0])):    
    raise Exception("[ERROR] [function verifyCorrectRootIndex]: rootsX and rootsY have different sizes.")  

  validRootX = []
  validRootY = []
  for i in range(len(rootsY[0])):
    correctOne = True
    for j in range(len(rootsY)):
      if rootsY[j][i] >= yMin and rootsY[j][i] <= yMax:
        validRootX.append(rootsX[j][i])        
        validRootY.append(rootsY[j][i])
      else:
        validRootX = []
        validRootY = []
        correctOne = False
        break

    if correctOne == True:
      break

  if correctOne == False:
    raise Exception("[ERROR] [function verifyCorrectRootIndex]: no valid root set found.")  
  
  return validRootX, validRootY  

def b_spline_eval(p, l, dx, degree=3):
  """
    Generates a list of points corresponding to a B-Spline.
    Source material 1: https://stackoverflow.com/questions/55808363/how-can-i-give-specific-x-values-to-scipy-interpolate-splev
    Source material 2: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splrep.html#scipy.interpolate.splrep

    :param p: TODO
    :param l: TODO
    :param dx: TODO
    :param degree: TODO
    :return: list of x and y coordinates of the points forming the Bezier curve.
  """ 
  xCtrlPoints = [value[0] for value in p]
  yCtrlPoints = [value[1] for value in p]
  xp = xCtrlPoints
  yp = yCtrlPoints    

  length = len(xp)  

  yMin = min(yp)  
  yMax = max(yp)      

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

    xEval.append(sx(u0))
    yEval.append(sy(u0))

  xEvalNew, yEvalNew = correctRoots(xEval, yEval, yMin, yMax)  
  
  return xEvalNew, yEvalNew
 
def plotChannelProfile2D(xCoordPoints, yCoordPoints, viewType):  
  """
    Code to plot two lists of 2D points representing a channe information given their x and y coordinates and its view type.
    The x label is fixed as the length in meters. The plot can represent either the channel top or side view.

    :param xCoordPoints: (list<float>) list of the input x-coordinate points.
    :param yCoordPoints: (list<float>) list of the input y-coordinate points.
    :param viewType: (string) decides which channel plot shall be saved. This param can be either 'TOP' or 'SIDE'.
  """  
  if viewType == 'TOP':
    title = 'Channel top view (sinuosity)'
    yLabel = 'Width (m)'
    fileName = 'channel_top_view'
  elif viewType == 'SIDE':
    title = 'Channel side view (slope)'
    yLabel = 'Elevation (m)'
    fileName = 'channel_side_view'
  else:
    raise Exception('Error: Incorrect viewType in function plotChannelProfile2D.')

  plt.plot(xCoordPoints, yCoordPoints)
  plt.title(title)
  plt.xlabel('Length (m)')
  plt.ylabel(yLabel)
  plt.savefig(fileName + '.png')

def zipFilesInDir(dirName, zipFileName, filter):
  """
    Compress all the files in a given directory to a zip file.

    :param dirName: (string) input name of the directory to be zipped.
    :param zipFileName: (string) output name of the zip generated.
    :param filter: TODO    
  """
  with ZipFile(zipFileName, 'w') as zipObj:
      for (folderName, _, filenames) in walk(dirName):
        for filename in filenames:
           if filter(filename):
              filePath = path.join(folderName, filename)
              zipObj.write(filePath, filename)

def generateStringFromEventModes(eventModeList):
  """
    Create a list of event modes (strings) to be incorpored into the title

    :param eventModeList: TODO
    :return: TODO
  """
  eventSimpleText = 'Events: '
  for e in eventModeList:
    if e == 'AGGRADATION':
      eventSimpleText = eventSimpleText + 'AGG, '
    elif e == 'INCISION':
      eventSimpleText = eventSimpleText + 'INC, '
    elif e == 'SEPARATOR':
      eventSimpleText = eventSimpleText + 'SEP, '
    else:
      raise Exception('Invalid event mode.')
  eventSimpleText = eventSimpleText[0:-2]  

  return eventSimpleText

def remove_values_from_list(the_list, val):
  """
    Remove all the occurrences of a value from a list
    Extracted from https://stackoverflow.com/questions/1157106/remove-all-occurrences-of-a-value-from-a-list

    :param the_list: list containing the values
    :param val: value to be found and removed (all its occurrences) in the list
    :return: new list without the value occurrences specified in val
  """
  return [value for value in the_list if value != val]

def round_values_from_list(the_list):
  """  
    Round all the float values in a list according to the round (math) function

    :param the_list: list containing the values
    :return: new list with all the previous values rounded
  """
  for i in range(len(the_list)):
    if type(the_list[i]) is float:
      the_list[i] = round(the_list[i],2)

  return the_list

def zero_values_from_list(the_list, val):
  """  
    Set the zero all the occurrences of the values specified by val in the list

    :param the_list: list containing the values
    :param val: value which occurrences should be set to zero in the list
    :return: new list with all the previous values set to zero
  """
  for i in range(len(the_list)):
    if the_list[i] == val:
      the_list[i] = 0
  
  round_values_from_list(the_list)

  return the_list

def preprocessSpecificEvents(ch_depth, ch_width, dep_height, dep_props, dep_sigmas, aggr_props, aggr_sigmas):
  """  
    Set the zero all the occurrences of the values specified by val in the list

    :param the_list: list containing the values
    :param val: value which occurrences should be set to zero in the list
    :return: new list with all the previous values set to zero
  """

  new_ch_depth = []
  new_ch_width = []
  new_dep_height = []
  new_dep_props = []
  new_dep_sigmas = []
  new_aggr_props = []
  new_aggr_sigmas = []

  for i in range(len(dep_props)):    
    dep_props[i]['value'] = zero_values_from_list(dep_props[i]['value'], 1e-06)      
    dep_sigmas[i]['value'] = zero_values_from_list(dep_sigmas[i]['value'], 1e-06)
    aggr_props[i]['value'] = zero_values_from_list(aggr_props[i]['value'], 1e-06)
    aggr_sigmas[i]['value'] = zero_values_from_list(aggr_sigmas[i]['value'], 1e-06)

    # Remove the 0 values (which were placeholders) from the lists
    #TODO
    dep_props[i]['value'] = remove_values_from_list(dep_props[i]['value'], 0)
    dep_sigmas[i]['value'] = remove_values_from_list(dep_sigmas[i]['value'], 0)
    aggr_props[i]['value'] = remove_values_from_list(aggr_props[i]['value'], 0)
    aggr_sigmas[i]['value'] = remove_values_from_list(aggr_sigmas[i]['value'], 0)
    
    new_ch_depth.append(ch_depth[i]['value'])
    new_ch_width.append(ch_width[i]['value'])
    new_dep_height.append(dep_height[i]['value'])
    new_dep_props.append(dep_props[i]['value'])
    new_dep_sigmas.append(dep_sigmas[i]['value'])
    new_aggr_props.append(aggr_props[i]['value'])
    new_aggr_sigmas.append(aggr_sigmas[i]['value'])
   
  return new_ch_depth, new_ch_width, new_dep_height, new_dep_props, new_dep_sigmas, new_aggr_props, new_aggr_sigmas  

# MAIN

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
  plotChannelProfile2D(channel_x, channel_y, 'TOP')
  plotChannelProfile2D(basin_x, basin_z, 'SIDE')

channel = mp.Channel(channel_x, channel_y)
basin = mp.Basin(basin_x, basin_z)

### EVENTS
# Dennis: create a new events variable called honestSpecificEvents, which contains the true values of
# channel depth, channel width, deposition height, layer deposition, gaussian deposition, layer aggradation
# and gaussian aggradation in string format instead of the ones created by create_tabular_param which become
# in format "interp1d"
events = [] 
honestSpecificEvents = []

for evt in events_json:
  nit = evt.get('nit', DEFAULT_EVENT_NIT)
  saved_ts = evt.get('saved_ts', DEFAULT_EVENT_SAVED_TS)
  dt = evt.get('dt', DEFAULT_EVENT_DT)
  mode = evt.get('mode', DEFAULT_EVENT_MODE)
  kv = evt.get('kv', DEFAULT_EVENT_KV)  
  kl = evt.get('kl', DEFAULT_EVENT_KL)
  number_layers = evt.get('number_layers', DEFAULT_EVENT_KL) # Added by Dennis

  cr_dist = evt.get('cr_dist', DEFAULT_EVENT_CR_DIST)
  cr_wind = evt.get('cr_wind', DEFAULT_EVENT_CR_WIND)

  ch_depth = evt.get('ch_depth', DEFAULT_EVENT_CH_DEPTH)
  ch_width = evt.get('ch_width', DEFAULT_EVENT_CH_WIDTH)

  dep_height = evt.get('dep_height', DEFAULT_EVENT_DEP_HEIGHT)

  dep_props = evt.get('dep_props', DEFAULT_EVENT_DEP_PROPS) # atualizar (sempre 7 + separator, demais zero)
  dep_sigmas = evt.get('dep_sigmas', DEFAULT_EVENT_DEP_SIGMAS)

  aggr_props = evt.get('aggr_props', DEFAULT_EVENT_AGGR_PROPS)
  aggr_sigmas = evt.get('aggr_sigmas', DEFAULT_EVENT_AGGR_SIGMAS)

  # Separator
  sep_thicnkess = evt.get('sep_thickness', DEFAULT_EVENT_SEP_THICKNESS)
  sep_type = evt.get('sep_type', DEFAULT_EVENT_SEP_TYPE)  

  old_ch_depth = ch_depth
  old_ch_width = ch_width
  old_dep_height = dep_height
  old_dep_props = dep_props
  old_dep_sigmas = dep_sigmas
  old_aggr_props = aggr_props
  old_aggr_sigmas = aggr_sigmas

  

  event = mp.ChannelEvent(
    nit = nit,
    saved_ts = saved_ts,
    dt = dt,
    mode = mode,
    kv = kv,
    kl = kl,
    number_layers = number_layers,
    cr_dist = cr_dist,
    cr_wind = cr_wind,
    ch_depth = create_tabular_param(ch_depth),
    ch_width = create_tabular_param(ch_width),
    dep_height = create_tabular_param(dep_height),
    dep_props = create_tabular_param(dep_props),
    dep_sigmas = create_tabular_param(dep_sigmas),
    aggr_props = create_tabular_param(aggr_props),
    aggr_sigmas = create_tabular_param(aggr_sigmas),
    sep_thickness = sep_thicnkess,
    sep_type = sep_type
  )
  events.append(event)

  

  # Dennis
  # Preprocessing in the strings to substitute the '1e-06' to zero
  ch_depth, ch_width, dep_height, dep_props, dep_sigmas, aggr_props , aggr_sigmas =  preprocessSpecificEvents(old_ch_depth, old_ch_width, \
                                                                                                              old_dep_height, old_dep_props, \
                                                                                                              old_dep_sigmas, old_aggr_props, \
                                                                                                              old_aggr_sigmas)  
  
  honestSpecificEvent = mp.ChannelSpecifics(ch_depth, ch_width, dep_height, dep_props, dep_sigmas, aggr_props, aggr_sigmas)
  honestSpecificEvents.append(honestSpecificEvent)

### RUN
belt = mp.ChannelBelt(channel, basin, honestSpecificEvents)
eventModeList = [] # Dennis: create a list of event modes (strings) to be incorpored into the title
for i, event in enumerate(events):
  print('Simulating event {} of {}'.format(i+1, len(events)))    
  belt.simulate(event, i)
  eventModeList.append(event.mode) # Dennis: create a list of event modes (strings) to be incorpored into the title
  # basin, channel, time, events information inside belt object

eventSimpleText = generateStringFromEventModes(eventModeList)

### CONFIG
ve = config_json.get('ve', DEFAULT_CONFIG_VE)
grid = config_json.get('dxdy', DEFAULT_CONFIG_GRID)
margin = config_json.get('margin', DEFAULT_CONFIG_MARGIN)
cross_sections = config_json.get('cross_sections', DEFAULT_CONFIG_CROSS_SECTIONS)
show_sections = config_json.get('show_sections', DEFAULT_CONFIG_SHOW_SECTIONS)
preview = config_json.get('preview', DEFAULT_CONFIG_PREVIEW)
title = config_json.get('title', DEFAULT_CONFIG_TITLE)
render = config_json.get('render', DEFAULT_CONFIG_RENDER)
export = config_json.get('export', DEFAULT_CONFIG_EXPORT)

print('Building 3D model using {} meters grid'.format(grid))
model = belt.build_3d_model(grid, margin)

if len(cross_sections) > 0:
  print('Rendering {} cross-section images'.format(len(cross_sections)))

# Variable introduced to show the cross sections
# Generate and organize the new matplotlib figures into a zip
if show_sections:
  cross_section_count = 0
  dir = tempfile.mkdtemp()
  for xsec in cross_sections:
    #filename = path.join(temp_dir, '{}'.format((int)(cross_section_count)+1)) # temp folder, all models
    filename = path.join(dir, '{}'.format((int)(cross_section_count) + 1)) # temp folder, all models

    print('- Cross-section @ {}'.format(xsec))
    model.plot_xsection(
      xsec = xsec, 
      ve = ve, 
      title = title + '\n' + eventSimpleText # Dennis: added here to contain information regarding the event order
    )    
    #plt.show()
    # DENNIS: added to save the figures instead of just showing them in a separate window
    plt.savefig(filename + '.pdf')
    plt.savefig(filename + '.svg')
    cross_section_count = cross_section_count + 1
  
  model.plot_table_simulation_parameters(title)
  filename_sim_parameters = path.join(dir, 'sim_parameters')
  plt.savefig(filename_sim_parameters + '.pdf')

  # Compact in a zip file all the PDF files in filename folder
  zipfile = path.join(dir, 'cross_sections_PDF.zip')
  zipFilesInDir(dir, zipfile, lambda fn: path.splitext(fn)[1] == '.pdf')
  copyfile(zipfile, 'cross_sections_PDF.zip')

  # Compact in a zip file all the SVG files in filename folder
  # Need to check here if the SVG file is consistent
  # Removed for now
  '''
  zipfile = path.join(temp_dir, 'cross_sections_SVG.zip')
  zipFilesInDir(temp_dir, zipfile, lambda fn: path.splitext(fn)[1] == '.svg')
  copyfile(zipfile, 'cross_sections_SVG.zip')
  '''

  rmtree(dir) # remove the temporary folder created to contain the cross section files before zipping

if export:
  print('Exporting 3D model')
  model.export_objs(ve = ve)

if render:
  print('Rendering 3D model')
  model.render()
