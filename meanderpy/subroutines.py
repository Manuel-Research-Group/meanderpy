from zipfile import ZipFile
from os import path, walk
import matplotlib.pyplot as plt
import sys
import numpy as np

from definitions import *

def zip_files_in_dir(dirName, zipFileName, filter):
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

def update_progress(progress):
    """
    Displays or updates a console progress bar.
    Adapted from https://stackoverflow.com/questions/3160699/python-progress-bar

    :param progress: (float) value 0 represents halt and 1 represents 100%.    
    """

    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{}] {:.2f}% {}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

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

def topostrat(topo, N = None):
    """
    Function for converting a stack of geomorphic surfaces into stratigraphic surfaces.

    :param topo: 3D numpy array of geomorphic surfaces
    :param N: TODO
    :return: strat - 3D numpy array of stratigraphic surfaces
    """    

    r,c,ts = np.shape(topo)    

    T = N if N is not None else ts # added to solve the incision problem    

    strat = np.copy(topo)     

    for i in (range(0,T)): # the layer 0 is the bottom one
        strat[:,:,i] = np.amin(topo[:,:,i:], axis=2)        
    
    return strat # matriz com todos os pontos (armazenado valor do z mínimo)

def plot2D(x, y, title, ylabel, fileName, save=True):
    """
    Plots in a Matplotlib graph the x and y array values.
    
    :param x: vector containing the x elements.
    :param y: vector containing the y elements.
    :param title: title of the plot.
    :param ylabel: label of the y axis.
    """
    
    plt.plot(x, y)        
    plt.title(title)
    plt.xlabel('Length (m)')
    plt.ylabel(ylabel)
    plt.ylim(-50, 500)
    if (save):
        plt.savefig(fileName)
        plt.clf()
    else:
        plt.show()

def plot3D(Z, grid_size = 1):
    """
    TODO

    :param Z: TODO
    :param grid_size: TODO
    :return: TODO
    """

    X, Y = np.meshgrid(np.linspace(0, Z.shape[1] * grid_size, Z.shape[1]), np.linspace(0, Z.shape[0] * grid_size, Z.shape[0]))
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z)

    max_range = np.array([X.max()-X.min(), Y.max()-Y.min()]).max() / 2.0

    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range/4, mid_y + max_range/4)

    return fig

def erosional_surface(cld_map, z_map, hw_map, cd_map):
    """
    TODO

    :param cld_map: centerline distance map
    :param z_map: represents the current z level of the basin
    :param hw_map: map that relates each pixel to its closer channel width that is called the half-width
    :param cd_map: the channel depth at each location which multiplies the terms that creates the parabolic formation
    :return: TODO
    """

    return cd_map * ((cld_map / hw_map) ** 2 - 1) + z_map

def gaussian_surface(sigma_map, cld_map, hw_map):
    """
    Calculates the gaussian 

    :param sigma_map: TODO
    :param cld_map: centerline distance map
    :param hw_map: map that relates each pixel to its closer channel width that is called the half-width
    :return: TODO
    """

    return np.exp(- 1 / 2 * ((cld_map / hw_map) / sigma_map) ** 2)

# GRAVEYARD
'''
def topostrat_evolution(topo):
    """
    Function for converting a stack of geomorphic surfaces into stratigraphic surfaces.

    :param topo: 3D numpy array of geomorphic surfaces
    :return: strat: 3D numpy array of stratigraphic surfaces
    """
    N = 4
    r,c,ts = np.shape(topo)
    strat = np.zeros((r,c,int(ts/N)))
    for i in (range(0,ts, N)):
        strat[:,:,int((i+1)/N)] = np.amin(topo[:,:,i:i+N], axis=2)
    return strat
'''