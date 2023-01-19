import tempfile
from os import path
from shutil import copyfile, rmtree
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from PIL import Image
from matplotlib.lines import Line2D
import open3d as o3d
import math
import struct
import os
from datetime import datetime

from definitions import *
from subroutines import *

class ChannelBelt3D():
    """
    Class for 3D models of channel belts. (Sylvester)
    """

    def __init__(self, topo, xmin, ymin, dx, dy, events, honestSpecificEvents, separator_type, width, elevation): # Added the events parameter by dennis
        """
        Initializes the ChannelBelt3D object.

        :param topo: set of topographic surfaces (3D numpy array) (Sylvester)
        :param xmin: TODO
        :param ymin: TODO
        :param dx: gridcell size (m) (Sylvester)
        :param dy: TODO
        :return: TODO
        """

        #self.raw_plot_xsection(0.1, topo)        
        self.strat = self.topostrat(topo)
        #self.raw_plot_xsection(0.1, self.strat)
        self.topo = topo
        #self.raw_plot_xsection(0.1, self.topo)

        self.xmin = xmin
        self.ymin = ymin        
        
        zmin, zmax = np.amin(self.strat[:,:,0]), np.amax(self.strat[:,:,-1])        

        dz = zmax - zmin
        
        self.zmin = zmin - dz * 0.1
        self.zmax = zmax + dz * 0.1           
    
        self.dx = dx
        self.dy = dy

        self.events = events # Added the events parameter by dennis
        self.honestSpecificEvents = honestSpecificEvents # Dennis
        
        self.separator_type = separator_type # Added the separator_type parameter by dennis

        self.width = width
        self.elevation = elevation

    def topostrat(self, topo, N = None):
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

    def raw_plot_xsection(self, xsec, matrix):
        """
        TODO

        :param xsec: TODO
        :param matrix: TODO        
        """

        sy, sx, sz = np.shape(matrix)
        xindex = int(xsec * sx)
        Xv = np.linspace(0, sy, sy)
        for i in range(0, sz, 4):            
            Y1 = matrix[:,xindex,i]
            plt.plot(Xv, Y1)

        plt.show()

    # Dennis: Added a new caption text containing the main information regarding the events
    # Added new parameters' information in a separate pdf file (table)
    def plot_table_simulation_parameters(self, title = ''):
        TABLE_FONT_SIZE = 2

        fig = plt.figure(figsize=(20,6)) # Dennis: changed from (20,5) to (20,6) to increase the title height
        
        # datetime object containing current date and time
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime('%d/%m/%Y %H:%M:%S')

        fig.suptitle(title + ' (' + dt_string + ')')

        # TODO: continue here the new subplot containing information of the setup from channels.json and setup.json
        axGeneral = fig.add_subplot(8,1,1)
        #axGeneral.set_title(title + ' (' + dt_string + ')')
        axGeneral.get_xaxis().set_visible(False)
        axGeneral.get_yaxis().set_visible(False)
        plt.box(on=None)
        column_texts = ('Channel width','Channel length','Channel elevation', 'Channel padding', 'Vertical exaggeration', 'Map scale', 'Cross sections', 'Title of the cross sections')
        row_texts = ('General Configs')
        cell_texts = []
        #line = [e.sep_type, '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', e.sep_thickness]
        #cell_texts.append(line)



        
        axEvents = fig.add_subplot(8,1,2)
        #axEvents.set_title(title + ' (' + dt_string + ')')
        axEvents.get_xaxis().set_visible(False)
        axEvents.get_yaxis().set_visible(False)
        plt.box(on=None)
        column_texts = ('Event type', 'Num. of layers', 'Iter. saving simul.', 'Time of each iter. (years)', \
                        'Agg./Inc. modul. (m/years)', 'Meandering modul. (m/years)', 'Channel depth (m)', 'Channel width (m)', \
                        'Deposition. height (m)', 'Layer deposition (%)', 'Gaussian deposition (%)', 'Layer aggradation (%)', \
                        'Gaussian aggradation (%)', 'Sep. thick. (m)')
        row_texts = []
        cell_texts = []
        eventCount = 1
        for e, h in zip(self.events, self.honestSpecificEvents):
            row_texts.append('Event ' + str(eventCount))

            # First element of the list
            if e.mode == 'SEPARATOR':
                line = [e.sep_type, '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', e.sep_thickness]
                #cell_texts.append((e.sep_type,e.number_layers,e.saved_ts,e.dt,e.kv,e.kl,'x'))
            elif e.mode == 'INCISION':
                line = [e.mode,e.number_layers,e.saved_ts,e.dt,e.kv,e.kl,h.ch_depth,h.ch_width,h.dep_height,h.dep_props,h.dep_sigmas,'-','-','-']
                #cell_texts.append((e.mode,e.number_layers,e.saved_ts,e.dt,e.kv,e.kl,'x'))
            elif e.mode == 'AGGRADATION':
                line = [e.mode,e.number_layers,e.saved_ts,e.dt,e.kv,e.kl,h.ch_depth,h.ch_width,h.dep_height,h.dep_props,h.dep_sigmas,h.aggr_props,h.aggr_sigmas, '-']

            # Remaining elements of the list
            #line += [e.number_layers,e.saved_ts,e.dt,e.kv,e.kl,h.ch_depth,h.ch_width,h.dep_height,h.dep_props,h.dep_sigmas,h.aggr_props,h.aggr_sigmas, e.sep_thickness]            
            cell_texts.append(line)
            eventCount += 1

        table = plt.table(cellText=cell_texts,
                    cellLoc='center',
                    rowLoc='center',
                    colLoc='center',
                    rowLabels=row_texts,                      
                    colLabels=column_texts,
                    loc='upper center')
        
        table.set_fontsize(TABLE_FONT_SIZE)
        table.auto_set_column_width(col=list(range(len(column_texts))))

        return fig

    # TODO: definir vetor de cores "separator_color"
    # separator: basal, inversion, condensed section
    def plot_xsection(self, xsec, ve = 1, substrat = True, title = '',
                    silt_color = [51/255, 51/255, 0], very_coarse_sand_color = [255/255, 153/255, 0], coarse_sand_color = [255/255, 204/255, 0],
                    sand_color = [255/255, 255/255, 0], fine_sand_color = [255/255, 255/255, 153/255], very_fine_sand_color = [255/255, 204/255, 153/255],
                    gravel_color = [255/255, 102/255, 0], separator_color = [[0/255,0/255,0/255],[255/255, 0/255, 0/255], [255/255, 0/255, 255/255], [0/255, 0/255, 255/255]]):
        """
        Method for plotting a cross section through a 3D model; also plots map of 
        basal erosional surface and map of final geomorphic surface. [Sylvester]

        :param xsec: location of cross section along the x-axis (in pixel/ voxel coordinates)
        :param ve: vertical exaggeration
        :param substrat: (bool) TODO
        :param title: (string) TODO
        :param silt_color: (list)
        :param sand_color: (list)
        :param gravel_color: (list)
        """
        
        BASAL_SURFACE = 1
        INVERSION_SURFACE = 2
        CONDENSED_SECTION = 3

        LINE_WIDTH = 4
        
        # DENNIS: modificado aqui
        #strat = self.topo
        strat = self.strat # aqui apenas strat final

        sy, sx, sz = np.shape(strat)
        
        if title != '': 
            title += '\n'
        
        xindex = int(xsec * sx)

        max_num_layers = 0
        for e in self.events:
            if e.number_layers > max_num_layers:
                max_num_layers = e.number_layers

        # gera as legendas para o Matplotlib
        # Added by Dennis: create the labels according to the number of layers in the events
        # TODO: criar legenda para cada uma das cores, dependendo dos separadores que foram usados
        if max_num_layers == 7:
            legend_elements = [
                Line2D([0], [0], color=silt_color, lw=LINE_WIDTH, label='Silt'),
                Line2D([0], [0], color=very_fine_sand_color, lw=LINE_WIDTH, label='Very Fine Sand'),
                Line2D([0], [0], color=fine_sand_color, lw=LINE_WIDTH, label='Fine Sand'),            
                Line2D([0], [0], color=sand_color, lw=LINE_WIDTH, label='Sand'),
                Line2D([0], [0], color=coarse_sand_color, lw=LINE_WIDTH, label='Coarse Sand'),
                Line2D([0], [0], color=very_coarse_sand_color, lw=LINE_WIDTH, label='Very Coarse Sand'),
                Line2D([0], [0], color=gravel_color, lw=LINE_WIDTH, label='Gravel'),
            ]            
        elif max_num_layers == 5:
            legend_elements = [
                Line2D([0], [0], color=silt_color, lw=LINE_WIDTH, label='Silt'),                
                Line2D([0], [0], color=fine_sand_color, lw=LINE_WIDTH, label='Fine Sand'),            
                Line2D([0], [0], color=sand_color, lw=LINE_WIDTH, label='Sand'),
                Line2D([0], [0], color=coarse_sand_color, lw=LINE_WIDTH, label='Coarse Sand'),                
                Line2D([0], [0], color=gravel_color, lw=LINE_WIDTH, label='Gravel'),                
            ]
        elif max_num_layers == 3:
            legend_elements = [
                Line2D([0], [0], color=silt_color, lw=LINE_WIDTH, label='Silt'),                        
                Line2D([0], [0], color=sand_color, lw=LINE_WIDTH, label='Sand'),                
                Line2D([0], [0], color=gravel_color, lw=LINE_WIDTH, label='Gravel'),                
            ]
        else:
            raise Exception('Invalid number of layers.')
        
        # Dennis: We only include the separator in the figure legend if at least one of the events is a separator
        hasSeparatorBasalSurface = False
        hasSeparatorInversionSurface = False
        hasSeparatorCondensedSection = False

        for e in self.events:
            if e.mode == 'SEPARATOR':
                if e.sep_type == 'BASAL_SURFACE':
                    hasSeparatorBasalSurface = True
                if e.sep_type == 'INVERSION':
                    hasSeparatorInversionSurface = True
                if e.sep_type == 'CONDENSED_SECTION':
                    hasSeparatorCondensedSection = True

        if (hasSeparatorCondensedSection):
            legend_elements.append(Line2D([0], [0], color=separator_color[CONDENSED_SECTION], lw=LINE_WIDTH, label='Condensed Section'))
        if (hasSeparatorInversionSurface):
            legend_elements.append(Line2D([0], [0], color=separator_color[INVERSION_SURFACE], lw=LINE_WIDTH, label='Inversion Surface'))
        if (hasSeparatorBasalSurface):
            legend_elements.append(Line2D([0], [0], color=separator_color[BASAL_SURFACE], lw=LINE_WIDTH, label='Basal Surface'))        

        # Matplotlib
        fig1 = plt.figure(figsize=(20,6)) # Dennis: changed from (20,5) to (20,6) to increase the title height # TODO
        #ax1 = fig1.add_subplot(4,1,(1,2))        
        ax1 = fig1.add_subplot(1,1,1)
        ax1.set_title('{}Cross section at ({:.3f}) - {:.3f} km'.format(title, xsec, xindex * self.dx + self.xmin))        

        # For now info is only displayed for the first event
        '''
        tex = '|Event 1configurations|\n' + 'nit: ' + str(self.events[0].nit) + '\n' \
                + 'saved_ts: ' + str(self.events[0].saved_ts) + '\n' \
                + 'dt: ' + str(self.events[0].dt) + '\n' \
                + 'mode: ' + str(self.events[0].mode) + '\n' \
                + 'kv: ' + str(self.events[0].kv) + '\n' \
                + 'kl: ' + str(self.events[0].kl) + '\n' \
                + 'number_layers: ' + str(self.events[0].number_layers) + '\n'
        ax1.text(-150, 225, tex, fontsize=10, va='bottom')
        '''

        Xv = np.linspace(self.ymin, self.ymin + sy * self.dy, sy)
        X1 = np.concatenate((Xv, Xv[::-1])) # faz array inverso
        
        if substrat:
            substract_color = [192/255, 192/255, 192/255]
            Yb = np.ones(sy) * self.zmin
            ax1.fill(X1, np.concatenate((Yb, strat[::-1,xindex,0])), facecolor=substract_color)
            legend_elements.append(Line2D([0], [0], color=substract_color, lw=LINE_WIDTH, label='Substract'))
        
        # atualizar: Y1...Y7
        # sz: numero total de camadas
        for i in range(0, sz, NUMBER_OF_LAYERS_PER_EVENT):
            Y1 = np.concatenate((strat[:,xindex,i],   strat[::-1,xindex,i+1])) 
            Y2 = np.concatenate((strat[:,xindex,i+1], strat[::-1,xindex,i+2]))
            Y3 = np.concatenate((strat[:,xindex,i+2], strat[::-1,xindex,i+3]))
            Y4 = np.concatenate((strat[:,xindex,i+3], strat[::-1,xindex,i+4]))
            Y5 = np.concatenate((strat[:,xindex,i+4], strat[::-1,xindex,i+5]))
            Y6 = np.concatenate((strat[:,xindex,i+5], strat[::-1,xindex,i+6]))
            Y7 = np.concatenate((strat[:,xindex,i+6], strat[::-1,xindex,i+7]))
            Y8 = np.concatenate((strat[:,xindex,i+7], strat[::-1,xindex,i+8]))
            
            ax1.fill(X1, Y1, facecolor=gravel_color)
            ax1.fill(X1, Y2, facecolor=very_coarse_sand_color) 
            ax1.fill(X1, Y3, facecolor=coarse_sand_color)
            ax1.fill(X1, Y4, facecolor=sand_color)
            ax1.fill(X1, Y5, facecolor=fine_sand_color)
            ax1.fill(X1, Y6, facecolor=very_fine_sand_color)
            ax1.fill(X1, Y7, facecolor=silt_color)
            # TODO: testar o tipo do separador para criar a cor específica (vermelho, roxo, azul)
            # recuperar ele do JSON (buscar o tipo e indexar)
            # QUAL RELAÇÃO DOS EVENTOS COM O STRAT e com o FOR acima?
            ax1.fill(X1, Y8, facecolor=separator_color[self.separator_type[i+8]])    

        # HERE    
        
        if ve != 1:
            ax1.set_aspect(ve, adjustable='datalim')
        
        # Still need to debug this... aparently ymin has no relation with the width from the interface
        #ax1.set_xlim(self.ymin, self.ymin + sy * self.dy) # TODO Dennis: check here the x limits
        print('WIDTH: ', self.width)
        ax1.set_xlim(-1.2*self.width, 1.2*self.width)        
        ax1.set_ylim(-0.1*self.elevation, 1.1*self.elevation)
        
        ax1.legend(handles=legend_elements, loc='upper right')

        #plt.savefig('teste.svg')
        
        ax1.set_xlabel('Width (m)')
        ax1.set_ylabel('Elevation (m)')

        plt.savefig('teste.svg')

        return fig1

    def plot(self, ve = 1, curvature = False, save = False):
        """
        TODO

        :param ve: TODO 
        :param curvature: (bool) TODO        
        :param save: (bool) TODO 
        """

        sy, sx, sz = np.shape(self.strat)
        x = np.linspace(self.xmin, self.xmin + sx * self.dx, sx)
        y = np.linspace(self.ymin, self.ymin + sy * self.dy, sy)

        xx, yy = np.meshgrid(x, y)
        zz = self.strat[:,:,-1 - 4] * ve

        grid = pv.StructuredGrid(xx, yy, zz)

        if curvature:
            grid.plot_curvature()
        else:
            grid.plot()

        if save:
            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(grid, color = 'brown')

            plotter.show(screenshot='airplane.png')

    def render(self, ve = 3):
        """
        TODO

        :param ve: TODO 
        """

        sy, sx, sz = np.shape(self.strat)
        x = np.linspace(self.xmin, self.xmin + sx * self.dx, sx)
        y = np.linspace(self.ymin, self.ymin + sy * self.dy, sy)
        
        xx, yy = np.meshgrid(x, y)

        zz = self.topo[:,:,0] * ve

        grid = pv.StructuredGrid(xx, yy, zz)

        plotter = pv.Plotter()
        plotter.add_mesh(grid)

        plotter.show()  

    # Generates a new PLY file containing the x,y,z coordinates in float instead of double
    def reducePlySize(self, inFileName, outFileName):
        """
        TODO

        :param inFileName: TODO 
        :param outFileName: TODO 
        """

        try:        
            # First part: read (and write) the ply header as text file
            with open(inFileName, "rt", encoding="Latin-1") as inFile:
                with open(outFileName, "wt") as outFile:
                    line = ''
                    while line != 'end_header\n':
                        line = inFile.readline()
                        if line == 'property double x\n':
                            outFile.write('property float x\n')
                        elif line == 'property double y\n':
                            outFile.write('property float y\n')
                        elif line == 'property double z\n':
                            outFile.write('property float z\n')
                            
                        else:                        
                            if line[0:15] == 'element vertex ':
                                nVertices = int(line[15:-1]) #gets que number of vertices
                            #if line[0:13] == 'element face ':
                            #    nFaces = int(line[13:-1]) #gets que number of faces                            

                            outFile.write(line)
                    
                    currentPos = inFile.tell()
            
            # Second part: read (and write) the ply vertices colors and faces as text file
            with open(inFileName, "rb") as inFile:                        
                with open(outFileName, "ab") as outFile:
                    inFile.seek(currentPos)                

                    # Read and trasnform all the vertex values to float, maintaining their colors
                    # Part 1: convert x, y, z from double to float... RGB keep the same as ubyte
                    for i in range(nVertices):
                        x = np.float32(struct.unpack('d',inFile.read(8))) # Read the x double value
                        y = np.float32(struct.unpack('d',inFile.read(8)))
                        z = np.float32(struct.unpack('d',inFile.read(8)))                    
                        x = struct.pack('f',x[0])                    
                        y = struct.pack('f',y[0])
                        z = struct.pack('f',z[0])

                        red = np.ubyte(struct.unpack('B',inFile.read(1)))                    
                        green = np.ubyte(struct.unpack('B',inFile.read(1)))
                        blue = np.ubyte(struct.unpack('B',inFile.read(1)))                      
                        red = struct.pack('B',red[0])                                        
                        green = struct.pack('B',green[0])
                        blue = struct.pack('B',blue[0])

                        outFile.write(x)
                        outFile.write(y)
                        outFile.write(z)
                        outFile.write(red)
                        outFile.write(green)
                        outFile.write(blue)    

                    # Read and trasnform all the vertex values to float, maintaining their colors
                    buffer = inFile.read(1)
                    while buffer != b"":                                     
                        outFile.write(buffer)
                        buffer = inFile.read(1)                

        except IOError:
            print("Error. Could not read files ", inFileName, " and ", outFileName)    

    def generateTriangleMesh(self, vertices, faces, colors, fileNameOut='out.ply', coloredMesh=True):
        """
        Generate and export a 3D model in PLY file format. The file has reduced size with float32 data instead of double.

        :param vertices: TODO 
        :param faces: TODO 
        :param faces: TODO 
        :param colors: TODO 
        :param fileNameOut: (string) TODO 
        :param coloredMesh: (bool) TODO 
        """

        mesh = o3d.geometry.TriangleMesh()
        
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        if coloredMesh:
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        
        o3d.io.write_triangle_mesh(fileNameOut, mesh, write_vertex_colors=coloredMesh, compressed=True)
        
        # New code for saving/overwriting a new PLY file with float32 instead of double (64)
        self.reducePlySize(fileNameOut, fileNameOut[:-4]+'_'+'.ply')
        if os.path.isfile(fileNameOut):
            os.remove(fileNameOut)
        if os.path.isfile(fileNameOut[:-4]+'_'+'.ply'):
            os.rename(fileNameOut[:-4]+'_'+'.ply', fileNameOut)

    def export_top_layer(self, structure, structure_colors, event_top_layer, number_layers_per_event, grid, top, filename, plant_view,
                        reduction = None, colored_mesh = True):
        """
        TODO

        :param structure: TODO
        :param structure_colors: TODO
        :param event_top_layer: TODO
        :param number_layers_per_event: TODO
        :param grid: TODO
        :param top: TODO
        :param filename: (string) TODO
        :param plant_view: TODO
        :param reduction: TODO
        :param colored_mesh: (bool) TODO
        """

        FLOAT_TO_INT_FACTOR = 1
        SUBSTRACT_COLOR_INDEX = 8

        sy, sx, sz = np.shape(structure) # sz contains the number of layers from strat
        # Array containing the possible colors from strat
        #colorCpInt = np.zeros([sy,sx])
        
        # Initialize the colors of the mesh with the substract color
        colorCpInt = np.zeros((sy,sx, 3))
        for xIndex in range(0, sx):
            for yIndex in range(0, sy):
                colorCpInt[yIndex,xIndex] = structure_colors[SUBSTRACT_COLOR_INDEX]
        
        # Stores information of the points and their colors of the first layer (top)
        for xIndex in range(0, sx):
            for yIndex in range(0, sy):
                #for zIndex in range(sz-1, 1, -1): # 0 is the bottom layer
                for zIndex in range(event_top_layer, 0, -1): # 0 is the bottom layer
                    if structure[yIndex,xIndex,zIndex] == 1:
                        # Computes in colorIndex the correct layer color according to the following formula
                        colorIndex = abs((zIndex % number_layers_per_event) - (number_layers_per_event-1))
                        colorCpInt[yIndex,xIndex] = structure_colors[colorIndex]
                        break        

        # Stores the data structure from points, colors and plant view
        surfacePointList = []
        colorPointList = []
        cols, rows, channel = np.shape(colorCpInt)
        topCp = top.copy()
        topCp = np.reshape(topCp, (rows,cols,channel))                                    
        for col in range(cols):
            for row in range(rows):    
                surfacePointList.append([topCp[row][col][0], topCp[row][col][1], topCp[row][col][2]])
                colorPointList.append([colorCpInt[col][row][0], colorCpInt[col][row][1], colorCpInt[col][row][2]])
                plant_view[col,row] = np.uint8(colorCpInt[col,row] * 255)
        
        im = Image.fromarray(plant_view)
        im.save(filename + ".png")        
        
        # Produce a PLY file with it's box already triangulated
        bottom = grid.points.copy()
        bottom[:,-1] = self.zmin
        grid.points = np.vstack((top, bottom))
        grid.dimensions = [*grid.dimensions[0:2], 2]
        plotter = pv.Plotter()
        plotter.add_mesh(grid)#, scalars="colors", rgb=True) # add to scene

        # Depending on the PC's setup, the export_obj from plotter may or may not include the suffix '.obj'
        if filename[-4:] != '.obj':
            filename = filename + '.obj'

        plotter.export_obj(filename)
        data = trimesh.load(filename, force='mesh')

        # Use only if the above does not work
        '''
        print("FILENAME HERE 1:", filename) 
        if filename[-4:] == '.obj':
            plotter.export_obj(filename)
        else:
            plotter.export_obj(filename + '.obj')   

        print("FILENAME HERE 2:", filename)     
            
        if filename[-4:] == '.obj':  
            data = trimesh.load(filename, force='mesh')
        else:
            data = trimesh.load(filename + '.obj', force='mesh')
        '''

        vertices = data.vertices
        faces = np.ones((data.faces.shape[0], data.faces.shape[1]+1)) * data.faces.shape[1]
        faces[:,1:] = data.faces
        faces = np.hstack(faces).astype(int)
        mesh = pv.PolyData(vertices, faces)
        if reduction is not None:
            mesh.decimate(reduction)

        # Remove .obj extension to be consistent
        if filename[-4:] == '.obj':            
            filename = filename[:-4]
        mesh.save(filename + '.ply')

        # Map the surface points (and their respective surface color points) to the block points, which have doubled the amount of points
        # with a projection on XY axis. The idea is to associate each color from the surface with the new ordered block points. Due to the
        # performance, instead of comparing both lists we converted the surface points (each point as [x,y,z]) to a dictionary composed by
        # their math.ceil values. This may lead to some precision errors.
        surface_points = np.asarray(surfacePointList)
        surface_colors = np.asarray(colorPointList)

        mesh = o3d.io.read_triangle_mesh(filename + ".ply")
        block_vertices = np.asarray(mesh.vertices)
        block_triangles = np.asarray(mesh.triangles)

        # Code to export the obj and ply surface mesh as a block with ground and walls with color
        if colored_mesh:
            # Create hash in surfaceDict to compare more efficiently "two lists" of elements
            block_colors = np.zeros([len(block_vertices),3])
            surfaceDict = {}
            for event_top_layer in range(len(surface_points)):
                x_str = str(math.ceil((surface_points[event_top_layer][0]*FLOAT_TO_INT_FACTOR)//1))
                y_str = str(math.ceil((surface_points[event_top_layer][1]*FLOAT_TO_INT_FACTOR)//1))
                z_str = str(math.ceil((surface_points[event_top_layer][2]*FLOAT_TO_INT_FACTOR)//1))
                hash_aux = x_str + ',' + y_str + ',' + z_str
                surfaceDict[hash_aux] = surface_colors[event_top_layer]

            cont = 0
            for v in block_vertices:                    
                x_str = str(math.ceil((v[0]*FLOAT_TO_INT_FACTOR)//1))
                y_str = str(math.ceil((v[1]*FLOAT_TO_INT_FACTOR)//1))
                z_str = str(math.ceil((v[2]*FLOAT_TO_INT_FACTOR)//1))
                hash_aux = x_str + ',' + y_str + ',' + z_str
                if hash_aux in surfaceDict:
                    block_colors[cont] = surfaceDict[hash_aux]
                cont = cont + 1                
            
            self.generateTriangleMesh(block_vertices, block_triangles, block_colors, filename + '.ply', coloredMesh=True)

        # Code to export the obj and ply surface mesh as a block with ground and walls (Beuren's original mesh)
        else:              
            block_colors = np.zeros([len(block_vertices),3])
            self.generateTriangleMesh(block_vertices, block_triangles, block_colors, filename + '.ply', coloredMesh=False)
  
    def export_objs(self, top_event_layers_zipname = 'event_layers.zip', savePlantView = True, plant_view_zipname = 'plant_view.zip', reduction = None, ve = 3):
        """
        Function to export the 3D meshes for each layer. Meshes are compressed as PLY files and compacted into a ZIP.
        Outputs a ZIP file containing the models for each layer (names as model1.ply, model2.ply, etc).

        :param top_event_layers_zipname: TODO 
        :param reduction: TODO 
        :param ve: vertical exaggeration TODO        
        """

        # Constants        
        LAYER_THICKNESS_THRESHOLD = 1e-1#0.9 #1e-2   

        # atualizar
        
        GRAVEL_COLOR = [255/255, 102/255, 0]
        VERY_COARSE_SAND_COLOR = [255/255, 153/255, 0]
        COARSE_SAND_COLOR = [255/255, 204/255, 0]
        SAND_COLOR = [255/255, 255/255, 0]
        FINE_SAND_COLOR = [255/255, 255/255, 153/255]
        VERY_FINE_SAND_COLOR = [255/255, 204/255, 153/255]
        SILT_COLOR = [51/255, 51/255, 0]
        SEPARATOR_COLOR = [0/255, 0/255, 255/255]
        SUBSTRACT_COLOR = [192/255, 192/255, 192/255]
        
        # Set the strat material colors to an array
        strat_colors = np.array([SEPARATOR_COLOR, SILT_COLOR, VERY_FINE_SAND_COLOR, FINE_SAND_COLOR, SAND_COLOR, COARSE_SAND_COLOR,
                                VERY_COARSE_SAND_COLOR, GRAVEL_COLOR, SUBSTRACT_COLOR])
        
        # dir contains the path of the temp directory, used to store the intermediate meshes
        dir = tempfile.mkdtemp()
        
        # strat: 3D numpy array of statigraphic surfaces (previously was named zz)
        #strat = topostrat(self.topo)
        strat = self.strat

        sy, sx, sz = np.shape(strat) # sz contains the number of layers from strat

        # Produce a list of 'sx' and 'sy' interpolated values from a given range
        x = np.linspace(self.xmin, self.xmin + sx * self.dx, sx)
        y = np.linspace(self.ymin, self.ymin + sy * self.dy, sy)

        # xx and yy form a grid (2D plane) according to x and y values
        xx, yy = np.meshgrid(x, y)
        
        # For each adjacent strat layer compares their thickness, setting to 0 if they are very close and 1 otherwise
        stratCp = strat.copy()
        for xIndex in range(0, sx):
            for yIndex in range(0, sy):
                for zIndex in range(sz-1, 1, -1):
                    if (abs(stratCp[yIndex,xIndex,zIndex-1] - stratCp[yIndex,xIndex,zIndex-2]) < LAYER_THICKNESS_THRESHOLD):
                        stratCp[yIndex,xIndex,zIndex-1] = 0
                    else:
                        stratCp[yIndex,xIndex,zIndex-1] = 1

                    #break
                stratCp[yIndex,xIndex,sz-1] = 0
        
        # Initializes the plant view of the channel for each of the layers.
        plant_view = np.uint8(np.zeros((sy,sx,3)))
        mesh_iterator = 0
        # Main loop to generate a mesh for each layer. The meshes are available in a zip file names i.ply.
        # For now, we have 4 layers in the following order: silt, sand, gravel and substract
        # Layer 0 corresponds to the initialized layer in the constructor of the channel belt        
        for event_top_layer in range(0, sz, NUMBER_OF_LAYERS_PER_EVENT):
            update_progress(event_top_layer/sz)            
            #filename = 'model{}'.format(int(i/3) + 1) # local folder
            filename = path.join(dir, '{}'.format((int)(event_top_layer/NUMBER_OF_LAYERS_PER_EVENT) + 1)) # temp folder, all models            

            #print('Entering topostrat from export_objs')
            strat = topostrat(self.topo, event_top_layer)
            
            # Produces a grid for the current z layer containing the points in grid.points
            grid = pv.StructuredGrid(xx, yy, strat[:,:,event_top_layer] * ve)

            # top contains all the surface points for each layer
            top = grid.points.copy()

            # Export one mesh
            self.export_top_layer(stratCp, strat_colors, event_top_layer, NUMBER_OF_LAYERS_PER_EVENT, grid, top, filename, plant_view, \
                        reduction, colored_mesh=True)

            mesh_iterator = mesh_iterator + 1

        # Compact in a zip file all the ply files in filename folder
        zipfile = path.join(dir, top_event_layers_zipname)
        zipFilesInDir(dir, zipfile, lambda fn: path.splitext(fn)[1] == '.ply')
        copyfile(zipfile, top_event_layers_zipname)

        # TODO: save plant view as zip file
        if (savePlantView):
            zipfile = path.join(dir, plant_view_zipname)
            zipFilesInDir(dir, zipfile, lambda fn: path.splitext(fn)[1] == '.png')
            copyfile(zipfile, plant_view_zipname)

        rmtree(dir) # remove the temporary folder created to contain the mesh files before zipping   

    def export(self, ve = 3):
        """
        TODO

        :param ve: TODO   
        """
        #np.savetxt('shape.txt',[sy, sx, sz],fmt='%.4f') # DEBUG
        #zz = topostrat_evolution(self.topo)
        print("entering topostrat from export")
        zz = topostrat(self.topo)
        np.save("terrain.npy", zz)