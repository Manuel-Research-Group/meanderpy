import bisect
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import seaborn as sns
import json

from subroutines import *
from channel_mapper import *
from channel_belt_3d import *

class ChannelBelt:
    """
    Class for ChannelBelt objects.
    """

    def __init__(self, channel, basin, honestSpecificEvents):
        """
        Initializes the ChannelBelt object. Times in years.

        :param channel: list of Channel objects [Sylvester]
        :param basin: TODO        
        """

        self.channels = [channel.copy()]        
        self.basins = [basin.copy()]
        self.times = [0.0]
        self.events = []
        self.honestSpecificEvents = honestSpecificEvents

    # 2 progress bars: meandering + modeling
    # essa parte é simulação 2D
    def simulate(self, event, eventOrder): # parte 2D
        """
        TODO

        :param event: event list. 
        :param eventOrder: order of the event in the event list. Use to allow saving all intermediate channel and basin profiles.            
        """        
        
        last_time = self.times[-1]
        event.start_time = last_time + event.dt    

        # Since an event has been appended for the first event, it should not be appended again later inside the loop below.
        # This variable makes sure that it doesn't happen.
        firstEvent = False    
        
        if len(self.events) == 0:
            channel = self.channels[0]
            basin = self.basins[0]
            self.events.append(event) #create a "base event" which is a copy of the first one (need to be confirmed if necessary)
            channel.refit(basin, event.ch_width, event.ch_depth)
            _, _, _, ds, _ = channel.derivatives()
            self.ds = np.mean(ds)
            event.start_time = 0
            firstEvent = True

        channel = self.channels[-1].copy()
        basin = self.basins[-1].copy()
        last_time = self.times[-1]
        
        for itn in range(1, event.nit+1):            
            update_progress(itn/event.nit)    

            #plot2D(channel.x, channel.y, 'Channel Preview', 'Width (m)') # need to update the call
            #plot2D(basin.x, basin.z, 'Basin Preview', 'Elevation (m)') # need to update the call

            channel.migrate(event.Cf, event.kl / YEAR, event.dt * YEAR)            
            channel.cut_cutoffs(event.cr_dist, self.ds)            
            channel.cut_cutoffs_R(event.cr_wind, self.ds)            
            channel.resample(self.ds) # deixar curva smooth e deixa dl (comprimento de largura da curva) constante              
            channel.refit(basin, event.ch_width, event.ch_depth) # avaliação da elevação (z) do canal a cada ponto com base na bacia              
            
            if event.mode == 'INCISION':
                #print('INCISION!')
                basin.incise(event.dens, event.kv / YEAR, event.dt * YEAR)
            if event.mode == 'AGGRADATION':
                #print('AGGRADATION!')
                basin.aggradate(event.dens, event.kv / YEAR, event.dt * YEAR, event.aggr_factor)
            # TODO Dennis: SEPARATE must be included as a function from basin            
            #if event.mode == 'SEPARATOR':
                #print('SIMULATE/SEPARATOR!')    

            # número de canais = time stamp
            # Save if it is not the first event or if it is the first event we avoid saving the first saved_ts layers because
            # it has already been saved when "len(self.events) == 0" (see the respective test in the beginning of this procedure)
            if ((itn % event.saved_ts == 0) and (event.mode == 'INCISION' or event.mode == 'AGGRADATION') and \
                ((not firstEvent) or (firstEvent and itn > event.saved_ts))):
                self.times.append(last_time + (itn+1) * event.dt)
                self.channels.append(channel.copy())
                self.basins.append(basin.copy())
                self.events.append(event)
                # TODO: call generateCrossSections() from here if possible
                
                # Used to create the gif files containing the basin animation (OFF FOR NOW - Dennis)
                #plot2D(basin.x, basin.z, 'Basin Preview', 'Elevation (m)', 'basin_' + str(eventOrder) +'-' + str(itn) + '.png', save=True) # vista lateral
                #plot2D(channel.x, channel.y, 'Channel Preview', 'Elevation (m)', 'channel_' + str(eventOrder) + '-' + str(itn) + '.png', save=True) # vista superior
            
        # DENNIS: saves a single layer of separator
        if ((event.mode == 'SEPARATOR') and (not firstEvent)):
            self.times.append(last_time + event.dt)
            self.channels.append(channel.copy())
            self.basins.append(basin.copy())
            self.events.append(event)   

        # dgb: Save the final mesh
        '''
        print("\nFinal mesh: ", itn, " <space>.")
        self.times.append(last_time + (itn+1) * event.dt)
        self.channels.append(channel.copy())
        self.basins.append(basin.copy())
        self.events.append(event)
        '''        
        
        #print("\n#times: ", len(self.times), " <space>.")

    def plot_basin(self, evolution = True):
        """
        TODO

        :param evolution: TODO
        :return: TODO
        """

        fig, axis = plt.subplots(1, 1)
        if not evolution:
            self.basins[-1].plot(axis)
        else:
            legends = []
            uniques = set()
            self.basins[0].plot(axis)
            legends.append('initial')
            for evt in self.events:
                i = self.times.index(evt.start_time)
                if not i in uniques:
                    uniques.add(i)
                    self.basins[i + int(evt.nit / evt.saved_ts) - 1].plot(axis)
                    legends.append('event-{}'.format(len(uniques)))
            axis.legend(legends)
        axis.set_xlabel('X (m)')
        axis.set_ylabel('Elevation (m)')
        return fig

    def plot(self, start_time=0, end_time = 0, points = False):
        """
        TODO

        :param start_time: TODO
        :param end_time: TODO
        :param points: TODO
        :return: TODO
        """

        start_index = 0
        if start_time > 0:
            start_index = bisect.bisect_left(self.times, start_time)

        end_index = len(self.times)
        if end_time > 0:
            end_index = bisect.bisect_right(self.times, end_time)
            
        fig, axis = plt.subplots(1, 1)
        axis.set_aspect('equal', 'datalim')

        for i in range(start_index, end_index):
            color = sns.xkcd_rgb["ocean blue"] if i == end_index - 1 else sns.xkcd_rgb["sand yellow"]
            self.channels[i].plot(axis, color, points)

        return fig

    # Variable introduced to show the cross sections
    # Generate and organize the new matplotlib figures into a zip
    # TODO: check the parameters below
    def generate_cross_sections(self, cross_sections, model, eventName, generateTableSimulator, ve):
        """
        Generate and organize the new matplotlib figures into a zip

        :param x: TODO
        """
        dir = tempfile.mkdtemp()
        cross_section_count = 0
        for xsec in cross_sections:
            #filename = path.join(temp_dir, '{}'.format((int)(cross_section_count)+1)) # temp folder, all models
            filename = path.join(dir, '{}'.format((int)(cross_section_count) + 1)) # temp folder, all models

            print('- Cross-section @ {}'.format(xsec))            
            model.plot_xsection(
            xsec, 
            ve     # added ve here to correct further classes: when generating 3d model and the 2d cross-sections
            )    
            #plt.show()
            # DENNIS: added to save the figures instead of just showing them in a separate window            
            plt.savefig(filename + '.pdf')
            plt.savefig(filename + '.svg')
            plt.savefig(filename + '.png')
            cross_section_count = cross_section_count + 1

        if generateTableSimulator:
            # Compact in a zip file all the simulation parameters in filename folder
            model.plot_table_simulation_parameters('title')
            filename_sim_parameters = path.join(dir, 'sim_parameters')
            plt.savefig(filename_sim_parameters + '.pdf')
        
        # Remove any invalid character for the file name in windows operating systems
        invalid = '<>:"/\|?* '
        for c in invalid:
            eventName = eventName.replace(c, '')

        # Compact in a zip file all the PDF cross section files in filename folder
        zipfileName = path.join(dir, 'cross_sections_PDF-' + eventName + '.zip')  
        zip_files_in_dir(dir, zipfileName, lambda fn: path.splitext(fn)[1] == '.pdf')
        copyfile(zipfileName, 'cross_sections_PDF-' + eventName + '.zip')

        # Compact in a zip file all the JPG cross section files in filename folder
        zipfileName = path.join(dir, 'cross_sections_JPG-' + eventName + '.zip')  
        zip_files_in_dir(dir, zipfileName, lambda fn: path.splitext(fn)[1] == '.png')
        copyfile(zipfileName, 'cross_sections_PNG-' + eventName + '.zip')

        # Compact in a zip file all the SVG files in filename folder  
        # Removed for now since SVG file is not vectorized
        zipfileName = path.join(dir, 'cross_sections_SVG' + eventName + '.zip')
        zipFilesInDir(dir, zipfileName, lambda fn: path.splitext(fn)[1] == '.svg')
        copyfile(zipfileName, 'cross_sections_SVG-' + eventName + '.zip')

        rmtree(dir)

    def build_3d_model(self, dx, margin = 500, width=500, elevation=500, ve=1): # recebe lista de bacias e lista de canais
        """
        TODO

        :param dx: grid generated from dxdy variable (coming from JSON "Map Scale"). See initialization of grid variable in runner.py
        :param margin: margin informed as "Channel Padding" in the "Channel Configuration" interface
        :return: TODO
        """

        xmax, xmin, ymax, ymin = [], [], [], []
        for channel in self.channels: # um canal para cada snapshot. Cada passo gera 4 malhas
            xmax.append(max(channel.x))
            xmin.append(min(channel.x))
            ymax.append(max(channel.y))
            ymin.append(min(channel.y))

        xmax = max(xmax)
        xmin = min(xmin)
        ymax = max(ymax)
        ymin = min(ymin)        

        # cria mapas
        mapper = ChannelMapper(xmin + margin, xmax - margin, ymin - margin, ymax + margin, dx, dx)

        channel = self.channels[0] #canais 2D vista superior
        basin = self.basins[0] # bacia 2D vista lateral
        ch_map, cld_map, md_map, cz_map, bz_map, sl_map, hw_map = mapper.create_maps(channel, basin)

        # surface: resultado atual do processo de corte e deposição (cut and fill)
        surface = bz_map # bz_map: altura do centro do canal explodido lateralmente

        N = len(self.channels) # N is the number of meshes to be saved according to the nit and saved_ts

        L = NUMBER_OF_LAYERS_PER_EVENT # atualizar

        topo = np.zeros((mapper.rheight, mapper.rwidth, N*L))    

        # Dennis
        # separator_type contains the same number of layers as topo and stores the separator type (when one exists) at the proper layer.
        # It was created as a separate array because topo is itself an array and not a structure to which we could add an aditional attribute.
        #
        separator_type = np.zeros((N*L), dtype=int)        

        for i in range(0, N): # Dennis - Obs: if used range(1,N) we avoid drawing the layers of the first event twice but it draws the substract wrongly
            update_progress(i/N)

            event = self.events[i] 

            firstEventIsSeparator = False
            if i == 0 and event.mode == 'SEPARATOR':
                firstEventIsSeparator = True            

            # Last iteration 
            # aggr_map: qual parte do terreno está sofrendo aggradation
            # surface: parte mais superior computada
            aggr_map = bz_map - surface 
            aggr_map[aggr_map < 0] = 0            

            # channel, centerline distance, channel z, basin z, slope, half width
            # ch_map:
            ch_map, cld_map, md_map, cz_map, bz_map, sl_map, hw_map = mapper.create_maps(self.channels[i], self.basins[i])            

            # channel depth
            dh_map = event.dep_height(sl_map)
            cd_map = event.ch_depth(sl_map)

            if (event.mode != 'SEPARATOR' or firstEventIsSeparator):            
                channel_surface = erosional_surface(cld_map, cz_map, hw_map, cd_map) # parte do cut
            else:
                channel_surface = surface

            # NEED TO ADJUST THIS TO HANDLE MORE THAN THREE LAYERS (GRAVEL, SAND, SILT)
                        
            # gr_p: gravel proportions. gr_s: gravel sigma.
            # sa_p: sand proportions. sa_s: sand sigma.
            # si_p: silt proportions. si_s: silt sigma.
            # t_p: soma das proporções total (não precisa dar 100%) = gr_p + sa_p + si_p

            # We then deposit (aggradate) a certain number of materials. For instance, when using gravel, sand and silt, the number of materials == 3.
            # When using gravel, gross sand, medium sand, fine sand, and silt, then number of materials == 5.
            # When using gravel, very gross sand, gross sand, medium sand, fine sand, very fine sand, and silt, then number of materials == 7.
            
            # atualizar: retornar 8 variáveis em vez das 3 (algumas terão zeros, para garantir que sejam 7 camadas + separator)
            # gravel, sand and silt variables:        
            # gr_p: gravel proportions. gr_s: gravel sigma.
            # vcsa_p: very coarse sand proportions. vcsa_s: very coarse sand sigma.
            # csa_p: coarse sand proportions. csa_s: coarse sand sigma.
            # sa_p: sand proportions. sa_s: sand sigma.
            # fsa_p: fine sand proportions. fsa_s: fine sand sigma.
            # vfsa_s: very fine sand proportions. vfsa_s: very fine sand sigma.        
            # si_p: silt proportions. si_s: silt sigma.  
            # sep_p: separator proportions. sep_s: separator sigma.

            #print('event mode:', event.mode)            
            #print('----------')
            
            gr_p, vcsa_p, csa_p, sa_p, fsa_p, vfsa_p, si_p, sep_p = event.dep_props(sl_map)
            gr_s, vcsa_s, csa_s, sa_s, fsa_s, vfsa_s, si_s, sep_s = event.dep_sigmas(sl_map)
            t_p = gr_p + vcsa_p + csa_p + sa_p + fsa_p + vfsa_p + si_p + sep_p                        

            gravel_surface = (gr_p / t_p) * dh_map * gaussian_surface(gr_s, cld_map, hw_map)
            very_coarse_sand_surface = (vcsa_p / t_p) * dh_map * gaussian_surface(vcsa_s, cld_map, hw_map)
            coarse_sand_surface = (csa_p / t_p) * dh_map * gaussian_surface(csa_s, cld_map, hw_map)
            sand_surface = (sa_p / t_p) * dh_map * gaussian_surface(sa_s, cld_map, hw_map)
            fine_sand_surface = (fsa_p / t_p) * dh_map * gaussian_surface(fsa_s, cld_map, hw_map)
            very_fine_sand_surface = (vfsa_p / t_p) * dh_map * gaussian_surface(vfsa_s, cld_map, hw_map)
            silt_surface = (si_p / t_p) * dh_map * gaussian_surface(si_s, cld_map, hw_map)            
            separator_surface = (sep_p / t_p) * 1 * event.sep_thickness            

            # Reusing deposition variables for aggradation purposes                       
            # AGGRADATION
            gr_p, vcsa_p, csa_p, sa_p, fsa_p, vfsa_p, si_p, sep_p = event.aggr_props(sl_map)
            gr_s, vcsa_s, csa_s, sa_s, fsa_s, vfsa_s, si_s, sep_s = event.aggr_sigmas(sl_map)
            t_p = gr_p + vcsa_p + csa_p + sa_p + fsa_p + vfsa_p + si_p + sep_p            
            
            gravel_surface += (gr_p / t_p) * aggr_map * gaussian_surface(gr_s, cld_map, hw_map)  # MANUEL            
            very_coarse_sand_surface   += (vcsa_p / t_p) * aggr_map * gaussian_surface(vcsa_s, cld_map, hw_map)    # MANUEL
            coarse_sand_surface   += (csa_p / t_p) * aggr_map * gaussian_surface(csa_s, cld_map, hw_map)    # MANUEL
            sand_surface   += (sa_p / t_p) * aggr_map * gaussian_surface(sa_s, cld_map, hw_map)    # MANUEL
            fine_sand_surface   += (fsa_p / t_p) * aggr_map * gaussian_surface(fsa_s, cld_map, hw_map)    # MANUEL
            very_fine_sand_surface   += (vfsa_p / t_p) * aggr_map * gaussian_surface(vfsa_s, cld_map, hw_map)    # MANUEL
            silt_surface   += (si_p / t_p) * aggr_map * gaussian_surface(si_s, cld_map, hw_map) # MANUEL                        
            separator_surface += (sep_p / t_p) * 1 * event.sep_thickness # MANUEL
            
            # MANUEL: modulate the aggradation mapps in the case of gravel and sand by Gaussians with standard 
            #         deviations defined experimentally to avoid gravel and sand moving up walls of the channel.
            #         This actually works as a way of implementing a smooth cutoff for these material depositions.
            #         The function gaussian_surface defines a Gaussian inside the channel, thus returning zero
            #         only at the channels boarders. To force a quicker fall off (although only reaching zero at the channel's
            #         boarder) we used these experimentally defined standard deviations when accumulating the results of aggradation.             
             
            # ADDED by MANUEL to smooth the aggradation maps due to their low resolutions
            gravel_surface = scipy.ndimage.gaussian_filter(gravel_surface, sigma = 10 / dx)
            very_coarse_sand_surface   = scipy.ndimage.gaussian_filter(very_coarse_sand_surface, sigma = 10 / dx)
            coarse_sand_surface   = scipy.ndimage.gaussian_filter(coarse_sand_surface, sigma = 10 / dx)
            sand_surface   = scipy.ndimage.gaussian_filter(sand_surface, sigma = 10 / dx)
            fine_sand_surface   = scipy.ndimage.gaussian_filter(fine_sand_surface, sigma = 10 / dx)
            very_fine_sand_surface   = scipy.ndimage.gaussian_filter(very_fine_sand_surface, sigma = 10 / dx)
            silt_surface   = scipy.ndimage.gaussian_filter(silt_surface, sigma = 10 / dx)                              

            # CUTTING CHANNEL
            surface = scipy.ndimage.gaussian_filter(np.minimum(surface, channel_surface), sigma = 10 / dx)

            topo[:,:,i*L + 0] = surface

            # DEPOSITING SEDIMENT - superfície acumula gravel (1) + sand (5) + silt (1) + separator (1)
            # Colocar isso com 8 camadas com condicionais (se proporção for zero não soma)  
            surface += gravel_surface
            topo[:,:,i*L + 1] = surface
            surface += very_coarse_sand_surface
            topo[:,:,i*L + 2] = surface
            surface += coarse_sand_surface
            topo[:,:,i*L + 3] = surface
            surface += sand_surface
            topo[:,:,i*L + 4] = surface
            surface += fine_sand_surface
            topo[:,:,i*L + 5] = surface
            surface += very_fine_sand_surface
            topo[:,:,i*L + 6] = surface
            surface += silt_surface
            topo[:,:,i*L + 7] = surface            
            surface += separator_surface
            topo[:,:,i*L + 8] = surface            
                
            sep_type_number = 0
            if (event.mode == 'SEPARATOR'):
                if (event.sep_type == 'BASAL_SURFACE'):
                    sep_type_number = 1
                elif (event.sep_type == 'INVERSION'):
                    sep_type_number = 2
                elif (event.sep_type == 'CONDENSED_SECTION'):
                    sep_type_number = 3                    
                else:
                    raise Exception("Separator type undefined.")

                separator_type[i*L + 8] = sep_type_number


        # TODO: need to improve here
        DEFAULT_CONFIG_CROSS_SECTIONS = []
        CONFIG_FILE = './config.json'
        config_file = open(CONFIG_FILE, 'r')
        config_json = json.load(config_file)
        generateTable = False
        saveAll = True              
        saveLast = False
        saveNone = False

        if saveAll:
            startPrintingAt = 1
        if saveLast:
            startPrintingAt = N-1
        if saveNone:
            startPrintingAt = N

        for i in range(startPrintingAt, N):
            topo_tmp = topo[:,:,:i*L]
            model_tmp = ChannelBelt3D(topo_tmp, xmin, ymin, dx, dx, self.events, self.honestSpecificEvents, separator_type, width, elevation)
            cross_sections = config_json.get('cross_sections', DEFAULT_CONFIG_CROSS_SECTIONS)
            if i == N-1:
                generateTable = True            
            self.generate_cross_sections(cross_sections, model_tmp, 'Saving Point ' + str(i), generateTable, ve)

        # retorna em topo modelo com todas as camadas    
        return ChannelBelt3D(topo, xmin, ymin, dx, dx, self.events, self.honestSpecificEvents, separator_type, width, elevation) # Dennis: added separator_type