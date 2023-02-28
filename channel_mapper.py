import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image, ImageDraw

class ChannelMapper:
    """
    Transforms 2D maps to 3D to be used for the gaussian surfaces

    :param crdist: TODO
    :param ds: TODO                
    """

    def __init__(self, xmin, xmax, ymin, ymax, xsize, ysize, downscale = 4, sigma = 2):
        """
        Initialize channel mapper.

        :param xmin: TODO
        :param xmax: TODO
        :param ymin: TODO
        :param ymax: TODO
        :param xsize: grid size in x axis.
        :param ysize: grid size in y axis.
        :param downscale: TODO
        :param sigma: TODO
        """

        self.xmin = xmin
        self.ymin = ymin
        
        self.downscale = downscale
        self.sigma = sigma
        
        self.xsize = int(xsize / downscale)
        self.ysize = int(ysize / downscale)

        self.dx = xmax - xmin
        self.dy = ymax - ymin

        self.width = int(self.dx / self.xsize)
        self.height = int(self.dy / self.ysize)

        self.rwidth = int(self.width/self.downscale)
        self.rheight = int(self.height/self.downscale)

    def __repr__(self):
        """
        TODO

        :return: (string) Information regarding grid size, image size and number of pixels.
        """

        return 'GRID-SIZE: ({};{})\nIMAGE-SIZE: ({};{})\n PIXELS: {}'.format(self.xsize, self.ysize, self.width, self.height, self.width * self.height)

    def map_size(self):
        """
        [DEBUG] TODO

        :return: TODO
        """

        return (self.xsize, self.ysize)

    def post_processing(self, _map):
        """
        TODO

        :param _map: TODO
        :return: TODO
        """

        return self.downsize(self.filter(_map))

    def filter(self, _map):
        """
        TODO

        :param _map: TODO
        :return: TODO
        """

        return scipy.ndimage.gaussian_filter(_map, sigma = self.sigma)

    def downsize(self, _map):
        """
        TODO

        :param _map: TODO
        :return: TODO
        """

        return np.array(Image.fromarray(_map).resize((self.rwidth, self.rheight), Image.BILINEAR))

    def create_maps(self, channel, basin):
        """
        TODO

        :param channel
        :param basin
        :return: TODO
        """

        ch_map = self.create_ch_map(channel)
        # MANUEL's comments on these channel maps
        cld_map = self.create_cld_map(channel)    # cld: centerline distance - distance from a point to the channel's centerline.
        md_map = self.create_md_map(channel)      # md: margin distance - distance from a point to the channel's closest margin.
        cz_map = self.create_z_map(channel)
        bz_map = self.create_z_map(basin)
        sl_map = self.create_sl_map(basin)

        hw_inside = cld_map + md_map
        hw_outside = cld_map - md_map

        hw_inside[np.array(np.logical_not(ch_map).astype(bool))] = 0.0
        hw_outside[np.array(ch_map.astype(bool))] = 0.0
        hw_map = hw_inside + hw_outside            # hw: half width - half width of the channel (stored in all pixels, but it seems to stored 
                                                   # values considering a cross section of the channel considering the closest point 
                                                   # where the margin is perpendicular to the vector formed by that point and the pixel 
                                                   # that stores the corresponding value)

        return (ch_map, cld_map, md_map, cz_map, bz_map, sl_map, hw_map)

    def create_md_map(self, channel):
        """
        Create margin distance

        :param channel: TODO        
        :return: TODO
        """

        xo, yo = channel.margin_offset()

        upper_pixels = self.to_pixels(channel.x + xo, channel.y + yo)
        lower_pixels = self.to_pixels(channel.x - xo, channel.y - yo)

        img = Image.new("1", (self.width, self.height), 1)
        draw = ImageDraw.Draw(img)
        draw.line(upper_pixels, fill=0) 
        draw.line(lower_pixels, fill=0)

        md_map = ndimage.distance_transform_edt(np.array(img), sampling=[self.xsize, self.ysize])

        return self.post_processing(md_map)

    def create_cld_map(self, channel):
        """
        Create centerline distance map.

        :param channel: TODO        
        :return: TODO
        """

        pixels = self.to_pixels(channel.x, channel.y)
        img = Image.new("1", (self.width, self.height), 1)
        draw = ImageDraw.Draw(img)
        draw.line(pixels, fill=0)
    
        cld_map = ndimage.distance_transform_edt(np.array(img), sampling=[self.xsize, self.ysize])

        return self.post_processing(cld_map)

    def create_ch_map(self, channel):
        """
        TODO

        :param channel: TODO        
        :return: TODO
        """

        x, y = channel.x, channel.y
        xo, yo = channel.margin_offset()

        xm = np.hstack(((x + xo), (x - xo)[::-1]))
        ym = np.hstack(((y + yo), (y - yo)[::-1]))

        xy = self.to_pixels(xm, ym)

        img = Image.new("1", (self.width, self.height), 0)
        draw = ImageDraw.Draw(img)
        draw.polygon(xy, fill=1)

        return self.downsize(np.array(img))

    def create_z_map(self, basin):
        """
        TODO

        :param basin: TODO      
        :return: TODO
        """

        x_p = ((basin.x - self.xmin) / self.dx) * self.width

        tck, _ = scipy.interpolate.splprep([x_p, basin.z], s = 0)
        u = np.linspace(0,1,self.width)
        _, z_level = scipy.interpolate.splev(u, tck)

        return self.post_processing(np.tile(z_level, (self.height, 1)))

    def create_sl_map(self, basin):
        """
        TODO

        :param basin: TODO
        :return: TODO
        """

        x_p = ((basin.x - self.xmin) / self.dx) * self.width
        
        tck, _ = scipy.interpolate.splprep([x_p, basin.slope()], s = 0)
        u = np.linspace(0,1,self.width)
        _, z_level = scipy.interpolate.splev(u, tck)

        return self.post_processing(np.tile(z_level, (self.height, 1)))

    def plot_map(self, _map):
        """
        [DEBUG] Debug method to plot the map as a colorbar.

        :param _map: TODO
        """

        plt.matshow(_map)
        plt.colorbar()
        plt.show()

    def to_pixels(self, x, y):
        """
        Method to plot the map as a colorbar.

        :param x: TODO
        :param y: TODO
        :return: TODO
        """

        x_p = ((x - self.xmin) / self.dx) * self.width
        y_p = ((y - self.ymin) / self.dy) * self.height

        xy = np.vstack((x_p, y_p)).astype(int).T
        return tuple(map(tuple, xy))