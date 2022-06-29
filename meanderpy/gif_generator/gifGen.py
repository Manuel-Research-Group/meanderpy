# Code extracted and adapted from https://www.blog.pythonlibrary.org/2021/06/23/creating-an-animated-gif-with-python/

import imageio

import os

#IN_FOLDER_DIRECTORY = "files"
FILE_TYPE = ".png"
GIF_DURATION = 0.2 # 0.3 for 3fps; 0.2 for 5fps; 0.1 for 10fps

filenames = []
#dir = "/"+IN_FOLDER_DIRECTORY+"/"
#filenames = os.listdir("." + dir)
filenames = os.listdir()

# Reorder the files' list according to the creation time
filenames = sorted(filenames, key=os.path.getmtime)

print('filenames: ', filenames)

images = []

for filename in filenames:    
    if filename[-4:] == '.png':
        images.append(imageio.imread(filename))


imageio.mimsave('out.gif', images, duration=GIF_DURATION)