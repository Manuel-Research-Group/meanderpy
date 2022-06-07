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

print('filenames: ', filenames)

# Reorder the files' list according to the integers (instead of 1, 10, 11, ...)
numbersFilenames = [] # list of integer containing the numbers
for filename in filenames:
    if filename[-4:] == '.png':
        num = filename.split(".")
        numbersFilenames.append(int(num[0]))

numbersFilenames.sort()

orderedFilenames = []
for filename in numbersFilenames:
    orderedFilenames.append(str(filename)+FILE_TYPE)

images = []

for filename in orderedFilenames:
    #images.append(imageio.imread("./"+dir+filename))
    images.append(imageio.imread(filename))

imageio.mimsave('out.gif', images, duration=GIF_DURATION)