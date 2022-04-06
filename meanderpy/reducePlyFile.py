# Generates a new PLY file containing the x,y,z coordinates in float instead of double
from asyncio.windows_events import NULL
from re import X
import numpy as np
import struct


def reducePlySize(inFileName, outFileName):
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

reducePlySize('4.ply', '4_OUT.ply')