# Generates a new PLY file containing the x,y,z coordinates in float instead of double
def reducePlySize(inFileName, outFileName):
    try:
        # First part: read (and write) the ply header as text file
        with open(inFileName, "r", encoding="Latin-1") as inFile:
            #with open(outFileName, "w") as outFile:
            line = ''
            
            while(line != 'end_header\n'):
                line = inFile.readline()
                if(line == 'property double x'):
                    print('hello')
                print(line)            

    except IOError:
        print("Error. Could not read file ", inFileName)

reducePlySize('4.ply', '4_OUT.ply')