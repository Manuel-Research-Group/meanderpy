from PIL import Image

#im1 = Image.open('3.jpg')
#im1 = im1.convert('RGBA')
#im1.save('3.png')

FILE_NAME = '3.png'

im1 = Image.open(FILE_NAME)
pix = im1.load()
height, width = im1.size

for i in range(height):
    for j in range(width):
        if pix[i,j][0] >= 240 and pix[i,j][1] >= 240 and pix[i,j][2] >= 240:
            pix[i,j] = (255, 255, 255, 0)

im1.save(FILE_NAME)

#print(pix[1,1][3])
