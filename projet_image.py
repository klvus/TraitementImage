import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
import skimage.io
import skimage.color
import skimage.filters
import colorsys as color
#image = (mplimg.imread('4.jpeg').copy()*255).astype(np.uint8)
image = plt.imread('4.jpeg')
plt.imshow(image)
plt.show()


"""gray"""
img_gris = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        img_gris[i, j] = (image[i, j, 0]*1.0 + image[i, j, 1]
                          * 1.0 + image[i, j, 2]*1.0)/3

plt.imshow(img_gris, cmap=plt.cm.gray, vmin=0, vmax=255)
#print(img_gris)
plt.show()


"""blur image"""
blurred_image = skimage.filters.gaussian(img_gris, sigma=15.0)
#print(blurred_image)
plt.imshow(blurred_image, cmap='gray')

plt.show()


"""histo"""

histogram, bin_edges = np.histogram(blurred_image, bins=256, range=(0.0, 1.0))

plt.plot(bin_edges[0:-1], histogram)
plt.title("Grayscale Histogram")
plt.xlabel("grayscale value")
plt.ylabel("pixels")
plt.xlim(0, 1.0)
plt.show()

img_gris2 = img_gris.copy()

"""binary"""
#t = 0.63
#t = 0.4
#t = 0.1   -> image51
t = 0.02
t2 = 0.1
binary_mask = ((blurred_image > t) & (blurred_image < t2))

"""reversed mask"""

mask2 = np.ones((binary_mask.shape[0], binary_mask.shape[1]), dtype=np.uint8)
for i in range(binary_mask.shape[0]):
    for j in range(binary_mask.shape[1]):
        if(binary_mask[i, j] == 1):
            img_gris2[i, j] = 0
            mask2[i, j] = 0

fig, ax = plt.subplots()
plt.imshow(binary_mask, cmap='gray')
plt.show()
#print(binary_mask)

plt.imshow(mask2, cmap='gray')
plt.show()

plt.imshow(img_gris2, cmap='gray')
plt.show()
#print(img_gris2)


"""air de la surface de traitement"""
seuil = 1
imgout = np.zeros(img_gris2.shape)
aire = 0
for i in range(img_gris2.shape[0]):
    for j in range(img_gris2.shape[1]):
        if img_gris2[i,j] >= seuil :
            imgout[i,j] = 1
            aire += 1
print(f"Pour Seuil = {seuil}, aire = {aire} pixels (soit {aire /(img_gris2.shape[0] * img_gris2.shape[1])*100}%)")
plt.imshow(imgout, cmap='gray')
plt.show()

"""histogramme"""
histogramme = [0]*256
for i in range (img_gris2.shape[0]):
    for j in range(img_gris2.shape[1]):
        histogramme[img_gris2[i,j]] += 1
        
        

for i, val in enumerate(histogramme):
    print(f"{i}: {val}")
    
max = 1
for i in range(1,256):
    if(histogramme[i]>histogramme[max]):
        max = i
        
print(max)
x=0
y=0
for i in range (img_gris2.shape[0]):
    for j in range(img_gris2.shape[1]):
        if img_gris2[i,j] >= max :
            x = i
            y = j
            
dom_color = image[x,y]
print(x)
print(y)
print(image[x,y])

#print(color.rgb_to_hsv(image[x,y][0], image[x,y][1], image[x,y][2]))
#print(color.rgb_to_hsv(102, 69, 16))


w, h = 512, 512
data = np.zeros((h, w, 3), dtype=np.uint8)
data[0:512, 0:256] = image[x,y]
plt.imshow(data)
plt.show()


if(dom_color[0] > 92 and dom_color[0] < 112):
    if(dom_color[1] > 59 and dom_color[1] < 79):
        if(dom_color[2] > 6 and dom_color[2] < 26):
            print("Classe Gold")



"""selection
selection = np.zeros_like(image)
selection[mask2] = img_gris[mask2]
"""

#fig, ax = plt.subplots()
#plt.imshow(selection)
#plt.show()
