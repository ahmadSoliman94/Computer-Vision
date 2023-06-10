import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import measure, color, io
from skimage.segmentation import clear_border



########################################### Cell Nuclei analysis using watershed segmentation ##########################################################

"""
This code is to segment the cell nuclei from the background using watershed segmentation.
We will segment th peform analsyis then dump the results in a csv file.
"""


# load image
img = cv2.imread('./Osteosarcoma_01.tif')   


# Extract the blue channel from the image / the blue channel has the best contrast between the nuclei and the background
cells = img[:, :, 0] # 0 is the blue channel

# got this from the metadata of the image
pixels_to_um = 0.454 # 1 pixel = 0.454 um or 454 nm


# threshold the image to binary using Otsu thresholding
ret1, thresh = cv2.threshold(cells, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) 


# Morphological operations to remove small noise - opening
kernel = np.ones((3, 3), np.uint8) # 3x3 kernel with all ones
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2) # opening is just another name of erosion followed by dilation. It is useful in removing noise


# clear the border of the image - border of the image is the noise
opening1 = clear_border(opening)


# identify sure background area - dilate the image
sure_bg = cv2.dilate(opening1, kernel, iterations = 10) # dilate the image to get the background

# identify sure foreground area - distance transform then thresholding
dist_transform = cv2.distanceTransform(opening1, cv2.DIST_L2, 5) # distance transform to get the foreground

ret2, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0) # threshold the distance transform image to obtain sure foreground

# identify unknown region - subtract sure foreground from sure background
sure_fg = np.uint8(sure_fg) # convert sure foreground to uint8
unknown = cv2.subtract(sure_bg, sure_fg) # subtract sure foreground from sure background to get the unknown region


'''
create a marker and label the regions inside. 
For sure region, both foreground and background will be labeled with positive numbers.
Unknown region will be labeled 0.
'''
ret3, markers = cv2.connectedComponents(sure_fg) # label the sure foreground regions with positive numbers

# add ten to all labels so that sure background is not 0, but 10
markers = markers + 10 


# Now, mark the region of unknown with zero
markers[unknown == 255] = 0 


# apply watershed
markers = cv2.watershed(img, markers) # border is marked with -1 , -1 means boundary


# color boundaries with red 
img[markers == -1] = [0, 255, 255] # color the boundary with red


img2 = color.label2rgb(markers, bg_label=0) # color the labels


# regionprops to get the properties of all the regions
regions = measure.regionprops(markers, intensity_image=cells)


# print the properties of all the regions
for prop in regions:
    print('Label: {} Area: {}'.format(prop.label, prop.area))
    
# create a list to hold the results of analysis
propList = ['Area',
            'equivalent_diameter', #Added... verify if it works
            'orientation', #Added, verify if it works. Angle btwn x-axis and major axis.
            'MajorAxisLength',
            'MinorAxisLength',
            'Perimeter',
            'MinIntensity',
            'MeanIntensity',
            'MaxIntensity']    

output_file = open('./cell_measurements.csv', 'w')
output_file.write(',' + ",".join(propList) + '\n') #join strings in array by commas, leave first cell blank
#First cell blank to leave room for header (column names)

for region_props in regions:
    #output cluster properties to the excel file
    output_file.write(str(region_props['Label']))
    for i,prop in enumerate(propList):
        if(prop == 'Area'): 
            to_print = region_props[prop]*pixels_to_um**2   #Convert pixel square to um square
        elif(prop == 'orientation'): 
            to_print = region_props[prop]*57.2958  #Convert to degrees from radians
        elif(prop.find('Intensity') < 0):          # Any prop without Intensity in its name
            to_print = region_props[prop]*pixels_to_um
        else: 
            to_print = region_props[prop]     #Reamining props, basically the ones with Intensity in its name
        output_file.write(',' + str(to_print))
    output_file.write('\n')

# save the image with markers
cv2.imwrite('./watershed_output.jpg', img)

# show the image
cv2.imshow('Opening', opening)
cv2.imshow('Opening1', opening1) # show the image after removing the border
cv2.imshow('Overlay on original image', img)
cv2.imshow('Colored Grains', img2)
cv2.waitKey(0)

plt.imshow(markers, cmap='jet')
plt.show()