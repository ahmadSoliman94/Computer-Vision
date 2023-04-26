import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import measure, color, io
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_ubyte, img_as_float


####################################### Grain Analysis  ########################################

""" 
This script is used to analyze the grain size of a metallic sample. Then the results are saved in a csv file.

steps:
1. Read the image and define the pixel to micron ratio.
2. Denoise the image and threshold  image to split the grains from the boundaries.
3. Clean the image and create a mask of the grains.
4. Label the grains in the masked image.
5. Measure the area of each grain.
6. save the results in a csv file.
"""

# 1. Read the image and define the pixel to micron ratio.


img = cv2.imread('./images/grains2.jpg', 0)  # read the image as a grayscale image.

pixel_to_micron = 0.5  # pixel to micron ratio , 1 pixel = 0.5 micron.

# 2. Denoise the image and threshold  image to split the grains from the boundaries.

# 2.1. using median filter:
# img1 = ndimage.median_filter(img, 3)  # apply median filter to the image. size is the size of the kernel. but here no need to be denoised.

# 2.2. threshold the image:
# plt.hist(img, 100, range=(0, 255))  # plot the histogram of the image. ravel() is used to convert the image to 1D., 


# Change the grey image to binary by thresholding. 
ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) # ret is the threshold value., thresh is the thresholded image.
print(ret) # print the best threshold value., that OTSU determined this value.


# 3. Clean the image becuse some boundries are faint. Then create a mask of the grains.

kernel = np.ones((3, 3), np.uint8)  # define a kernel for erosion and dilation.

'''
# apply opening to remove the noise. opening = erosion + dilation.
erosion is used to removes the white pixels from the boundaries of the foreground object.
diolation is used to add the white pixels to the boundaries of the foreground object.
'''
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)  # iterations is the number of times erosion and dilation are applied.

# 3.1 create a mask of the grains.
mask = opening == 255 # set the white pixels to True and the black pixels to False. 
print(mask)



# 4. Label the grains in the masked image.

'''
We have well seprarated grains. Each grain is labeled with a different integer.
scipy.ndimage.label() labels the connected regions in an integer array. that will be unique.
4-connectedness is used. because we want to label the grains that are connected by their edges. 
'''

s = [[1,1,1],[1,1,1],[1,1,1]] # define the structure for 4-connectedness. 1 is the foreground and 0 is the background.
labels, num = ndimage.label(mask, structure=s) # labels is the labeled image. num is the number of labels. s is the structure for 4-connectedness.

# color the labeled image.
img2 = color.label2rgb(labels, bg_label=0) # bg_label is the background label. it is 0 by default.

print(f"Number of grains: {num}") # print the number of grains.


# 5. Measure the area of each grain.

'''
regionprops() measures a set of properties for each labeled region.
'''

clusters = measure.regionprops(labels) # clusters is a list of properties of each grain.
# print(clusters[0].perimeter) # print the perimeter of the first grain.


# 5.1. Measure the area of each grain.
# for cluster in clusters:
#     area = cluster.area # area of each grain.
#     label = cluster.label # label of each grain.
#     print(f"Label: {label}, Area: {area*pixel_to_micron**2} micron^2") # print the area of each grain in micron^2.


# 6. save the results in a csv file.

# prop_list is a list of properties that we want to measure for each grain.
propList = ['Area',
            'equivalent_diameter', #Added... verify if it works
            'orientation', #Added, verify if it works. Angle btwn x-axis and major axis.
            'MajorAxisLength',
            'MinorAxisLength',
            'Perimeter',
            'MinIntensity',
            'MeanIntensity',
            'MaxIntensity']    

out_file = open('grain_measurements.csv', 'w') # open a csv file to save the results.
out_file.write((',' + ",".join(propList) + '\n')) # write the header of the csv file.

for cluster_props in clusters:
    #output cluster properties to the excel file
    out_file.write(str(cluster_props['Label']))
    for i,prop in enumerate(propList):
        if(prop == 'Area'): 
            to_print = cluster_props[prop]*pixel_to_micron**2   #Convert pixel square to um square
        elif(prop == 'orientation'): 
            to_print = cluster_props[prop]*57.2958  #Convert to degrees from radians
        # elif(prop.find('Intensity') < 0):          # Any prop without Intensity in its name
        #     to_print = cluster_props[prop]*pixel_to_micron
        # else: 
        #     to_print = cluster_props[prop]     #Reamining props, basically the ones with Intensity in its name
        out_file.write(',' + str(to_print))
    out_file.write('\n')
out_file.close()   #Closes the file, otherwise it would be read only. 


# show the images 
cv2.imshow('orginal', img)
cv2.imshow('thresholded image', thresh)
cv2.imshow('After cleaning', opening)
cv2.imshow('labeled image', img2)
cv2.waitKey(0)

io.imshow(mask) # cv2.imshow() can not show the mask image. because it is a boolean image.
# io.imshow(mask[250:280, 250:280]) # Zoom in the image.
io.show()