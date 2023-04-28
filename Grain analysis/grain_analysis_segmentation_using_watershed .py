import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import measure, color, io
from skimage.segmentation import clear_border


####################################### Grain Analysis & segmentation using Watershed ########################################


'''
Steps:
1. import libraries.
2. Perform Preprocessing: convert image to greyscale.
3. Threshold Processing  using Otsu to seprate the foreground and background of an image. 
4. Clean up the images from noises using Morphological opreations.
5. Grasping the black background and foreground of the images. 
and create Unkown area by calculating the differnce between the sure grounds. 
6. Place markers on local minima.
7. Apply Watershed to markers.
8. save results in csv file.
'''
# 1. read the image and convert it to grayscale:
img1 = cv2.imread("./images/grains2.jpg")
img = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

pixels_to_um = 0.5 # 1 pixel = 500 nm , convert pixel to micron.

# 2. denoise the image and threshold  image to split the grains from the boundaries:
ret1, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# 3. clean up the imges using morphological operations to noise using opening: 

kernel = np.ones((3,3),np.uint8) # define a kernel for erosion and dilation.
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2) # iterations is the number of times erosion and dilation are applied.

opening1 = clear_border(opening) # remove the grains that are touching the border of the image.

'''
the regions at the center of cells is for sure cells, 
The region far away from the cells is for sure background,
we need to extract the area in between (having the same grey level as the cells) using erosion.
But the cells touching the border will get eroded away. 
So separate touching objects, the  best approach would be distance transform and then thresholding.
'''

# identify sure background area

'''
dilating pixes a few times increases cell boundary to background. 
The area in between sure background and foreground is our ambiguous area.
Watershed should find this area for us.
'''
sure_bg = cv2.dilate(opening1,kernel,iterations=2) # dilate the image to get the sure background.

# identify sure foreground area using distance transform and thresholding
dist_transform = cv2.distanceTransform(opening1,cv2.DIST_L2,3) # distance transform to get the sure foreground. 3 is the mask size.

# threshold the dist transform by 20% its max value. High value like 0.5 will not recognize some grain boundaries.
ret2, sure_fg = cv2.threshold(dist_transform,0.2*dist_transform.max(),255,0) # 0 is the threshold value, 255 is the max value.

# find the uncertain region
sure_fg = np.uint8(sure_fg) # convert the sure foreground to 8 bit image.
uncertain_region = cv2.subtract(sure_bg,sure_fg) # subtract the sure foreground from the sure background to get the uncertain region.

# create a marker and label the regions inside it.

'''
for sure region, both foreground and background will be labeled with positive numbers.
uncertain region will be labeled 0.
'''

ret3, markers = cv2.connectedComponents(sure_fg) # label the sure foreground regions with positive numbers.

'''
one problem rightnow is that the entire background pixels is given value 0.
Watershed algorithm considers pixels at the boundaries as unknown, 
for that we will add 10 to all labels so that sure background pixels are not 0, but 10.
'''
markers = markers+10 # add 10 to all labels.

# mark the uncertain region with zero
markers[uncertain_region==255] = 0 # 255 is the value of the uncertain region.

# apply watershed to separate the grains
markers1 = cv2.watershed(img1,markers) # #The boundary region will be marked -1

# color the boundaries with yellow

img1[markers == -1] = [0,255,255]   

img2 = color.label2rgb(markers1, bg_label=0) # color the background with black.


#Now, time to extract properties of detected cells
# regionprops function in skimage measure module calculates useful parameters for each object.
regions = measure.regionprops(markers, intensity_image=img)

#Can print various parameters for all objects
#for prop in regions:
#    print('Label: {} Area: {}'.format(prop.label, prop.area))

#Best way is to output all properties to a csv file
#Let us pick which ones we want to export. 

propList = ['Area',
            'equivalent_diameter', #Added... verify if it works
            'orientation', #Added, verify if it works. Angle btwn x-axis and major axis.
            'MajorAxisLength',
            'MinorAxisLength',
            'Perimeter',
            'MinIntensity',
            'MeanIntensity',
            'MaxIntensity']    
    

output_file = open('image_measurements_watershed.csv', 'w')
output_file.write('Grain #' + "," + "," + ",".join(propList) + '\n') #join strings in array by commas, 
#First cell to print grain number
#Second cell blank as we will not print Label column

grain_number = 1
for region_props in regions:
    output_file.write(str(grain_number) + ',')
    #output cluster properties to the excel file
#    output_file.write(str(region_props['Label']))
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
    grain_number += 1
    
output_file.close()   #Closes the file, otherwise it would be read only. 

# show the images
cv2.imshow("original",img1)
cv2.imshow("opening",opening)
cv2.imshow("opening1",opening1) # show the image after removing the grains that are touching the border of the image.
cv2.imshow("colored gains",img2)
cv2.waitKey(0)

plt.imshow(markers1, cmap='jet') # show the image after applying watershed.
plt.show()

