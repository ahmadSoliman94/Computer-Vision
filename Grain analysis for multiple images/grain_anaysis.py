import cv2
import glob
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import measure, color, io


############################################ Grain Analysis  ###########################################################

'''
In this code we are going to analyze the grain size. Then the results are saved in a csv file.
It will be done for multiple images. and using watershed algorithm to segment the grains.

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

#  create a function to segment the grains using watershed algorithm.
def grain_segmentation(img):
    
    # ret is the threshold value., thresh is the thresholded image.
    ret1, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
    
    
    # clean up the imges using morphological operations to noise using opening.
    kernel = np.ones((3,3),np.uint8) # define a kernel for erosion and dilation.
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 1) # iterations is the number of times erosion and dilation are applied.
    
    
    # identify sure background area using dilation
    sure_bg = cv2.dilate(opening,kernel,iterations=2) # dilate the image to get the sure background.
    
    # identify sure foreground area using distance transform and thresholding
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3) # distance transform to get the sure foreground. 3 is the mask size.
    
    # threshold the dist transform by 20% its max value. High value like 0.5 will not recognize some grain boundaries.
    ret2, sure_fg = cv2.threshold(dist_transform,0.2*dist_transform.max(),255,0) # 0 is the threshold value, 255 is the max value.
    
    # identify unknown area by subtracting sure foreground from sure background
    sure_fg = np.uint8(sure_fg) # convert sure_fg to 8 bit image.
    unknown = cv2.subtract(sure_bg,sure_fg) # subtract sure_fg from sure_bg to get the unknown region.
    
    # create a marker and label the regions inside. 
    
    '''
    The regions we know for sure (sure foreground) are labelled with any positive integers, 
    but different integers, and the area we don’t know for sure (unknown region) is just left as zero. For markers,
    we use the function: cv2.connectedComponents(). It labels background of the image with 0, then other objects are labelled with integers starting from 1
    '''
    
    ret3, markers = cv2.connectedComponents(sure_fg) # label the sure_fg.
    
    # add 10 to all labels so that sure background is not 0, but 10 because 0 is considered as unknown
    markers = markers+10
    
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0 # mark the region of unknown with zero.
    
    # apply watershed to markers
    markers1 = cv2.watershed(img,markers) # apply watershed to markers.
    
    # color boundaries. 
    img[markers1 == -1] = [255,0,0] # color boundaries in blue. -1 is the value of the boundaries.
    
    
    img2 = color.label2rgb(markers1, bg_label=0) # color the labels.
    
    # show the results.
    cv2.imshow("colored gains",img2)

    cv2.imshow(' colored boundries over original image',img)    
    
    
    # extract properties of the  detected grains.
    regions = measure.regionprops(markers, intensity_image=img) # extract properties of the  detected grains.
    return regions