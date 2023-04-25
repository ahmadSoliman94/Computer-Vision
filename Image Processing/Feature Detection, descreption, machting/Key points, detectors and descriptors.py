import cv2
import numpy as np


#################################### Feature Detection ####################################

# 1. Harris Corner Detection: steps: 
'''
steps: 
1. convert the image to grayscale. 
2. convert the grayscale image to float32. 
3. apply the cornerHarris function. 
4. dilate the result to mark the corners. 
5. display the result.
'''

''' 
cv2.cornerHarris(src, blockSize, ksize, k[, dst[, borderType]]) → dst
Parameters:
blockSize: Neighborhood size.
ksize: Aperture parameter for the Sobel() operator. Sobel() is used to calculate the derivatives.
k : Harris detector free parameter in the equation.
'''

# read the image.
img = cv2.imread("./grains.jpg")

# convert the image to grayscale.
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# convert the grayscale image to float32. because cornerHarris function takes float32 as input.
gray = np.float32(gray)

# apply the cornerHarris function. 
harris = cv2.cornerHarris(gray, 2, 3, 0.04) 

# dilate the result to mark the corners.
# dst = cv2.dilate(harris,None)

# threshold for an optimal value, it may vary depending on the image.
img[harris>0.01*harris.max()]=[0,0,255] # red color.

# display the result.
cv2.imshow('Harris Corners',img)
cv2.waitKey(0)


############################################################################################################

 # 2. Shi-Tomasi Corner Detector: is a good alternative to Harris Corner Detector. because it takes minimum eigenvalue as a criteria.


'''
cv2.goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance[, corners[, mask[, blockSize[, useHarrisDetector[, k]]]]]) → corners
parameters:
maxCorners: maximum number of corners to return. If there are more corners than are found, the strongest of them is returned.
qualityLevel: parameter characterizing the minimal accepted quality of image corners. 
The parameter value is multiplied by the best corner quality measure, which is the minimal eigenvalue.
The corners with the quality measure less than the product are rejected. 
For example, if the best corner has the quality measure = 1500, and the qualityLevel=0.01 , then all the corners with the quality measure less than 15 are rejected. is between 0-1.
minDistance: minimum possible Euclidean distance between the returned corners.
corners: output vector of detected corners.
mask: optional region of interest. If the image is not empty (it needs to have the type CV_8UC1 and the same size as image ), it specifies the region in which the corners are detected.
blockSize: size of an average block for computing a derivative covariation matrix over each pixel neighborhood.
useHarrisDetector: parameter indicating whether to use a Harris detector.
k : free parameter of the Harris detector.


'''
# read the image.
img = cv2.imread('./grains.jpg')

# convert the image to grayscale.
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# apply the goodFeaturesToTrack function.
corners = cv2.goodFeaturesToTrack(gray,50,0.05,10) # 50 is maxCorners, 0.05 is qualityLevel, 10 is minDistance.
corners = np.intp(corners) # convert the corners to int.

# draw the corners on the image.
for i in corners:
    x,y = i.ravel() # ravel() is used to convert 2d array to 1d array.
    cv2.circle(img,(x,y),3,[0,0,255],-1) # 255 is color, -1 is thickness.
    print(x,y) # print the coordinates of the corners.
    
    
    
# display the result.
cv2.imshow('Shi-Tomasi Corners',img)	
cv2.waitKey(0)


############################################################################################################
# 3. FAST Algorithm: is a corner detector which is faster than Harris Corner Detector and Shi-Tomasi Corner Detector.
# is only keypoint detector, not keypoint descriptor.



# read the image.
img = cv2.imread('./grains.jpg', 0) # 0 is for grayscale.

# create the FAST object.
detector = cv2.FastFeatureDetector_create(50)  # detect the corners with threshold value 50. that means if the intensity of the corner is greater than 50, then it is a corner.

# find the keypoints with FAST.
kp = detector.detect(img, None) 

# draw the keypoints on the image.
img2 = cv2.drawKeypoints(img, kp, None, flags=0)

cv2.imshow('Corners',img2)
cv2.waitKey(0)

############################################################################################################

# Feature Descriptors: are used to describe the features.
# 1. ORB: Oriented FAST and Rotated BRIEF. is a fusion of FAST keypoint detector and BRIEF descriptor with many modifications to enhance the performance.
# BRIEF: Binary Robust Independent Elementary Features. is a feature descriptor that computes a binary descriptor with a fixed size and high efficiency.

# read the image.
img = cv2.imread('./grains.jpg', 0) 

# create the ORB object.
orb = cv2.ORB_create(100) # 100 is the number of keypoints.

kp , des = orb.detectAndCompute(img, None) # detect the keypoints and compute the descriptors. # None is the mask.
# print(des) # output is a 2d array. each row is a descriptor. each column is a value of the descriptor.
# print(kp) # output is a list of keypoints. each keypoint is an object of KeyPoint class.

'''
flags:
cv2.DRAW_MATCHES_FLAGS_DEFAULT: default value.
cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG: draw matches in the output image.
cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS: do not draw keypoints.
cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS: draw keypoints with size and orientation. etc. 
'''
img2 = cv2.drawKeypoints(img, kp, None, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS) # draw the keypoints on the image. # None is the output image.  

# display the result.
cv2.imshow('ORB with keypoints',img2)
cv2.waitKey(0)

