import numpy as np
import cv2
from matplotlib import pyplot as plt


############################################ Image Registration ############################################

'''
Feature detection: is the process of finding points in an image that are interesting.
Feature description: is the process of creating a description that identifies the point.
Feature matching: is the process of finding similar points in different images.
'''

# Brute-Force Matching with ORB Descriptors:

'''
Brute-Force Matcher: is a simple matcher that takes the descriptor of one feature in first set and is matched with
all other features in second set using some distance calculation and find the best match.

- distance calculation: is the method to calculate the distance between two features.
types of distance calculation:
recommended: Hamming distance.

Hamming distance: is the number of bits that are different in the binary representation of two numbers.
1. calculate the XOR of the two features.
2. count the number of set bits in the result.
for example: 1101 ^ 1010 = 0111, so the Hamming distance is 3.

'''

# 1. read the images.
im1 = cv2.imread('./monkey_distorted.jpg') # Image that needs to be registered.
im2 = cv2.imread('./monkey.jpg') # Reference image.

# 2. convert the images to grayscale.
img1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

# 3. create the ORB detector.
orb = cv2.ORB_create(50) # 50 is the number of keypoints.

# 4. find the keypoints and descriptors with ORB.
kp1, des1 = orb.detectAndCompute(img1, None) 
kp2, des2 = orb.detectAndCompute(img2, None)


# 5. create the BFMatcher object.
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING) # Hamming distance.

# 6. match the descriptors.
matches = matcher.match(des1, des2, None) # None is the mask.

# 7. sort the matches by distance.
matches = sorted(matches, key = lambda x:x.distance) # x.distance is the distance between the two features.

# 8. draw the first 10 matches.
img3 = cv2.drawMatches(im1,kp1, im2, kp2, matches[:10], None) # kp1 and kp2 are the keypoints of the two images.

# 9. show matches image. 
cv2.imshow("Matches image", img3)
cv2.waitKey(0)


############################################################################################################

# image registration: 

'''
Image regestration using Homography is a type of feature-based image registration technique.
Homography is a transformation that maps points in one image to their corresponding points in another image. 
It is a 3x3 matrix that can be computed using a set of matched points between the two images.
'''


# create an array of zeros with the same size of the matches array.
points1 = np.zeros((len(matches), 2), dtype=np.float32)  
points2 = np.zeros((len(matches), 2), dtype=np.float32) # 2 is the number of dimensions.

# fill the points1 and points2 arrays with the coordinates of the matched keypoints.
for i, match in enumerate(matches):
    
    # get the coordinates , .pt is the coordinates of the keypoint.
    points1[i, :] = kp1[match.queryIdx].pt # queryIdx is the index of the keypoint in the first set.
    points2[i, :] = kp2[match.trainIdx].pt # trainIdx is the index of the keypoint in the second set. 
    

# find the homography matrix.

'''
RANSAC is commonly used for outlier rejection in problems such as image registration, object recognition, and stereo vision. 
For example, in the context of image registration using homography, 
RANSAC can be used to estimate the best set of matching keypoints, 
by iteratively re-sampling a subset of keypoints and fitting a homography model to the subset.

The algorithm works by randomly selecting a small subset of the observed data and estimating the parameters of the model using only the points in the subset. 
The algorithm then measures the error between the model and the remaining data points, 
identifying which of them are inliers (consistently estimated by the model) and which are outliers (inconsistent with the model).

The process is repeated many times, each time selecting a new random subset of the data, 
and keeping track of the set of inliers that produces the best fit model. 
Once the algorithm has collected enough sets of inliers and corresponding models, 
it selects the set with the largest number of inliers and estimates the model using all of the inliers.
'''

h, mask = cv2.findHomography(points1, points2, cv2.RANSAC) # h is the homography matrix, mask is the mask array.

# use the homography matrix to warp the images.
height, width, channels = im2.shape # get the shape of the image.
im1reg = cv2.warpPerspective(im1, h, (width, height)) # im1reg is the registered image. warpPerspective is used to warp the image.

# print the homography matrix.
print("Homography matrix: \n", h)

# show the registered image.
cv2.imshow("Registered image", im1reg)
cv2.waitKey(0)