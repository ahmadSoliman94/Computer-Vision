import cv2
import matplotlib.pyplot as plt
import numpy as np



############################################ ORB ##########################################################

# load image
img = cv2.imread('./images/img.jpg')

# convert image to rgb
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# convert train image to gray scale
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)



# create test image by adding Scale Invaraint and Rotational Invaraint
img_test = cv2.pyrDown(img)
img_test = cv2.pyrDown(img_test)
rows, cols = img_test.shape[:2] # get the shape of the image :2 means only the first two elements of the shape the rows and cols. 

rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), 30, 1) # get the rotation matrix. 30 degree and scale 1
test_image = cv2.warpAffine(img_test, rotation_matrix, (cols, rows)) # apply the rotation matrix to the image

# convert test image to gray scale
test_gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
 
# Display the images
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
ax1.set_title('Train Image')
ax1.imshow(img)

ax2.set_title('Test Image')
ax2.imshow(test_image)

plt.show()


#############################################################################################################

# Initiate ORB detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with ORB
train_keypoints, train_descriptor = orb.detectAndCompute(img_gray, None) # None is mask
test_keypoints, test_descriptor = orb.detectAndCompute(test_gray, None) # None is mask

# draw keypoints location, size and orientation on the train image
keypoints_without_size = np.copy(img) # np.copy() is to make a copy of the image without the keypoints size
keypoints_with_size = np.copy(img) # np.copy() is to make a copy of the image with the keypoints size

cv2.drawKeypoints(img, train_keypoints, keypoints_without_size, color = (0, 255, 0)) # draw keypoints without size on the train image and the color is green
cv2.drawKeypoints(img, train_keypoints, keypoints_with_size, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # draw keypoints with size on the train image


# Display the images
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

ax1.set_title('Train Image with Keypoints size')
ax1.imshow(keypoints_with_size, cmap='gray')

ax2.set_title('Train Image without Keypoints size')
ax2.imshow(keypoints_without_size, cmap='gray')

plt.show()

# print the number of keypoints detected in the train image
print("\nNumber of Keypoints Detected In The Train Image: ", len(train_keypoints))

# print the number of keypoints detected in the test image
print("Number of Keypoints Detected In The Test Image: ", len(test_keypoints))



#############################################################################################################

# create a Brute Force Matcher object.

'''
crossCheck = True means the two features in the two images should match each other
NORM_HAMMING is the distance between the two features is calculated by the hamming distance which is the number of bits that are different between two descriptors.
example: 0101 and 0111 the hamming distance is 1 because only one bit is different.
'''

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True) 

# match the keypoints using Brute Force Matcher
matches = bf.match(train_descriptor, test_descriptor)

# the matches with shorter distance are the ones we want.
matches = sorted(matches, key = lambda x : x.distance) # sort the matches based on the distance between the two features

# draw  matches
results = cv2.drawMatches(img, train_keypoints, test_gray, test_keypoints, matches, test_gray, flags = 2) # 1 means draw only the keypoints in the test image, 2 means draw the keypoints and the lines between them

# Display the best matching points
cv2.imshow('Best Matching Points', results)
cv2.waitKey(0)

# print the number of matching points
print("\nNumber of Matching Keypoints Between The Train and The Test Images: ", len(matches))