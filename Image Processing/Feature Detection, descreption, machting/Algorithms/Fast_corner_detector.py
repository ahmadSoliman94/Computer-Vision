
# import the necessary packages
import cv2
import numpy as np


######################################## FAST corner detection ########################################

# 1. read the image.
img = cv2.imread('./images/house.jpg', 0) # 0: grayscale, 1: color, -1: unchanged

# 2. initialize the FAST object.
fast = cv2.FastFeatureDetector_create()

# 3. find and draw the keypoints.
kp = fast.detect(img, None) # None: mask
img2 = cv2.drawKeypoints(img, kp, None, color=(255,0,0)) # color: blue

# 4. print all default params.
print(f"Threshold: {fast.getThreshold()}")
print(f"nonmaxSuppression: {fast.getNonmaxSuppression()}")
print(f"neighborhood: {fast.getType()}")
print(f"Total Keypoints with nonmaxSuppression: {len(kp)}")

# 5. disable nonmaxSuppression.
fast.setNonmaxSuppression(0) # 0: disable, 1: enable
kp1 = fast.detect(img, None)
print(f"Total Keypoints without nonmaxSuppression: {len(kp1)}")
img3 = cv2.drawKeypoints(img, kp1, None, color=(255,0,0)) # color: blue


# 6. show the image.
cv2.imshow('FAST', img2)
cv2.imshow('FAST_without_nonsuppression', img3)

cv2.waitKey(0)
cv2.destroyAllWindows() 