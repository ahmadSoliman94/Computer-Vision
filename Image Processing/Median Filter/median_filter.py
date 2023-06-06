import cv2
import numpy as np
from scipy.ndimage.filters import convolve
from skimage import io
from skimage.filters import median
from skimage.morphology import disk

# ========================================== median filter ========================================== # 

# is better than mean filter in removing salt and pepper noise.

# read images

img_gaussian_noise = cv2.imread('./BSE_25sigma_noisy.jpg', 0)
img_salt_pepper_noise = cv2.imread('./BSE_salt_pepper.jpg', 0)

# choose image 
img = img_salt_pepper_noise

# median filter
median_using_cv2 = cv2.medianBlur(img, 3)

median_using_skimage = median(img, disk(3), mode='constant', cval=0.0) # disk(3) is a structuring element

# show images
cv2.imshow('original', img)
cv2.imshow('median_using_cv2', median_using_cv2)
cv2.imshow('median_using_skimage', median_using_skimage)

cv2.waitKey(0)
cv2.destroyAllWindows()