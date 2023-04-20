import cv2
import numpy as np
from matplotlib import pyplot as plt

######################################### Image enhancements #########################################

# 1. Histogram equalization:
img = cv2.imread("./images/Alloy.jpg", 0)
# equ = cv2.equalizeHist(img) # Histogram equalization is used to enhance the contrast of the image.

# plt.hist(equ.flat, bins=100, range=(0,100))
# plt.show()


# 2. Contrast Limited Adaptive Histogram Equalization (CLAHE):
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) # clipLimit is the threshold for contrast limiting., tileGridSize is the size of grid for histogram equalization.
cl1 = clahe.apply(img)



# cv2.imshow("Original Image", img)
# cv2.imshow("Equalized", equ)
# cv2.imshow("CLAHE", cl1)

##########################################################################

# Image thresholding:

# 1. binary thresholding:

plt.hist(cl1.flat, bins =100, range=(0,255))
plt.show()

''' the threshold value is 150, the max value is 185, the thresholding type is binary. 
When the pixel value is greater than the threshold value, it is assigned the max value, else it is assigned 0.
'''
# ret is the threshold value that is used.
ret,thresh1 = cv2.threshold(cl1,185,150,cv2.THRESH_BINARY) # the max value is 150 and 185 is the threshold value that means all pxiels in grey.
ret2,thresh2 = cv2.threshold(cl1,185,255,cv2.THRESH_BINARY_INV) # All thresholded pixels in white


# 2. Otsu's thresholding: automatically calculates the threshold value.
ret3,th3 = cv2.threshold(cl1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # we did cv2.THRESH_BINARY+cv2.THRESH_OTSU because we want to use both the thresholding methods.
#  print(ret3) # to print the threshold value that is used.


#####################################################

# clean up the noisy images: can using median filter, guassian filter or NL means: filter to preserve the edges.

# img1 = cv2.imread("./images/Alloy_noisy.jpg", 0)
blur = cv2.GaussianBlur(cl1,(5,5),0) # 5x5 kernel is used to blur the image, 0 is the standard deviation in x direction.
ret4,th4 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imshow("Original", img)
cv2.imshow("Binary thresholded", thresh1)
cv2.imshow("Inverted Binary thresholded", thresh2)
cv2.imshow("Otsu", th3)
cv2.imshow("OTSU Gaussian cleaned", th4)
print(ret4)




cv2.waitKey(0)          
cv2.destroyAllWindows() 