import cv2
import numpy as np
from matplotlib import pyplot as plt


######################################## Denoising and Detection using opencv ########################################

# Averaging filter, Gaussian filter, Median filter, Bilateral filter.
img_path = r"./images/"
img = cv2.imread( img_path +'BSE_Google_noisy.jpg', 1) # 1 is for color image, 0 is for grayscale image, -1 is for unchanged image.
kernel = np.ones((5,5),np.float32)/25 # 5x5 kernel with all elements as 1/25., the purpose of this kernel is to blur the image.
filt_2D = cv2.filter2D(img,-1,kernel)     # filter2D is used to convolve a kernel with an image., -1 is the depth of the output image.
blur = cv2.blur(img,(5,5))   # applying a 5x5 kernel to blur the image.
blur_gaussian = cv2.GaussianBlur(img,(5,5),0)  # Gaussian filter is used to remove the noise from the image., 0 is denoted as the standard deviation in x direction.
median_blur = median = cv2.medianBlur(img,5)  # Median filter is used to remove the salt and pepper noise from the image.

'''
# Bilateral filter is used to remove the noise while preserving the edges., 
# 9 is the diameter of the pixel neighborhood that is used during filtering., 
# 75 is the filter sigma in the color space.
# 75 is the filter sigma in the coordinate space.
'''
# bilateral_blur = cv2.bilateralFilter(img,9,75,75)  

# cv2.imshow("Original", img)
# cv2.imshow("2D filtered", filt_2D)
# cv2.imshow("Blur", blur)
# cv2.imshow("Gaussian Blur", blur_gaussian)
# cv2.imshow("Median Blur", median_blur)
# cv2.imshow("Bilateral", bilateral_blur)
# cv2.waitKey(0)          
# cv2.destroyAllWindows() 

#########################################################################################

# Edge detection:

img = cv2.imread("images/Neuron.jpg", 0) # 0 is for grayscale image.
edges = cv2.Canny(img,100,200) # 100 is the lower threshold, 200 is the upper threshold.

cv2.imshow("Original Image", img)
cv2.imshow("Canny", edges)

cv2.waitKey(0)          
cv2.destroyAllWindows() 