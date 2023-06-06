import cv2
import numpy as np
from scipy.ndimage.filters import convolve
from skimage import io
from skimage.filters import median
from skimage.restoration import denoise_bilateral


# ==================== Bilateral Filter ====================

img_gaussian_noise = cv2.imread('./BSE_25sigma_noisy.jpg', 0)
img_salt_pepper_noise = cv2.imread('./BSE_salt_pepper.jpg', 0)


img = img_salt_pepper_noise

'''
params:
    img: input image
    kernel_size: size of the kernel
    sigma: standard deviation of the spatial Gaussian
    sigma_r: standard deviation of the range Gaussian
    border_type: pixel extrapolation method
'''
bilateral_using_cv2 = cv2.bilateralFilter(img, 5, 20, 100, borderType=cv2.BORDER_CONSTANT)


'''
params:
    img: input image
    sigma_color: standard deviation of the color space
    sigma_spatial: standard deviation of the position

For large sigma_color values the filter becomes closer to gaussian blur.
For small sigma_color values the filter is closer to an edge detector.
'''
bilateral_using_skimage = denoise_bilateral(img, sigma_color=0.05, sigma_spatial=15)



cv2.imshow("Original", img)
cv2.imshow("cv2 bilateral", bilateral_using_cv2)
cv2.imshow("Using skimage bilateral", bilateral_using_skimage)

cv2.waitKey(0)          
cv2.destroyAllWindows() 
