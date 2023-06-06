import cv2
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage.filters import convolve
from skimage import io, img_as_float



# =========================================== Image Convolution ===========================================

# read images

img_gaussian_noise = img_as_float(io.imread('./BSE_25sigma_noisy.jpg', as_gray=True)) 
img_salt_pepper_noise = img_as_float(io.imread('./BSE_salt_pepper.jpg', as_gray=True))

img = img_salt_pepper_noise # choose image

# covolving the image with a normalized box filter

'''
Box filter is a filter that is used for blurring and smoothing an image.
it takes the average of all the pixels under the kernel area and replaces the central element with this average.
'''

# define the kernel
kernel = np.ones((5,5),np.float32)/25 # Averaging filter of 5x5 all numbers are  add to one.

# gaussian filter
gaussian_kernel = np.array([[1/16, 1/8, 1/16],
                           [1/8, 1/4, 1/8],
                           [1/16, 1/8, 1/16]])  # Gaussian filter of 3x3

# laplacian filter
laplacian_kernel = np.array([[0, 1, 0], 
                            [1, -4, 1],
                            [0, 1, 0]]) # Laplacian filter of 3x3

# gabor filter
gabor = cv2.getGaborKernel((5, 5), 1.4,45, 5, 1) # Gabor filter of 5x5 with 1.4 sigma, 45 degree orientation and 5 frequency , 1 phase offset, Phase offset means the phase difference between the sine and cosine filters.




# apply the filter

'''
ddepth = -1 means the output image will have the same depth as the source.
exapmle: if the source depth is float64. then the output will be float64.

borderType = cv2.BORDER_CONSTANT means the border will be filled with a constant value.(i.e. black or 0)
Border_REPLICATE means the border will be replicated from the edge pixels.
'''

conv_using_cv2 = cv2.filter2D(img, -1, gabor, borderType=cv2.BORDER_CONSTANT) 

conv_using_scipy = convolve2d(img, gabor, mode='same') # mode='same' means the output will be the same size as the input image.

conv_using_scipy2 = convolve(img, gabor, mode='constant', cval=0.0) # mode='constant' adds a constant value to the image. cval=0.0 means the constant value is 0.0


# display the images
cv2.imshow("Original", img)
cv2.imshow("cv2 filter", conv_using_cv2)
cv2.imshow("Using scipy", conv_using_scipy)
cv2.imshow("Using scipy2", conv_using_scipy2)

cv2.waitKey(0)
cv2.destroyAllWindows()
