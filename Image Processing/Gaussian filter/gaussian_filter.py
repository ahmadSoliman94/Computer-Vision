import cv2
import numpy as np
from skimage import io, img_as_float
from skimage.filters import gaussian



# =========================================== Gaussian Filter ===========================================

# read images
img_gaussian_noise = img_as_float(io.imread('./BSE_25sigma_noisy.jpg', as_gray=True))
img_salt_pepper_noise = img_as_float(io.imread('./BSE_salt_pepper.jpg', as_gray=True))

img = img_gaussian_noise # choose image

gaussian_kernel = np.array([[1/16, 1/8, 1/16],   #3x3 kernel
                [1/8, 1/4, 1/8],
                [1/16, 1/8, 1/16]])


# apply the filter

'''
ddepth = -1 means the output image will have the same depth as the source.
exapmle: if the source depth is float64. then the output will be float64.
# borderType=cv2.BORDER_CONSTANT means the border will be filled with a constant value.(i.e. black or 0)
'''
conv_using_cv2  = cv2.filter2D(img, -1, gaussian_kernel, borderType=cv2.BORDER_CONSTANT) 


gaussian_using_cv2 = cv2.GaussianBlur(img, (3,3), 0, borderType=cv2.BORDER_CONSTANT) # (3,3) is the kernel size and 0 is the standard deviation in x and y direction. borderType=cv2.BORDER_CONSTANT means the border will be filled with a constant value.(i.e. black or 0)


'''
sigma defines the standard deviation of the gaussian kernel. slightly different from the standard deviation in cv2.GaussianBlur()
mode='constant' means the border will be filled with a constant value.(i.e. black or 0)
cval=0.0 means the constant value is 0.0
'''
gaussian_using_skimage = gaussian(img, sigma=1, mode='constant', cval=0.0)


# display the result
cv2.imshow("Original", img)
cv2.imshow("cv2 filter", conv_using_cv2)
cv2.imshow("Using cv2 gaussian", gaussian_using_cv2)
cv2.imshow("Using skimage", gaussian_using_skimage)

cv2.waitKey(0)
cv2.destroyAllWindows()