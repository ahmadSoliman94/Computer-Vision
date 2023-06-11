import cv2
import numpy as np
from skimage import io, img_as_float
from skimage.restoration import denoise_tv_chambolle
from matplotlib import pyplot as plt



# ======================================= Total variation =======================================

''' works for radom noise but as good for salt and pepper noise '''

# read images
img_gaussian_noise = img_as_float(io.imread('./BSE_25sigma_noisy.jpg', as_gray=True))
img_salt_pepper_noise = img_as_float(io.imread('./BSE_salt_pepper.jpg', as_gray=True))


img = img_salt_pepper_noise # change this to img_salt_pepper_noise to see the effect of total variation on salt and pepper noise


# show histogram of the image
plt.hist(img.ravel(), bins=100, range=(0.0, 1.0)) 
plt.show()


# apply total variation filter

'''
The total variation filter is a useful filter for removing noise from an image,
while preserving the edges. 
'''

denoise_img = denoise_tv_chambolle(img, weight=0.1, eps=0.0002,max_num_iter=200) # weight: The weight parameter of the TV-regularization, eps: Relative difference of the value of the cost function that determines the stop criterion. , n_iter_max: Maximum number of iterations used for the optimization.


# show histogram of the image
plt.hist(denoise_img.ravel(), bins=100, range=(0.0, 1.0))
plt.show()


# display results
cv2.imshow('Original', img)
cv2.imshow('Total variation', denoise_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
