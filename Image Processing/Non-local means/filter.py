import cv2
import numpy as np
from skimage import io, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma



# ======================================= Non-local means =======================================

'''
The non-local means algorithm replaces the value of a pixel by an average of a selection of other pixels values: small patches centered on 
the other pixels are compared to the patch centered on the pixel of interest, 
and the average is performed only for pixels that have patches close to the current patch. 
This leads to better denoising performances, at the expense of computation time and memory consumption.
'''

# read images
img_gaussian_noise = img_as_float(io.imread('./BSE_25sigma_noisy.jpg', as_gray=True))
img_salt_pepper_noise = img_as_float(io.imread('./BSE_salt_pepper.jpg', as_gray=True))


img = img_gaussian_noise # change this to img_salt_pepper_noise to see the effect of non-local means on salt and pepper noise

# estimate the noise standard deviation from the noisy image
sigma_est = np.mean(estimate_sigma(img)) # estimate_sigma() returns the average estimated noise standard deviation across color channels.


# apply non-local means filter

'''
When the argument is False, a spatial Gaussian weighting is applied 
to the patches when computing patch distances. When fast_mode is True a 
faster algorithm employing uniform spatial weighting on the patches is applied.

Larger h allows more smoothing between disimilar patches.
'''
denoise_img = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=True, patch_size=5, patch_distance=3) # patch_size: size of patches used for denoising. Should be odd. , patch_distance: Maximum distance in pixels where to search patches used for denoising.


# display results
cv2.imshow('Original', img)
cv2.imshow('Non-local means', denoise_img)
cv2.waitKey(0)
cv2.destroyAllWindows()