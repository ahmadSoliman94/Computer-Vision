import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage as nd 

from skimage import exposure
from skimage import io, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.segmentation import random_walker



 ############################################### Random Walker Segmentation ###############################################
 
# convert the image to float: 
img_path = r"./images/"
img = img_as_float(io.imread(img_path +"Alloy_noisy.jpg"))

# plt.hist(img.flat, bins=100, range=(0, 1))
# plt.show() 

#############################################################################################

 # denoise the image using NL-means filter: 
 
sigma_est = np.mean(estimate_sigma(img))
denoise_img = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=True, 
                               patch_size=5, patch_distance=3, channel_axis=None)
                           
# plt.hist(denoise_img.flat, bins=100, range=(0, 1)) 
# plt.show()


##############################################################################################

# Histogram Equalization:
eq_img = exposure.equalize_adapthist(denoise_img)
# plt.imshow(eq_img, cmap='gray')
# # plt.hist(eq_img.flat, bins=100, range=(0, 1))
# plt.show()

# Let us stretch the hoistogram between 0.7 and 0.95, The range of the binary image spans over (0, 1).

##############################################################################################

# create markers for the random walker algorithm:
markers = np.zeros(img.shape, dtype=np.uint) # markers is an array of zeros with same shape as the image.

markers[(eq_img < 0.8) & (eq_img > 0.7)] = 1 # if the pixel value is between 0.7 and 0.8, then we set the value of the marker to 1.
markers[(eq_img > 0.85) & (eq_img < 0.99)] = 2 # if the pixel value is between 0.85 and 0.99, then we set the value of the marker to 2.
# plt.imshow(markers)

##############################################################################################

# Apply the random walker algorithm:

'''
labels: is the segmented image.
beta: is to control the influence of the deffusion term.
Deffusion in image segmentation is the process of smoothing the image. by removing the noise or unwanted details.
mode: is the mode of the algorithm. there are two modes: 'cg_mg' and 'bf'. cg_mg is the conjugate gradient method and bf is the brute force method.
'''
labels = random_walker(eq_img, markers, beta=10, mode='bf') 
plt.imsave("images/markers.jpg", markers)

segm1 = (labels == 1) # segm1 is the segmented image where the pixel value is 1.
segm2 = (labels == 2) # segm2 is the segmented image where the pixel value is 2.
all_segments = np.zeros((eq_img.shape[0], eq_img.shape[1], 3)) # to combine the two segmented images, we create an array of zeros with the same shape as the image.

all_segments[segm1] = (1,0,0)
all_segments[segm2] = (0,1,0) 

# plt.imshow(all_segments)

##############################################################################################

# remove morphological noise.

segm1_closed = nd.binary_closing(segm1, np.ones((3,3)))
segm2_closed = nd.binary_closing(segm2, np.ones((3,3)))

all_segments_cleaned = np.zeros((eq_img.shape[0], eq_img.shape[1], 3)) 

all_segments_cleaned[segm1_closed] = (1,0,0)
all_segments_cleaned[segm2_closed] = (0,1,0)

plt.imshow(all_segments_cleaned) 
plt.imsave("images/random_walker.jpg", all_segments_cleaned)

plt.show()