# Histogram based segmentation:

import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage as nd

from skimage import io
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_ubyte, img_as_float




img_path = r"./images/"
img = io.imread( img_path + "BSE_Google_noisy.jpg")
plt.imshow(img, cmap=plt.cm.gray, interpolation='nearest')  
# plt.show()

################################################################

# 1. clean the noise using NL-means filter:

float_img = img_as_float(img) # convert to float
sigma_est = np.mean(estimate_sigma(float_img)) # estimate the standard deviation of the noise

denoise_img = denoise_nl_means(float_img, h=1.15 * sigma_est, fast_mode=True,  # denoise the image
                               patch_size=5, patch_distance=3)
                           
denoise_img_as_8byte = img_as_ubyte(denoise_img) # convert the image to 8-bit unsigned integer.
# plt.imshow(denoise_img_as_8byte, cmap=plt.cm.gray, interpolation='nearest') # nearest uses the nearest pixel value.
# plt.show()


################################################################

# 2. to look at the histogram to see howmany peaks we have, and then we can decide how many classes we need to segment the image.

'''
.flat is used to flatten 2d dimension to 1d dimension, because the hist function only accepts 1d array.
bins = 100 means that we want to divide the range of the data into 100 bins.
range = (0,255) means that we want to look at the data between 0 and 255.
'''
# plt.hist(denoise_img_as_8byte.flat, bins=100, range=(0,255 )) 
# plt.show()


segm1 = (denoise_img_as_8byte <= 57) # seqm1 is mean that the pixel value is less than 57.
segm2 = (denoise_img_as_8byte > 57) & (denoise_img_as_8byte <= 110) 
segm3 = (denoise_img_as_8byte > 110) & (denoise_img_as_8byte <= 210)
segm4 = (denoise_img_as_8byte > 210)

# to show all these images in single visualization
#Construct a new empty image with same shape as original except with 3 layers.

'''
denoise_img_as_8byte.shape[0] means that we want to get the first dimension of the array.
denoise_img_as_8byte.shape[1] means that we want to get the second dimension of the array.
3 means that we want to have 3 layers.
'''
all_segments = np.zeros((denoise_img_as_8byte.shape[0], denoise_img_as_8byte.shape[1], 3)) # de

all_segments[segm1] = (1,0,0) # red
all_segments[segm2] = (0,1,0) # green
all_segments[segm3] = (0,0,1) # blue
all_segments[segm4] = (1,1,0) # yellow
plt.imshow(all_segments)
plt.show()

################################################################

# 3. Lot of yellow dots, red dots and stray dots. how to clean ?
# We can use binary opening and closing operations. 

'''
binary opening is used to remove small objects from the background of an image and takes care of isolated pixels within the window
binary closing is used to remove small holes from the foreground of an image and takes care of isolated holes within the defined window
'''
segm1_opened = nd.binary_opening(segm1, np.ones((3,3))) # np.ones((3,3)) means that we want to use a 3x3 matrix.
segm1_closed = nd.binary_closing(segm1_opened, np.ones((3,3)))

segm2_opened = nd.binary_opening(segm2, np.ones((3,3)))
segm2_closed = nd.binary_closing(segm2_opened, np.ones((3,3)))

segm3_opened = nd.binary_opening(segm3, np.ones((3,3)))
segm3_closed = nd.binary_closing(segm3_opened, np.ones((3,3)))

segm4_opened = nd.binary_opening(segm4, np.ones((3,3)))
segm4_closed = nd.binary_closing(segm4_opened, np.ones((3,3)))


all_segments_cleaned = np.zeros((denoise_img_as_8byte.shape[0], denoise_img_as_8byte.shape[1], 3)) 

all_segments_cleaned[segm1_closed] = (1,0,0)
all_segments_cleaned[segm2_closed] = (0,1,0)
all_segments_cleaned[segm3_closed] = (0,0,1)
all_segments_cleaned[segm4_closed] = (1,1,0)

plt.imshow(all_segments_cleaned)
plt.show()
