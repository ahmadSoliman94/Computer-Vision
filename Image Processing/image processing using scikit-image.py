
#PSF
import scipy.stats as st
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import cv2

from skimage import io, color, restoration, img_as_float
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.filters import roberts, sobel, scharr, prewitt, try_all_threshold, threshold_otsu
from skimage import feature
from skimage.morphology import disk
from skimage.filters.rank import entropy



# 1. Resize, rescale

image_path = r'./images/'
img = io.imread(image_path + "test_image.jpg", as_gray=True)


# Rescale, resize image by a given factor. While rescaling image
# gaussian smoothing can performed to avoid anti aliasing artifacts.
img_rescaled = rescale(img, 1.0 / 4.0, anti_aliasing=False)  # Check rescales image size in variable explorer

#Resize, resize image to given dimensions (shape)
img_resized = resize(img, (200, 200),               #Check dimensions in variable explorer
                       anti_aliasing=True)

#Downscale, downsample using local mean of elements of each block defined by user
img_downscaled = downscale_local_mean(img, (4, 3))
plt.imshow(img_downscaled)

#############################################################################################################

# 2. Edge detection

image_path = r'./images/'
img = io.imread(image_path + "cropped_img.jpg", as_gray=True)  #Convert to grey scale
print(img.shape)
#plt.imshow(img, cmap=plt.cm.gray, interpolation='nearest')

edge_roberts = roberts(img) 
#plt.imshow(edge_roberts, cmap=plt.cm.gray, interpolation='nearest') 
edge_sobel = sobel(img) # Sobel edge detection
edge_scharr = scharr(img)  # Scharr is more accurate than Sobel
edge_prewitt = prewitt(img) # Prewitt edge detection


fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True,
                         figsize=(8, 8))
ax = axes.ravel()

ax[0].imshow(img, cmap=plt.cm.gray)
ax[0].set_title('Original image')

ax[1].imshow(edge_roberts, cmap=plt.cm.gray)
ax[1].set_title('Roberts Edge Detection')

ax[2].imshow(edge_sobel, cmap=plt.cm.gray)
ax[2].set_title('Sobel')

ax[3].imshow(edge_scharr, cmap=plt.cm.gray)
ax[3].set_title('Scharr')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()


Another edge filter is Canny. This is not just a single operation. 
It does noise reduction, gradient calculation, and edge tracking among other things. 
Canny creates a binary file, true or false pixels. 


'''
sigma is the standard deviation of the Gaussian filter.
A larger value of sigma results in a more blurred image and can help to reduce noise or small-scale features that may interfere with edge detection, 
but may also lead to loss of detail or smoothing out of edges.
'''

edge_canny = feature.canny(img, sigma=3) 
plt.imshow(edge_canny)


########################################################################################################

# 3. Image deconvolution
# Uses deconvolution to sharpen images. 



img = img_as_float(io.imread("images/BSE_Google_blurred.jpg"))
print(img.shape)



#psf = np.ones((3, 3)) / 9  #point spread function to be used for deconvolution.

#The following page was used as reference to generate the kernel
#https://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm

def gkern(kernlen=21, nsig=2):   
    
    '''
    generates a 2D Gaussian kernel,
    kernlen is the length of the kernel in pixels
    nsig is the standard deviation of the Gaussian function.
    '''

    lim = kernlen//2 + (kernlen % 2)/2 # the parameter lim is used to center the kernel.
    x = np.linspace(-lim, lim, kernlen+1) # np.linspace used to generate a range of values.
    kern1d = np.diff(st.norm.cdf(x)) # np.diff used to calculate the difference between two consecutive elements.
    kern2d = np.outer(kern1d, kern1d) # np.outer used to calculate the outer product of two vectors.
    return kern2d/kern2d.sum() # Normalise the kernel.

psf = gkern(5,3)   #Kernel length and sigma
print(psf)


deconvolved, _ = restoration.unsupervised_wiener(img, psf) # restoration.unsupervised_wiener used to perform deconvolution.
plt.imsave("images/deconvolved.jpg", deconvolved, cmap='gray')

##################################################################################################

# 4. Let's find a way to calculate the area of scratch in would healing assay

#Entropy filter
#e.g. scratch assay where you have rough region with cells and flat region of scratch.
#entropy filter can be used to separate these regions.

img = io.imread("images/scratch.jpg")
print(img.shape)

#Checkout this page for entropy and other examples
#https://scikit-image.org/docs/stable/auto_examples/


entropy_img = entropy(img, disk(3)) # disk(3) is the size of the structuring element used for the local entropy.
plt.imsave("images/entropy_img.jpg", entropy_img, cmap='gray')

#Once you have the entropy iamge you can apply a threshold to segment the image
#If you're not sure which threshold works fine, skimage has a way for you to check all 

'''
fig, ax = try_all_threshold(entropy_img, figsize=(10, 8), verbose=False) # try_all_threshold used to check all thresholding methods.
plt.show()
'''

#Now let us test Otsu segmentation. 

thresh = threshold_otsu(entropy_img)   # threshold_otsu used to find the optimal threshold value.
binary= entropy_img <=thresh  # let us generate a binary image by separating pixels below and above threshold value.
plt.imshow(binary, cmap=plt.cm.gray)
plt.show()
print("The percent white region is: ", (np.sum(binary == 1)*100)/(np.sum(binary == 0) + np.sum(binary == 1)))   # to calculate the percent white region.

#We can do the same exercise on all images in the time series and plot the area to understand cell proliferation over time
