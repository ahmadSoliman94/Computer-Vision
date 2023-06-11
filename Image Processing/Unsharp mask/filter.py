from skimage import io
from skimage.filters import unsharp_mask

import matplotlib.pyplot as plt


# =========================== Unsharp Mask ===========================

'''
Unsharp mask enhances edges by subtracting an unsharp (smoothed) version of the image from the original.
Effectively making the filter a high pass filter. 

enhanced image = original + amount * (original - blurred)

Amount of sharpening can be controlled via scaling factor, a multiplication factor
for the sharpened signal. 
'''

#This code shows that unsharp is nothing but original + amount *(original-blurred)
from skimage import io, img_as_float
from skimage.filters import unsharp_mask
from skimage.filters import gaussian

# 
img = img_as_float(io.imread("./einstein_blurred.jpg", as_gray=True)) # read the image as a float between 0 and 1

# gaussian_img = gaussian(img, sigma=1, mode='constant', cval=0.0) # gaussian filter with sigma=1

# img2 = (img - gaussian_img)*1. # amount = 1

# img3 = img + img2 # original + amount * (original - blurred)


# # Show 3 images in one plot
# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 2.5), sharex=True, sharey=True)
# ax = axes.ravel()

# ax[0].imshow(img, cmap="gray")
# ax[0].set_title("Original image")

# ax[1].imshow(img2, cmap="gray")
# ax[1].set_title("Unsharp mask")

# ax[2].imshow(img3, cmap="gray")
# ax[2].set_title("Sharpened image")

# fig.tight_layout()
# plt.show()

unsharped_img = unsharp_mask(img, radius=3, amount=1)


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12, 12))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(img, cmap='gray')
ax1.title.set_text('Input Image')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(unsharped_img, cmap='gray')
ax2.title.set_text('Unsharped Image')

plt.show()