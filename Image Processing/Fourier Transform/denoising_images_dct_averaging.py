import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.fftpack import dct, idct

from os import listdir


# ========================= Image denoising using DCT avergaing =========================

# path to the images
images_path = './noise_images/'

# load the images
filenames = listdir(images_path)

# sort the images
# filenames.sort()


# list to store the images
images = []

# loop over the filenames
for filename in filenames:

    # store the image in the list
    images.append((cv2.imread(images_path + filename, 0)).astype(np.float32))


# get the width and height of the images
width, height = images[0].shape


# Apply the weighted average to images and corresponding DCT images, respectively. 

avg_image = np.zeros((width, height), np.float32)
avg_dct = np.zeros((width, height), np.float32)


# loop over the images
for i in range(len(images)):

    avg_img = cv2.addWeighted(avg_image, i/(i+1.0), images[i], 1/(i+1.0), 0)  # orginal image average
    dct_avg_img = cv2.addWeighted(avg_dct, i/(i+1.0), dct(images[i]), 1/(i+1.0), 0)  # DCT average


# inverse DCT
reverse_img  = idct(dct_avg_img) # convert the image back to spatial domain


# save the images
plt.imsave("noise_images/00-dct_averaged_img.jpg", reverse_img, cmap="gray")
plt.imsave("noise_images/00-averaged_img.jpg", avg_img, cmap="gray")


fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(images[0], cmap='gray')
ax1.title.set_text('Input Image 1')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(images[1], cmap='gray')
ax2.title.set_text('Input Image 2')

ax3 = fig.add_subplot(2,2,3)
ax3.imshow(avg_img, cmap='gray')
ax3.title.set_text('Average of Images')
ax4 = fig.add_subplot(2,2,4)
ax4.imshow(reverse_img, cmap='gray')
ax4.title.set_text('Image from DCT average')
plt.show()