import os
import cv2 
import numpy as np
import matplotlib.pyplot as plt

# ============================== Segment images using K-means ==============================

# Read image
img = cv2.imread('./img.jpg')

# Convert to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Reshape image to a 2D array of pixels and 3 color values (RGB)
pixel_values = img.reshape((-1, 3)) # -1 means the dimension is calculated based on the other dimensions

# Convert to float
pixel_values = np.float32(pixel_values)

print(pixel_values.shape) # (262144, 3)

# Define stopping criteria (the maximum number of iterations or a very small change in the cluster centers)

'''
Stopping criteria: 
1. cv2.TERM_CRITERIA_EPS - stop the algorithm iteration if specified accuracy, epsilon, is reached.
2. cv2.TERM_CRITERIA_MAX_ITER - stop the algorithm after the specified number of iterations, max_iter.
3. cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER - stop the iteration when any of the above condition is met. 
the last condition is required for algorithm to fully complete the task.
'''
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2) 


# Number of clusters (K)
k = 3
_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) # 10 is the number of attempts because sometimes algorithm can fall into local minimum and try other attempts of random centers to find better convergence.

# Convert data into 8-bit values
centers = np.uint8(centers)

# Flatten the labels array 
labels = labels.flatten()

# Convert all pixels to the color of the centroids
segmented_image = centers[labels.flatten()]

# Reshape back to the original image dimension
segmented_image = segmented_image.reshape(img.shape)



# disable only the cluster number  (turn the pixel into black)
masked_image = np.copy(img) # copy the original image
masked_image = masked_image.reshape((-1, 3)) # reshape the image to a 2D array of pixels and 3 color values (RGB)
cluster = 1 # the cluster number to disable
masked_image[labels == cluster] = [0, 0, 0] # turn the pixel into black
masked_image = masked_image.reshape(img.shape) # reshape the image back to the original 

# Show the multiple images using subplot and adjust the size of the images
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (20, 10))
ax1.set_title('Original Image')
ax1.imshow(img)
ax2.set_title('Segmented Image')
ax2.imshow(segmented_image)
ax3.set_title('Masked Image')
ax3.imshow(masked_image)
plt.show()




