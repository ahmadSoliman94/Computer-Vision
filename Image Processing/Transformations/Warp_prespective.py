import cv2
import numpy as np
import os


# ================================= Warp Prespective =================================

'''
the perspective transformation is associated with the change in the viewpoint. 
This type of transformation does not preserve parallelism, length, and angle. But they do preserve collinearity and incidence. 
This means that the straight lines will remain straight even after the transformation.
'''


import cv2
import numpy as np

# Read image
image = cv2.imread('./Magazine.jpg')

# Define width and height of the image
width, height = 250, 350

# Define the 4 corner points of the image
pts1 = np.float32([[1050, 270], [1542, 543], [420, 740], [927, 1090]]) # These points are the 4 corner points of the image

# Define the 4 corner points of the output image
pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

# Compute the perspective transform matrix
matrix = cv2.getPerspectiveTransform(pts1, pts2)

# Apply the perspective transformation to the image
output = cv2.warpPerspective(image, matrix, (width, height))

# Draw the points on the image
for x in range(0, len(pts1)):
    cv2.circle(image, (int(pts1[x][0]), int(pts1[x][1])), 5, (0, 0, 255), -1) # Corrected the center points to integers

# Display the image
cv2.imshow('image', image)
cv2.imshow('output', output)
cv2.waitKey(0)
cv2.destroyAllWindows()

