
import numpy as np
import cv2


################### Image Segmentation using K-Means clustering ###################

# 1. Read the image:	
img = cv2.imread("./images/BSE_Image.jpg")

# 2. Convert MxNx3 image into Kx3 where K=MxN:
img2 = img.reshape((-1,3)) # -1 reshape means, in this case MxN ,3 is the dimension.

# 3. Convert the unit8 values to float:
img2 = np.float32(img2)

'''
4. Define criteria, number of clusters and apply k-means:
Criteria: It is the iteration termination criteria. When this criteria is satisfied, algorithm iteration stops.
a- cv2.TERM_CRITERIA_EPS - stop the algorithm iteration if specified accuracy, epsilon, is reached.
b- cv2.TERM_CRITERIA_MAX_ITER - stop the algorithm after the specified number of iterations, max_iter.
c- cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER - stop the iteration when any of the above condition is met.
Max_iter - An integer specifying maximum number of iterations.In this case it is 10
Epsilon is the accuracy value. If accuracy value is reached, the algorithm stops. In this case it is 1.
'''

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) # criteria is a tuple of 3 parameters.q


# 5. Define the number of clusters (K):
k = 4 

# 6. Define number of attempts: 

""""
- Number of attempts is the number of times the algorithm is executed using different initial labelings.
- The algorithm returns the labels that yield the best compactness. This compactness is returned as output.
- Compactness is a measure that represents how close are the data points in a cluster.
"""

attempts = 10  

# 7. Specify the initial seed locations:

""" 
There are two ways to specify initial seed locations:
1. cv2.KMEANS_PP_CENTERS - It is the default flag. It selects initial centers in a smart way to speed up convergence.
2. cv2.KMEANS_RANDOM_CENTERS - It selects initial centers randomly for k-means clustering.
"""

ret, label, center = cv2.kmeans(img2, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS) # apply kmeans() to get the labels and centers., ret is the compactness

''' 
The output parameters of cv2.kmeans() are:
a- Compactness (ret): It is the sum of squared distance from each point to their corresponding centers.
b- Labels: This is the label array.
c- Centers: This is array of centers of clusters.
'''

# 8. Now convert the center back into uint8, and make original image:
center = np.uint8(center) # convert center to uint8


# 9. Next, we have to access the labels to regenerate the clustered image:
res = center[label.flatten()] # apply the labels to the center to reconstruct the clustered image 

# 10. Reshape the res into the original image dimensions:
res2 = res.reshape((img.shape)) # reshape the result into the original image dimensions.


# 11. save the image:
cv2.imwrite('segmented_image.jpg', res2)


