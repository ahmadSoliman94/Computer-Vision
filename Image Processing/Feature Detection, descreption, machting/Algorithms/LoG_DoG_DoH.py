import matplotlib.pyplot as plt
import cv2 

###################################### LoG & DoG & DoH ############################################

''' 
To detect blobs in an image, we can use:
1. Laplacian of Gaussian (LoG): which is taking the Laplacian of Gaussian filter.
2. Difference of Gaussian (DoG): which is taking the difference between two Gaussian filters.
3. Determinant of Hessian (DoH): which is taking the determinant of the Hessian matrix.

- Hessian matrix: is a square matrix of second-order partial derivatives of a scalar-valued function.
'''

# 1. read the image.
img = cv2.imread('./images/images.jpg', 0) # 0: grayscale, 1: color, -1: unchanged

# 2. Set up the detector with default parameters.
params = cv2.SimpleBlobDetector_Params()
print(params)

# 3. Change thresholds.
params.minThreshold = 0 # 0: black, 255: white
params.maxThreshold = 255

# 4. Filter by Area.
params.filterByArea = True # filter by area: is the area of the blob in pixels.
params.minArea = 1 # minArea: The minimum area of a blob to be detected.
params.maxArea = 1000 # maxArea: The maximum area of a blob to be detected.

# 5. Filter by color.
params.filterByColor = False # filter by color: is the mean intensity of the blob.
params.blobColor = 0 # blobColor: The color of the blob to be detected. 0 for dark blobs, 255 for light blobs.

# 6. Filter by Circularity.
params.filterByCircularity = True # filter by circularity: means how much the shape is circle.
params.minCircularity = 0.5 # minCircularity: The minimum circularity of a blob to be detected.
params.maxCircularity = 1.0 # maxCircularity: The maximum circularity of a blob to be detected.

# 7. Filter by Convexity: convexity means how much the shape is convex. 
params.filterByConvexity = True # filter by convexity: is the convexity of blob. 
params.minConvexity = 0.1 # minConvexity: The minimum convexity of a blob to be detected.
params.maxConvexity = 1.0 # maxConvexity: The maximum convexity of a blob to be detected.

# 8. set up the detector with parameters.
detector = cv2.SimpleBlobDetector_create(params)


# 9. Detect blobs.
keypoint = detector.detect(img)
print(f"Number of blobs: {len(keypoint)}")

# 10. Draw detected blobs as red circles.
img_with_keypoints = cv2.drawKeypoints(img, keypoint, None, color=(0,0,255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # Rich keypoints will have size and orientation information of the keypoint.

# 11. Show keypoints.
cv2.imshow('Blobs', img_with_keypoints)
cv2.waitKey(0)

# 12. Save the image.
cv2.imwrite('./images/Blobs.jpg', img_with_keypoints)