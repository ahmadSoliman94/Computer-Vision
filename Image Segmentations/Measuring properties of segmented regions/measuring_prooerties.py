from skimage import measure, io, img_as_ubyte
import matplotlib.pyplot as plt
from skimage.color import label2rgb, rgb2gray
import numpy as np
from skimage.filters import threshold_otsu

# ============================= Measure properties of labeled image regions. =====================================

# 1. Load the image 
image = img_as_ubyte(rgb2gray(io.imread("./cast_iron.jpg"))) # Convert to grayscale and convert to 8-bit image

# 2. Threshold the image to obtain binary image using threshold otsu. 
threshold = threshold_otsu(image)


# 3. Label the image regions.

'''
measure.label() function labels the connected regions of an integer array.
Image regions are represented as connected components of pixels with the same integer value.
image < threshold returns a boolean array of the same shape as image, where the pixel values are True if the corresponding pixel values in image are less than threshold, and False otherwise.
connectivity : int, optional - Defines the connectivity of the neighborhood of a pixel. 
imge.ndim returns the number of dimensions of the image.
'''
labeled_image = measure.label(image < threshold, connectivity=image.ndim) # Connectivity 2 means 8-connected pixels

# 4. Display the labeled image
#plt.imshow(labeled_image, cmap='gray')
#plt.show()


# 5. return an RGB image where color-coded labels are painted over the image.

image_label_overlay = label2rgb(labeled_image, image=image) # image_label_overlay is a 3D array

# 6. Display the image_label_overlay
plt.imshow(image_label_overlay)
plt.imsave("labeled_cast_iron.jpg", image_label_overlay) 
plt.show()


# 7. Compute image properties and return them as a pandas-compatible table.

'''
Available properties are:
1. area : int - Number of pixels of region.
2. bbox : tuple, min_row, min_col, max_row, max_col - Bounding box ``(min_row, min_col, max_row, max_col)`` of the region.
3. centroid : array - Centroid coordinate tuple ``(row, col)``.
4. convex_area : int - Number of pixels of convex hull image.
5. convex_image : (H, W) ndarray of bool - Binary convex hull image which has the same size as bounding box.
6. coords : (N, 2) ndarray of int - Coordinate list ``(row, col)`` of the region.
7. eccentricity : float - Eccentricity of the ellipse that has the same second-moments as the region. The eccentricity is the ratio of the focal distance (distance between focal points) over the major axis length.
8. equivalent_diameter : float - The diameter of a circle with the same area as the region.
9. euler_number : int - Euler characteristic of region. Computed as number of objects (= 1) subtracted by number of holes (8-connectivity).
10. extent : float - Ratio of pixels in the region to pixels in the total bounding box. Computed as ``area / (rows * cols)``
11. filled_area : int - Number of pixels of filled region.
12. filled_image : (H, W) ndarray of bool - Binary region image with filled holes which has the same size as bounding box.
.... for more properties, visit https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops
'''

props = measure.regionprops_table(labeled_image, image, 
                          properties=['label',
                                      'area', 'equivalent_diameter',
                                      'mean_intensity', 'solidity'])

# 8. Store the properties in a DataFrame


import pandas as pd
df = pd.DataFrame(props)
print(df.head())

