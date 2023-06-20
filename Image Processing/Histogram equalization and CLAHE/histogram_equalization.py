import cv2
from skimage import io
from matplotlib import pyplot as plt


# =================== Histogram Equalization ===================

# Read the image
img = cv2.imread("./img.jpeg", 1) # 0: grayscale, 1: color

# Convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) # bgr 2 lab so CLAHE can be applied to the luminance channel


# Split the image into its channels
l, a, b = cv2.split(img_gray)

# plt.hist(l.flatten(), bins=100, range=(0,255)) # flatten() converts the 2D array into a 1D array
# plt.show()

# Apply histogram equalization to the luminance channel
l_eq = cv2.equalizeHist(l)


# Merge the channels back together
img_eq = cv2.merge((l_eq, a, b))

# Convert back to BGR
img_eq = cv2.cvtColor(img_eq, cv2.COLOR_LAB2BGR)

# Apply CLAHE to the luminance channel
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
l_clahe = clahe.apply(l)

# Merge the channels back together
img_clahe = cv2.merge((l_clahe, a, b))

# Convert back to BGR
img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_LAB2BGR)

# Display the images
cv2.imshow("Original", img)
cv2.imshow("Histogram Equalization", img_eq)
cv2.imshow("CLAHE", img_clahe)

cv2.waitKey(0)
cv2.destroyAllWindows()