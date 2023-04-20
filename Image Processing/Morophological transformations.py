import cv2
import matplotlib.pyplot as plt
import numpy as np


############################################ Morphological Transformations ############################################

# 1. read the image.
img = cv2.imread("images/BSE_Google_noisy.jpg", 0)

# 2. apply the  otsu thresholding.
ret,th = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# 3. define the kernel.
kernel = np.ones((3,3),np.uint8)   # 3x3 kernel with all ones. 

# 4. apply erosion to removes the white pixels from the boundaries of the foreground object.
erosion = cv2.erode(th,kernel,iterations = 1)

# 5. apply dilation to add the white pixels to the boundaries of the foreground object.
dilation = cv2.dilate(erosion,kernel,iterations = 1)

# 6. apply opening: erosion followed by dilation.
opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

# 7. apply closing: dilation followed by erosion.
closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

# 8. appply colsing followed by opening.
closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

# 9. apply opening followed by closing.
opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

# 10. apply the morphological gradient: difference between dilation and erosion of an image.
gradient = cv2.morphologyEx(th, cv2.MORPH_GRADIENT, kernel)

# 11. apply the top hat: difference between input image and opening of the image.
tophat = cv2.morphologyEx(th, cv2.MORPH_TOPHAT, kernel)

# 12. apply the black hat: difference between the closing of the input image and input image.
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

cv2.imshow("Original Image", img)
cv2.imshow("Otsu", th)
# cv2.imshow("Erosion", erosion)
# cv2.imshow("Dilation", dilation)
cv2.imshow("Opening", opening)
cv2.imshow("Closing", closing)
cv2.imshow("Closing followed by opening", opening)
cv2.imshow("Opening followed by closing", closing)
cv2.imshow("Morphological Gradient", gradient)
cv2.imshow("Top Hat", tophat)
cv2.imshow("Black Hat", blackhat)


cv2.waitKey(0)          
cv2.destroyAllWindows() 