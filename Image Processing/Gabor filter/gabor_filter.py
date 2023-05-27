import numpy as np
import cv2
import matplotlib.pyplot as plt


#################################### Gabor filter #############################################################


# Gabor filter: 

'''
parameters:
1. ksize: size of the filter.
2. sigma: standard deviation of the gaussian function and its contorl the size of the Gabor envelope. 
3. theta: orientation of the normal to the parallel stripes of a Gabor function.
4. lambd: wavelength of the sinusoidal factor. Increasing the wavelength produces thicker stripes and decreasing the wavelength produces thinner stripes.
5. gamma: spatial aspect ratio.
6. psi: phase offset. it can be used to control the phase of the sinusoidal factor.
7. ktype: type of filter coefficients. It can be CV_32F or CV_64F.
'''

ksize = 5 # size of gabor filter (n, n)
sigma = 3 # standard deviation of the gaussian function
theta = 1*np.pi/4 # orientation of the normal to the parallel stripes of a Gabor function /4 shows horizontal stripes,
lamda = 1*np.pi/4 # wavelength of the sinusoidal factor
gamma = 0.7 # spatial aspect ratio
psi = 0 # phase offset

# create kernel 
kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, psi, ktype=cv2.CV_32F)

# read image
img = cv2.imread('./synthetic.jpg')


# convert to gray scale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

fimg = cv2.filter2D(img_gray, cv2.CV_8UC3, kernel) # filter image cv.CV_8UC3: 8-bit unsigned integer, 3 channels

# resize kernel
kernel_resized = cv2.resize(kernel, (400, 400)) 





# show images
cv2.imshow('kernel', kernel)
cv2.imshow('kernel_resized', kernel_resized)
cv2.imshow('original', img)
cv2.imshow('filtered', fimg)
cv2.waitKey(0)
cv2.destroyAllWindows()