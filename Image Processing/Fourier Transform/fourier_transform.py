import cv2
from matplotlib import pyplot as plt
import numpy as np


# ========================= Fourier Transform =========================

''' Fourier Transform is defined as a mathematical technique that decomposes a signal into its constituent frequencies.'''

# Generate a 2D sine wave
x = np.arange(256) # 1D array of 256 elements
y = np.sin(2*np.pi*x/30) # 1D array of 256 elements (2*pi*x/30 is the sine wave equation)), 30 to control the frequency of the sine wave, 30 to increase the frequency of the sine wave.
y += max(y) # Add the maximum value of y to all elements of y

img = np.array([[y[j]*127 for j in range(256)] for i in range(256)], dtype=np.uint8) # 2D array of 256x256 elements  *127 to make the image brighter j, i are the coordinates of the image-



dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT) # Descrete Fourier Transform (DFT) of the image.

dft_shift = np.fft.fftshift(dft) # Shift the zero-frequency component to the center of the spectrum. spectrum is the frequency domain representation of the image.

# Calculate magnitude spectrum from the DFT
# Added 1 as we may see 0 values and log of 0 is indeterminate
magnitude_spectrum = 20 * np.log((cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))+1) # 20 * np.log is the equation to calculate the magnitude spectrum of the image. [:,:,0] is the real part of the magnitude spectrum and [:,:,1] is the imaginary part of the magnitude spectrum.



# fig = plt.figure(figsize=(12, 12))
# ax1 = fig.add_subplot(2,2,1)
# ax1.imshow(img)
# ax1.title.set_text('Input Image')
# ax2 = fig.add_subplot(2,2,2)
# ax2.imshow(magnitude_spectrum)
# ax2.title.set_text('FFT of image')
# plt.show()


##########################################################################################################


# ========================= Image filters using Fourier Transform =========================

# read the image
img = cv2.imread('./BSE_Google.jpg', 0)

# output image with 3 channels (RGB) 1st is real and 2nd is imaginary and 3rd is magnitude.

''' for fft in opencv we need to convert the image to float32 and then take the dft of the image.'''

dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT) # Descrete Fourier Transform (DFT) of the image.

# shift the zero-frequncy component to the center of the spectrum
dft_shift = np.fft.fftshift(dft) #  spectrum is the frequency domain representation of the image.

# magnitude spectrum: 20*log(1 + magnitude) we add 1 to deal with log(0)

magnitude_spectrum = 20 * np.log((cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))+1) 


# Create High Pass Filter (HPF) mask, center circle is 0, remaining all ones-

''' 
High pass filter is used for edge detection. It helps in finding the edges in an image. 
because low frequencies at center are blocked and only high frequencies are allowed. Edges are high frequency components.
Amplifies noise.
'''

rows, cols = img.shape # get the shape of the image

crow, ccol = int(rows/2), int(cols/2) # get the center of the image

mask = np.ones((rows, cols, 2), np.uint8) # create a mask of ones of the same size as the image

r = 80 # radius of the circle

center = [crow, ccol] # center of the circle

x, y = np.ogrid[:rows, :cols] # get the coordinates of the image

mask_area = (x - center[0])**2 + (y - center[1])**2 <= r*r # create a circle of radius r <= r*r means less than or equal to r*r

mask[mask_area] = 0 # set the area inside the circle to 0


# Create Low Pass Filter (LPF) mask, center circle is 1, remaining all zeros-

'''
Low pass filter is used for blurring and noise reduction. It helps in removing noises from an image.
because high frequencies at center are blocked and only low frequencies are allowed. Edges are high frequency components.

rows, cols = img.shape
crow, ccol = int(rows / 2), int(cols / 2)

mask = np.zeros((rows, cols, 2), np.uint8)
r = 100
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
mask[mask_area] = 1

# Band Pass Filter (BPF) mask, Concentric circle mask, only the points living in concentric circle are ones.

rows, cols = img.shape
crow, ccol = int(rows / 2), int(cols / 2)

mask = np.zeros((rows, cols, 2), np.uint8)
r_out = 80
r_in = 10
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = np.logical_and(((x - center[0]) ** 2 + (y - center[1]) ** 2 >= r_in ** 2),
                           ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= r_out ** 2))
mask[mask_area] = 1

'''

# apply mask and inverse DFT
fshift = dft_shift * mask 


fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1])) # calculate the magnitude spectrum of the image


# inverse shift
f_ishift = np.fft.ifftshift(fshift) # inverse shift the zero-frequency component to the center of the spectrum  

# inverse DFT
img_back = cv2.idft(f_ishift) # inverse DFT of the image

img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1]) # calculate the magnitude spectrum of the image


# plot the images
fig = plt.figure(figsize=(12, 12))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(img, cmap='gray')
ax1.title.set_text('Input Image')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(magnitude_spectrum, cmap='gray')
ax2.title.set_text('FFT of image')
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(fshift_mask_mag, cmap='gray')
ax3.title.set_text('FFT + Mask')
ax4 = fig.add_subplot(2,2,4)
ax4.imshow(img_back, cmap='gray')
ax4.title.set_text('After inverse FFT')
plt.show()

