import matplotlib.pyplot as plt
from skimage import io, img_as_float
from skimage.metrics import peak_signal_noise_ratio
import bm3d
import cv2


# ==================== BM3D ==================== # 

'''
BM3D: is  a technique for attenuation of additive white Gaussian noise in images.
It is based on the block-matching and 3D filtering approach.
'''

# Read image
noisy_img = img_as_float(io.imread("./BSE_25sigma_noisy.jpg", as_gray=True))

# Apply BM3D

'''
parameters:
    noisy_img: input noisy image
    sigma_psd: noise standard deviation, starndard deviation: is a mesure of the spread of a distribution.
    stage_arg: BM3DStages.HARD_THRESHOLDING: is a technique for attenuation of additive white Gaussian noise in images.
    All stages performs both hard thresholding and Wiener filtering. 

'''
denoised_img = bm3d.bm3d(noisy_img, sigma_psd=0.2, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)

# Calculate PSNR

''' 
The Peak Signal-to-Noise Ratio (PSNR) is a metric used to measure the quality of a reconstructed or compressed signal in comparison to the original, unprocessed signal. 
It is commonly used in fields such as image and video processing to assess the fidelity of the processed data.

PSNR is calculated by measuring the ratio of the peak signal power to the power of the noise. 
The higher the PSNR value, the better the quality of the reconstructed signal. The formula for calculating PSNR is as follows:

PSNR = 20 * log10(MAX) - 10 * log10(MSE)

Where:
- MAX represents the maximum possible pixel value of the image or signal (e.g., 255 for an 8-bit grayscale image).
- MSE (Mean Squared Error) is the average squared difference between the original and processed signals.

The PSNR value is typically expressed in decibels (dB) and provides a quantitative measure of the distortion or loss introduced during the signal processing.
 Higher PSNR values indicate less distortion and higher fidelity, while lower values indicate more noticeable distortion and reduced fidelity.

It's important to note that PSNR has some limitations. It is based solely on pixel-level comparison and doesn't always align with human perception of quality. P
SNR is most effective when comparing similar types of signals, such as comparing compressed images with their original counterparts or assessing the quality of video encoding algorithms.

'''

psnr = peak_signal_noise_ratio(img_as_float(noisy_img), denoised_img, data_range=1)

# print PSNR
print(f"PSNR of denoised image: {psnr} dB")


# Display images
cv2.imshow("Original", noisy_img)
cv2.imshow("Denoised", denoised_img)
cv2.waitKey(0)
cv2.destroyAllWindows()