import numpy as np
from skimage import io, img_as_float
import imquality.brisque as brisque

# ================= BRISQUE =================


''' To install imquality
https://pypi.org/project/image-quality/
'''
img0 = img_as_float(io.imread('./noisy_images/BSE.jpg'))
img25 = img_as_float(io.imread('noisy_images/BSE_1sigma_blur.jpg'))
img50 = img_as_float(io.imread('noisy_images/BSE_2sigma_blur.jpg'))
img75 = img_as_float(io.imread('noisy_images/BSE_3sigma_blur.jpg'))
img100 = img_as_float(io.imread('noisy_images/BSE_5sigma_blur.jpg'))
img200 = img_as_float(io.imread('noisy_images/BSE_10sigma_blur.jpg'))


'''
If the BRISQUE score is low then the image is of good quality. 

for example:
we can use it to find the best image from a set of images.
we can give a threshold value (50) and if the score is less than the threshold then the image is of good quality.

'''

# score0 = brisque.score(img0)
# score25 = brisque.score(img25)
# score50 = brisque.score(img50)
# score75 = brisque.score(img75)
# score100 = brisque.score(img100)
# score200 = brisque.score(img200)

# print("BRISQUE Score for 0 blur = ", score0)
# print("BRISQUE Score for 1 sigma blur = ", score25)
# print("BRISQUE Score for 2 sigma blur = ", score50)
# print("BRISQUE Score for 3 sigma blur = ", score75)
# print("BRISQUE Score for 5 sigma blur = ", score100)
# print("BRISQUE Score for 10 sigma blur = ", score200)


# ================= PSNR =================

'''
PSNR in image processing is a metric for the quality of reconstruction of lossy compression codecs (for example, image compression).
is calculated as the logarithm of the mean squared error of the reconstructed image to the original image.
'''

'''
if the PSNR value is high then the image is of good quality.
'''

# Peak signal to noise ratio (PSNR) is Not a good metric.

from skimage.metrics import peak_signal_noise_ratio

psnr_25 = peak_signal_noise_ratio(img0, img25)
psnr_50 = peak_signal_noise_ratio(img0, img50)
psnr_75 = peak_signal_noise_ratio(img0, img75)
psnr_100 = peak_signal_noise_ratio(img0, img100)
psnr_200 = peak_signal_noise_ratio(img0, img200)

print("PSNR for 1 sigma blur = ", psnr_25)
print("PSNR for 2 sigma blur = ", psnr_50)
print("PSNR for 3 sigma blur = ", psnr_75)
print("PSNR for 5 sigma blur = ", psnr_100)
print("PSNR for 10 sigma blur = ", psnr_200)
